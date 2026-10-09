#!/usr/bin/env python
"""MegaLoc retrieval-based relocalization on self-collected ZC bags.

复用 reloc3r 评测已抽好的帧（去畸变图 + T_map_cam 位姿，外参链已定稿修正）：
  割草场景  /data/code/reLocation-all/reloc3r/data/zc_fix/<bag>/seq-01
  研发大楼  /data/code/reLocation-all/reloc3r/data/zc_0923/<bag>/seq-01
MegaLoc(gberton, 本地权重, 322x322 ImageNet norm) 提描述子 -> top-K 检索
db 帧 -> top-1 命中帧位姿作为重定位结果，与查询 GT 比平移/旋转误差；
另给 Recall@K@距离 口径。

Run: /data/software/conda/anaconda3/envs/mv2d/bin/python scripts/eval_zc_megaloc.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reproduce_inference import DEFAULT_WEIGHTS, build_transform, extract_descriptors, load_model  # noqa: E402

FIX = Path("/data/code/reLocation-all/reloc3r/data/zc_fix")
B9 = Path("/data/code/reLocation-all/reloc3r/data/zc_0923")
SCENES = {
    "mowing": {
        "db": FIX / "20260921_064638/seq-01",
        "queries": {"same_day_062224": FIX / "20260921_062224/seq-01",
                     "cross_day_081533": FIX / "20260920_081533/seq-01"},
    },
    "building_0923": {
        "db": B9 / "20260923_055425_real_robot_record_external_lz4/seq-01",
        "queries": {"query_060415": B9 / "20260923_060415_real_robot_record_external_lz4/seq-01"},
    },
}
OUT = ROOT / "outputs" / "eval_zc_megaloc"
HIT_D = 2.0  # standard VPR-on-GT hit radius (m), same as reloc3r doc


def seq_frames(d: Path):
    pngs = sorted(d.glob("frame-*.color.png"))
    assert pngs, f"no frames in {d}"
    T = np.stack([np.loadtxt(str(p.with_suffix("").with_suffix(".pose.txt"))) for p in pngs])  # [N,4,4] c2w
    return pngs, T[:, :3, 3], T[:, :3, :3]


def rot_angles(Rq, Rd):
    Rrel = np.einsum("nki,nkj->nij", Rq, Rd)  # Rq^T @ Rd
    cos = (np.trace(Rrel, axis1=1, axis2=2) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


@torch.inference_mode()
def embed(model, paths, transform, device, bs=64):
    imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths]).to(device)
    out = []
    for i in range(0, len(imgs), bs):
        out.append(model(imgs[i:i + bs]).float().cpu())
    return torch.cat(out).numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = load_model(Path(DEFAULT_WEIGHTS), device)
    transform = build_transform()
    all_metrics = []

    for scene, cfg in SCENES.items():
        db_pngs, db_t, db_R = seq_frames(cfg["db"])
        db_desc = embed(model, db_pngs, transform, device)
        print(f"[{scene}] db={len(db_pngs)} frames", flush=True)
        for qname, qdir in cfg["queries"].items():
            q_pngs, q_t, q_R = seq_frames(qdir)
            q_desc = embed(model, q_pngs, transform, device)
            sim = q_desc @ db_desc.T
            order = np.argsort(-sim, axis=1)

            top1 = order[:, 0]
            terr = np.linalg.norm(q_t - db_t[top1], axis=1)
            rerr = rot_angles(q_R, db_R[top1])
            # recall@K: any of top-K db frames within HIT_D of query GT camera center
            dmat = np.linalg.norm(q_t[:, None] - db_t[None, :], axis=2)
            rec = {}
            for k in (1, 5, 10):
                hits = (dmat[np.arange(len(q_t))[:, None], order[:, :k]] < HIT_D).any(1)
                rec[f"R@{k}@{HIT_D:g}m"] = float(hits.mean())
            # top-1 distance of nearest-GT db frame = db sampling floor reference
            nn_gt = dmat.min(1)
            m = {
                "scene": scene, "query": qname, "db": cfg["db"].parent.name,
                "n_db": len(db_pngs), "n_q": len(q_pngs),
                "top1_snap_terr_med_m": float(np.median(terr)),
                "top1_snap_terr_p90_m": float(np.percentile(terr, 90)),
                "top1_snap_rerr_med_deg": float(np.median(rerr)),
                "top1_snap_rerr_p90_deg": float(np.percentile(rerr, 90)),
                **{f"terr<{d}m": float((terr < d).mean()) for d in (1, 2, 5)},
                **rec,
                "gt_nearest_db_frame_med_m": float(np.median(nn_gt)),
                "random_baseline_mean_m": float(dmat.mean()),
            }
            all_metrics.append(m)
            print(json.dumps(m, indent=2), flush=True)

            np.savez_compressed(OUT / f"res__{scene}__{qname}.npz",
                                terr=terr, rerr=rerr, nn_gt=nn_gt,
                                top1=top1, order=order, sim=sim)
            # CDF
            fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
            ax[0].plot(np.sort(terr), np.linspace(0, 1, len(terr)))
            ax[0].set(xlabel="top-1 snap translation err (m)", ylabel="CDF", xscale="log",
                      title=f"{scene}/{qname} terr med={m['top1_snap_terr_med_m']:.2f}")
            ax[1].plot(np.sort(rerr), np.linspace(0, 1, len(rerr)))
            ax[1].set(xlabel="rot err (deg)", ylabel="CDF", xscale="log",
                      title=f"rerr med={m['top1_snap_rerr_med_deg']:.1f}deg")
            fig.tight_layout()
            fig.savefig(OUT / f"cdf__{scene}__{qname}.png", dpi=120)
            plt.close(fig)
            # retrieval grid: 6 sampled queries, top-5 with GT distance
            picks = np.linspace(0, len(q_pngs) - 1, 6).astype(int)
            fig, axes = plt.subplots(len(picks), 6, figsize=(12, 2.4 * len(picks)))
            for r, qi in enumerate(picks):
                axes[r, 0].imshow(Image.open(q_pngs[qi]))
                axes[r, 0].set_title(f"Q #{qi}\nterr {terr[qi]:.2f}m", fontsize=8)
                axes[r, 0].axis("off")
                for t in range(5):
                    gj = order[qi, t]
                    axes[r, t + 1].imshow(Image.open(db_pngs[gj]))
                    d = dmat[qi, gj]
                    axes[r, t + 1].set_title(f"#{t+1} db{gj}\n{d:.2f}m", fontsize=8)
                    for sp in axes[r, t + 1].spines.values():
                        sp.set_color("green" if d < HIT_D else "red")
                        sp.set_linewidth(2)
                    axes[r, t + 1].axis("off")
            fig.suptitle(f"MegaLoc top-5 (green<2m of query GT) — {scene}/{qname}")
            fig.tight_layout()
            fig.savefig(OUT / f"grid__{scene}__{qname}.png", dpi=110)
            plt.close(fig)

    (OUT / "metrics_all.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"[done] {time.time()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
