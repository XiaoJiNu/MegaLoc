#!/usr/bin/env python
"""MegaLoc official-style inference reproduction on Nordland-Part-2020.

Protocol:
  - winter queries vs summer gallery
  - ImageNet normalize + resize 322x322 (README eval recipe)
  - Recall@K with success iff |frame_index(q) - frame_index(retrieved)| <= 25
  - descriptors L2-normalized [N, 8448]; cosine = matmul

Python: /data/software/conda/anaconda3/envs/mv2d/bin/python
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as tfm
from PIL import Image
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from megaloc_model import MegaLoc  # noqa: E402

DEFAULT_WEIGHTS = Path("/data/code/reLocation/.hf_cache/gberton_MegaLoc/model.safetensors")
DEFAULT_SUMMER = Path("/data/data/reLocationData/nordland/nordland-part-2020/summer")
DEFAULT_WINTER = Path("/data/data/reLocationData/nordland/nordland-part-2020/winter")
DEFAULT_OUT = ROOT / "outputs" / "inference_repro"

FRAME_RE = re.compile(r"(\d+)")


def frame_index(path: Path) -> int:
    m = FRAME_RE.search(path.name)
    if not m:
        raise ValueError(f"cannot parse frame index from {path.name}")
    return int(m.group(1))


def list_images(folder: Path):
    paths = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg"))
    seen = set()
    out = []
    for p in paths:
        if p.name in seen:
            continue
        seen.add(p.name)
        out.append(p)
    out.sort(key=frame_index)
    return out


def build_transform():
    return tfm.Compose(
        [
            tfm.ToTensor(),
            tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tfm.Resize(size=[322, 322], antialias=True),
        ]
    )


def load_model(weights: Path, device: str):
    if not weights.is_file():
        raise FileNotFoundError(
            f"weights not found: {weights}. Place model.safetensors from gberton/MegaLoc there."
        )
    model = MegaLoc()
    state = load_file(str(weights))
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.inference_mode()
def extract_descriptors(model, paths, transform, device, batch_size: int):
    descs = []
    n = len(paths)
    for i in range(0, n, batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        d = model(imgs).float().cpu()
        descs.append(d)
        done = min(i + batch_size, n)
        if (i // batch_size) % 5 == 0 or done == n:
            print(f"  extract [{done}/{n}]", flush=True)
    return torch.cat(descs, dim=0)


def recall_at_k(sim, q_idx, g_idx, ks=(1, 5, 10), tol: int = 25):
    order = np.argsort(-sim, axis=1)
    out = {}
    top1_delta = []
    for k in ks:
        hits = 0
        for qi in range(sim.shape[0]):
            tops = order[qi, :k]
            ok = any(abs(int(q_idx[qi]) - int(g_idx[j])) <= tol for j in tops)
            hits += int(ok)
        out[f"R@{k}"] = hits / sim.shape[0]
    for qi in range(sim.shape[0]):
        j = int(order[qi, 0])
        top1_delta.append(abs(int(q_idx[qi]) - int(g_idx[j])))
    out["mean_top1_abs_delta"] = float(np.mean(top1_delta))
    out["median_top1_abs_delta"] = float(np.median(top1_delta))
    return out, order, np.asarray(top1_delta)


def save_viz(out_dir, q_paths, g_paths, order, q_idx, g_idx, tol, n_viz=5, topk=5):
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas = np.array([abs(int(q_idx[i]) - int(g_idx[order[i, 0]])) for i in range(len(q_paths))])
    picks = []
    picks.append(int(np.argmin(deltas)))
    picks.append(int(np.argsort(deltas)[len(deltas) // 2]))
    picks.append(int(np.argmax(deltas)))
    step = max(1, len(q_paths) // (n_viz + 1))
    for i in range(step, len(q_paths), step):
        if len(picks) >= n_viz:
            break
        if i not in picks:
            picks.append(i)
    picks = picks[:n_viz]

    saved = []
    for qi in picks:
        fig, axes = plt.subplots(1, 1 + topk, figsize=(3 * (1 + topk), 3.2))
        q_img = Image.open(q_paths[qi]).convert("RGB")
        axes[0].imshow(q_img)
        hit0 = abs(int(q_idx[qi]) - int(g_idx[order[qi, 0]])) <= tol
        axes[0].set_title(
            f"Q winter idx={q_idx[qi]}\n(top1 {'HIT' if hit0 else 'MISS'} ±{tol})",
            fontsize=9,
        )
        axes[0].axis("off")
        for t in range(topk):
            gi = int(order[qi, t])
            g_img = Image.open(g_paths[gi]).convert("RGB")
            dlt = abs(int(q_idx[qi]) - int(g_idx[gi]))
            axes[t + 1].imshow(g_img)
            axes[t + 1].set_title(f"#{t+1} summer idx={g_idx[gi]}\n|d|={dlt}", fontsize=9)
            axes[t + 1].axis("off")
            for spine in axes[t + 1].spines.values():
                spine.set_color("green" if dlt <= tol else "red")
                spine.set_linewidth(2)
        fig.suptitle(f"MegaLoc retrieval — Nordland-Part-2020 (tol=±{tol})", fontsize=11)
        fig.tight_layout()
        path = out_dir / f"viz_q{q_idx[qi]:05d}_top{topk}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(str(path))
        print(f"  viz -> {path}", flush=True)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--summer", type=str, default=str(DEFAULT_SUMMER))
    ap.add_argument("--winter", type=str, default=str(DEFAULT_WINTER))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--tol", type=int, default=25)
    ap.add_argument("--n-viz", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    weights = Path(args.weights)

    print(f"[MegaLoc infer] device={device} weights={weights}", flush=True)
    t0 = time.time()
    model = load_model(weights, device)
    print(f"  model loaded in {time.time()-t0:.1f}s", flush=True)

    g_paths = list_images(Path(args.summer))
    q_paths = list_images(Path(args.winter))
    if args.max_images:
        g_paths = g_paths[: args.max_images]
        q_paths = q_paths[: args.max_images]
    print(f"  gallery(summer)={len(g_paths)} queries(winter)={len(q_paths)}", flush=True)
    if len(g_paths) == 0 or len(q_paths) == 0:
        raise RuntimeError("empty gallery or query set")

    transform = build_transform()
    t1 = time.time()
    print("[extract] gallery (summer)", flush=True)
    g_desc = extract_descriptors(model, g_paths, transform, device, args.batch_size)
    print("[extract] queries (winter)", flush=True)
    q_desc = extract_descriptors(model, q_paths, transform, device, args.batch_size)
    print(
        f"  extract done in {time.time()-t1:.1f}s  shapes q={tuple(q_desc.shape)} g={tuple(g_desc.shape)}",
        flush=True,
    )

    sim = (q_desc @ g_desc.T).numpy()
    q_idx = np.array([frame_index(p) for p in q_paths], dtype=np.int64)
    g_idx = np.array([frame_index(p) for p in g_paths], dtype=np.int64)

    metrics, order, top1_delta = recall_at_k(sim, q_idx, g_idx, ks=(1, 5, 10), tol=args.tol)
    payload = {
        "dataset": "nordland_part_2020",
        "protocol": "winter_query_summer_gallery_delta_index",
        "radius_frames": int(args.tol),
        "n_gallery": len(g_paths),
        "n_query": len(q_paths),
        "descriptor_dim": int(q_desc.shape[1]),
        "weights": str(weights),
        "preprocess": "ImageNet_norm+Resize_322x322",
        "device": device,
        "R@1": metrics["R@1"],
        "R@5": metrics["R@5"],
        "R@10": metrics["R@10"],
        "mean_top1_abs_delta": metrics["mean_top1_abs_delta"],
        "median_top1_abs_delta": metrics["median_top1_abs_delta"],
        "gap_vs_paper": (
            "Paper Table 1 is primarily MSLS (not run). "
            "Nordland-Part-2020 uses sequential index +/-25, not GPS seasons; "
            "metrics are not Table-1 comparable."
        ),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)

    torch.save(
        {
            "q_desc": q_desc,
            "g_desc": g_desc,
            "q_idx": torch.from_numpy(q_idx),
            "g_idx": torch.from_numpy(g_idx),
            "q_paths": [str(p) for p in q_paths],
            "g_paths": [str(p) for p in g_paths],
        },
        out_dir / "descriptors.pt",
    )

    print("[viz]", flush=True)
    viz_paths = save_viz(out_dir, q_paths, g_paths, order, q_idx, g_idx, args.tol, n_viz=args.n_viz)
    payload["viz"] = viz_paths
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] metrics={metrics_path} elapsed={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
