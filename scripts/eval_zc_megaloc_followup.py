#!/usr/bin/env python
"""Answer the doc's open questions from saved res__*.npz (no GPU needed).

Q1 时序一致性: top-1 命中 db 索引滑窗中值滤波(去跳变错检)后重 snap，误差变化。
Q2 免级联的 top-K 相似度加权位姿平均(检索层插值, 近似级联收益下界)。
Run: /data/software/conda/anaconda3/envs/mv2d/bin/python scripts/eval_zc_megaloc_followup.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_zc_megaloc import SCENES, OUT, seq_frames  # noqa: E402

W = 7  # temporal window (3.5 s at 2 Hz)


def medfilt_idx(x, w):
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.stack([np.median(xp[i:i + w]) for i in range(len(x))])


out = []
for scene, cfg in SCENES.items():
    _, db_t, db_R = seq_frames(cfg["db"])
    for q in cfg["queries"]:
        r = np.load(OUT / f"res__{scene}__{q}.npz")
        order, sim, terr = r["order"], r["sim"], r["terr"]
        top1 = order[:, 0]
        # Q1: median-filter the db index; snap to filtered index
        fi = np.rint(medfilt_idx(top1.astype(float), W)).astype(int)
        _, q_t, q_R = seq_frames(cfg["queries"][q])
        terr_q1 = np.linalg.norm(db_t[fi] - q_t, axis=1)
        # Q2: top-5 similarity-weighted average of db positions
        k = 5
        s = sim[:, :k]
        wgt = np.exp(8.0 * (s - s.max(1, keepdims=True)))  # softmax, scale-free
        wgt /= wgt.sum(1, keepdims=True)
        pos_avg = np.einsum("nk,nkd->nd", wgt, db_t[order[:, :k]])
        terr_q2 = np.linalg.norm(pos_avg - q_t, axis=1)
        m = {"scene": scene, "query": q,
             "base": [float(np.median(terr)), float(np.percentile(terr, 90)), float((terr < 2).mean())],
             "q1_medfilt": [float(np.median(terr_q1)), float(np.percentile(terr_q1, 90)), float((terr_q1 < 2).mean())],
             "q2_topk_avg": [float(np.median(terr_q2)), float(np.percentile(terr_q2, 90)), float((terr_q2 < 2).mean())]}
        out.append(m)
        print(json.dumps(m), flush=True)

(OUT / "followup.json").write_text(json.dumps(out, indent=2))
