# MegaLoc INFERENCE Reproduction — Implementation Plan

- **Timestamp (Asia/Shanghai):** 20260915_095406
- **Machine:** Legion (machineId 252f7d1f-95fc-44c6-94d1-8f76153ee6af)
- **Code root:** `/data/code/reLocation-all/MegaLoc`
- **Python ONLY:** `/data/software/conda/anaconda3/envs/mv2d/bin/python`
- **Mirror status:** `/home/yr/grok-bot-work/trail_repro/megaloc_infer/`

## Goal

Run MegaLoc official/demo inference end-to-end on Nordland-Part-2020:
load pretrained weights → extract descriptors → retrieve → Recall@1/5/10 (±25 frames) → visualizations.

## Key paths (EN)

| Item | Path |
|------|------|
| Repo | `/data/code/reLocation-all/MegaLoc` |
| Weights (local HF) | `/data/code/reLocation/.hf_cache/gberton_MegaLoc/model.safetensors` (~873 MB) |
| Gallery (summer) | `/data/data/reLocationData/nordland/nordland-part-2020/summer` (1750 PNG) |
| Queries (winter) | `/data/data/reLocationData/nordland/nordland-part-2020/winter` (1750 PNG) |
| Script | `scripts/reproduce_inference.py` |
| Outputs | `outputs/inference_repro/` |
| Docs | `doc/reproduction/` |

## 关键关键路径 (CN)

| 项 | 路径 |
|----|------|
| 代码根 | `/data/code/reLocation-all/MegaLoc` |
| 预训练权重 | `/data/code/reLocation/.hf_cache/gberton_MegaLoc/model.safetensors` |
| 数据 summer/winter | `/data/data/reLocationData/nordland/nordland-part-2020/{summer,winter}` |
| 推理脚本 | `scripts/reproduce_inference.py` |
| 指标/可视化 | `outputs/inference_repro/` |
| 父代理镜像 | `/home/yr/grok-bot-work/trail_repro/megaloc_infer/` |

## Protocol

1. **Load model:** local `MegaLoc()` + `safetensors` state_dict (no HF download if weights exist). Fallback: torch.hub / hf_hub_download.
2. **Preprocess:** ImageNet mean/std, resize **322×322** (official README eval recipe).
3. **Retrieval:** winter = queries, summer = gallery; cosine similarity via L2-normalized descriptors `[N, 8448]`.
4. **Success:** `|frame_index(query) − frame_index(retrieved)| ≤ 25` (Nordland sequential index tolerance; same filenames across seasons: `images-NNNNN.png`).
5. **Metrics:** Recall@1 / @5 / @10; also mean |Δindex| of top-1.
6. **Viz:** ≥3 PNGs of query vs top-1 / top-5 under `outputs/inference_repro/`.
7. **Scale:** try full 1750×1750; if OOM/time, subset ≥200+200 then escalate.

## Non-goals

- Do **not** download full MSLS.
- No reckless sudo/apt; watch `/data` free (~90G).
- Local git commit OK; **no push**.

## Steps checklist

- [ ] Inspect README / hubconf / weights / data
- [ ] Write this plan
- [ ] Minimal deps check (safetensors, torchvision, PIL, matplotlib already in mv2d)
- [ ] Implement `scripts/reproduce_inference.py`
- [ ] Run inference → `metrics.json` + viz PNGs
- [ ] Timestamped completion doc + mirror to trail_repro
- [ ] Local git add/commit if `.git` exists

## Gap note vs paper

Paper Table 1 primary = **MSLS** (skipped). This run uses Nordland-Part-2020 index protocol (±25 frames), **not** GPS-labelled full seasons — numbers are directional, not Table-1 comparable.
