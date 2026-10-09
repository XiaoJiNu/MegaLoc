#!/usr/bin/env python3
"""MegaLoc Recall@1/@10 on VPR-datasets-downloader layout (25m GPS/UTM)."""
import argparse, json, os, re, time
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as tfm
from PIL import Image
from tqdm import tqdm

def parse_utm(path: str):
    # @utm_e@utm_n@...
    parts = Path(path).name.split("@")
    return float(parts[1]), float(parts[2])

def list_images(folder):
    paths = sorted([str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    return paths

@torch.inference_mode()
def extract(model, paths, device, batch=32):
    transform = tfm.Compose([
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        tfm.Resize(size=[322, 322], antialias=True),
    ])
    descs = []
    for i in tqdm(range(0, len(paths), batch), desc="extract"):
        batch_paths = paths[i:i+batch]
        imgs = []
        for p in batch_paths:
            im = Image.open(p).convert("RGB")
            imgs.append(transform(im))
        x = torch.stack(imgs).to(device)
        d = model(x)
        descs.append(d.cpu())
    return torch.cat(descs, 0).numpy()

def recalls(db_desc, q_desc, db_utm, q_utm, ks=(1,10), radius=25.0):
    # L2-normalized already from MegaLoc; cosine = dot
    sims = q_desc @ db_desc.T  # [Q, D]
    top = np.argsort(-sims, axis=1)
    db_utm = np.asarray(db_utm)
    q_utm = np.asarray(q_utm)
    out = {}
    for k in ks:
        hits = 0
        for i in range(len(q_utm)):
            idx = top[i, :k]
            # positives within radius meters (Euclidean on UTM east/north)
            d = np.linalg.norm(db_utm[idx] - q_utm[i], axis=1)
            if np.any(d <= radius):
                hits += 1
        out[f"R@{k}"] = hits / len(q_utm)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help=".../pitts30k with images/{split}/database|queries")
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default="/data/code/reLocation/.hf_cache/gberton_MegaLoc/model.safetensors")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-db", type=int, default=0)
    ap.add_argument("--max-q", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    db_dir = root / "images" / args.split / "database"
    q_dir = root / "images" / args.split / "queries"
    db_paths = list_images(db_dir)
    q_paths = list_images(q_dir)
    if args.max_db: db_paths = db_paths[:args.max_db]
    if args.max_q: q_paths = q_paths[:args.max_q]
    db_utm = np.array([parse_utm(p) for p in db_paths], dtype=np.float64)
    q_utm = np.array([parse_utm(p) for p in q_paths], dtype=np.float64)

    # load model
    import sys
    sys.path.insert(0, "/data/code/reLocation-all/MegaLoc")
    from megaloc_model import MegaLoc
    from safetensors.torch import load_file
    model = MegaLoc()
    sd = load_file(args.weights)
    model.load_state_dict(sd, strict=False)
    model.eval().to(args.device)

    t0 = time.time()
    db_desc = extract(model, db_paths, args.device, args.batch_size)
    q_desc = extract(model, q_paths, args.device, args.batch_size)
    # ensure L2
    db_desc = db_desc / (np.linalg.norm(db_desc, axis=1, keepdims=True) + 1e-12)
    q_desc = q_desc / (np.linalg.norm(q_desc, axis=1, keepdims=True) + 1e-12)
    metrics = recalls(db_desc, q_desc, db_utm, q_utm)
    metrics.update({
        "n_db": len(db_paths),
        "n_q": len(q_paths),
        "split": args.split,
        "radius_m": 25.0,
        "elapsed_s": round(time.time()-t0, 2),
        "dataset_root": str(root),
        "descriptor_dim": int(db_desc.shape[1]),
    })
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / "descriptors.npz", db=db_desc, q=q_desc, db_utm=db_utm, q_utm=q_utm)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
