"""Convert a NeRF 'synthetic Blender' dataset (lego, ship, chair...) to this project's format.

  python nerf/import_blender.py --src C:/ruta/nerf_synthetic/lego --out data/lego
  python nerf/train.py --data data/lego --out data/lego/nerf.pt

Expects <src>/transforms_{train,val,test}.json and the referenced RGBA PNG images.
The alpha channel is kept (training uses a random background, the view uses a white one) and the world is rotated from
Z-up (Blender) to Y-up (viewer).
"""
import argparse
import json
import math
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scale", type=float, default=0.5, help="800x800 -> 400x400 by default")
ap.add_argument("--aabb", type=float, default=1.5, help="the Blender scene fits in ~[-1.5,1.5]^3")
ap.add_argument("--splits", default="train,val")
args = ap.parse_args()

# (x, y, z)_blender -> (x, z, -y): Z-up becomes Y-up
Tw = np.eye(4)
Tw[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]

os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
frames, W = [], None
for split in args.splits.split(","):
    path = os.path.join(args.src, f"transforms_{split}.json")
    if not os.path.exists(path):
        continue
    tj = json.load(open(path))
    for k, f in enumerate(tj["frames"]):
        im = cv2.imread(os.path.join(args.src, f["file_path"] + ".png"), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise SystemExit(f"Could not read {f['file_path']}.png")
        if im.shape[2] == 3:  # no alpha: saved as opaque
            im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
        H0, W0 = im.shape[:2]
        W, H = int(round(W0 * args.scale)), int(round(H0 * args.scale))
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        name = f"{split}_{k:04d}.png"
        cv2.imwrite(os.path.join(args.out, "images", name), im)
        c2w = Tw @ np.array(f["transform_matrix"], dtype=np.float64)
        frames.append({"file_path": f"images/{name}", "transform_matrix": c2w.tolist()})
    focal = 0.5 * W0 / math.tan(0.5 * tj["camera_angle_x"]) * args.scale

if not frames:
    raise SystemExit("No transforms_*.json found in --src")
meta = {"w": W, "h": H, "fl_x": focal, "fl_y": focal, "cx": W / 2, "cy": H / 2,
        "aabb": args.aabb, "frames": frames}
json.dump(meta, open(os.path.join(args.out, "transforms.json"), "w"), indent=1)
print(f"{len(frames)} images of {W}x{H}, focal {focal:.1f}, aabb {args.aabb} -> {args.out}/transforms.json")
