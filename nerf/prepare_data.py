"""Step 2: video -> frames -> camera poses (COLMAP via pycolmap) -> normalized transforms.json.

  python nerf/prepare_data.py --video src/videos/casco.mp4 --out data/casco --every 12 --scale 0.5
"""
import argparse
import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import pycolmap

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="src/videos/casco.mp4")
ap.add_argument("--out", default="data/casco")
ap.add_argument("--every", type=int, default=12, help="keep 1 frame out of every N")
ap.add_argument("--scale", type=float, default=0.5, help="scale of the training images")
ap.add_argument("--obj_radius", type=float, default=0.8, help="object radius after normalization (aabb = [-1,1]^3)")
ap.add_argument("--skip_colmap", action="store_true")
args = ap.parse_args()

full = os.path.join(args.out, "frames_full")
imgs_dir = os.path.join(args.out, "images")
sparse = os.path.join(args.out, "sparse")
db = os.path.join(args.out, "database.db")

# ------------------------------------------------------------------ 1) frames
if not args.skip_colmap or not os.path.isdir(full) or not os.listdir(full):  # --skip_colmap reuses the existing frames
    shutil.rmtree(full, ignore_errors=True)
    os.makedirs(full)
    subprocess.run(["ffmpeg", "-v", "error", "-i", args.video, "-vf", f"select='not(mod(n,{args.every}))'",
                    "-vsync", "vfr", "-qscale:v", "2", os.path.join(full, "%04d.jpg")], check=True)
print(len(os.listdir(full)), "frames")

# ------------------------------------------------------------------ 2) COLMAP
if not args.skip_colmap:
    for p in (sparse, db):
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
    os.makedirs(sparse)
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "SIMPLE_RADIAL"
    print("== extracting features")
    pycolmap.extract_features(db, full, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro)
    print("== matching (exhaustive)")
    pycolmap.match_exhaustive(db)
    print("== incremental reconstruction")
    recs = pycolmap.incremental_mapping(db, full, sparse)
    if not recs:
        raise SystemExit("COLMAP could not reconstruct the scene")
    best = max(recs, key=lambda k: recs[k].num_reg_images())
    rec = recs[best]
    rec.write(sparse)
else:
    rec = pycolmap.Reconstruction(os.path.join(sparse))
print(f"== registered {rec.num_reg_images()} images, {rec.num_points3D()} points, "
      f"reprojection error {rec.compute_mean_reprojection_error():.2f}px")

# ------------------------------------------------------------------ 3) poses -> NeRF/OpenGL convention
names, c2ws = [], []
cam = None
for im in sorted(rec.images.values(), key=lambda i: i.name):
    cam = rec.cameras[im.camera_id]
    T = np.eye(4)
    T[:3, :] = im.cam_from_world().matrix()
    c2w = np.linalg.inv(T)
    c2w[:3, 1:3] *= -1  # COLMAP (x right, y down, z forward) -> OpenGL (x right, y up, z backward)
    names.append(im.name)
    c2ws.append(c2w)
c2ws = np.stack(c2ws)

# world with "up" = +Y (average of the cameras' Y axis)
up = c2ws[:, :3, 1].mean(0)
up /= np.linalg.norm(up)
y = np.array([0, 1, 0.0])
v = np.cross(up, y)
s, c = np.linalg.norm(v), up @ y
if s < 1e-8:
    R = np.eye(3)
else:
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
Tw = np.eye(4)
Tw[:3, :3] = R
c2ws = Tw @ c2ws
pts = (R @ np.array([p.xyz for p in rec.points3D.values()]).T).T

# scene center: point closest to all optical axes (least squares)
A, b = np.zeros((3, 3)), np.zeros(3)
for m in c2ws:
    o, d = m[:3, 3], -m[:3, 2]
    P = np.eye(3) - np.outer(d, d)
    A += P
    b += P @ o
center = np.linalg.solve(A, b)
# object radius: 3D points near the center (robust to far-away background points)
dist = np.linalg.norm(pts - center, axis=1)
r_obj = np.percentile(dist, 50)
scale = args.obj_radius / r_obj
c2ws[:, :3, 3] = (c2ws[:, :3, 3] - center) * scale
cam_dist = np.linalg.norm(c2ws[:, :3, 3], axis=1)
print(f"center {center.round(3)}, object radius (median) {r_obj:.3f}, scale {scale:.3f}")
print(f"camera distance to center: min {cam_dist.min():.2f} mean {cam_dist.mean():.2f} max {cam_dist.max():.2f}")

# ------------------------------------------------------------------ 4) undistorted and rescaled images
os.makedirs(imgs_dir, exist_ok=True)
f, cx, cy = cam.focal_length, cam.principal_point_x, cam.principal_point_y
k = cam.params[3] if len(cam.params) > 3 else 0.0
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
frames = []
H0 = W0 = None
for n, m in zip(names, c2ws):
    im = cv2.imread(os.path.join(full, n))
    im = cv2.undistort(im, K, np.array([k, 0, 0, 0.0]))
    H0, W0 = im.shape[:2]
    W, H = int(round(W0 * args.scale)), int(round(H0 * args.scale))
    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    out_name = os.path.splitext(n)[0] + ".png"
    cv2.imwrite(os.path.join(imgs_dir, out_name), im)
    frames.append({"file_path": f"images/{out_name}", "transform_matrix": m.tolist()})

meta = {"w": W, "h": H, "fl_x": f * args.scale, "fl_y": f * args.scale,
        "cx": cx * args.scale, "cy": cy * args.scale, "aabb": 1.0, "frames": frames}
json.dump(meta, open(os.path.join(args.out, "transforms.json"), "w"), indent=1)
print(f"transforms.json: {len(frames)} frames of {W}x{H}, focal {meta['fl_x']:.1f}")
