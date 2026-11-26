"""Step 3: 'plain' viewer (OpenCV). Generates an orbit video and/or opens an interactive viewer.

  python nerf/viewer_cv.py --video            # writes data/casco/orbit.mp4
  python nerf/viewer_cv.py                    # interactive window (drag = orbit, wheel = zoom, WASD/QE, ESC)
"""
import argparse
import math

import cv2
import numpy as np
import torch

from orbit import OrbitCamera
from render import load_model, render_image

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="data/casco/nerf.pt")
ap.add_argument("--video", action="store_true")
ap.add_argument("--out", default="data/casco/orbit.mp4")
ap.add_argument("--frames", type=int, default=120)
ap.add_argument("--scale", type=float, default=1.0, help="scale of the training resolution")
ap.add_argument("--samples", type=int, default=128)
args = ap.parse_args()

model, ck = load_model(args.ckpt)
meta = ck["meta"]
H, W = int(meta["h"] * args.scale), int(meta["w"] * args.scale)
fovy = 2 * math.atan(0.5 * meta["h"] / meta["fl_y"])
cam = OrbitCamera.from_c2w(meta["frames"][0]["transform_matrix"], (0, 0, 0), fovy)


def draw(cam):
    img = render_image(model, torch.from_numpy(cam.c2w()).cuda(), H, W, cam.focal(H), args.samples)
    return cv2.cvtColor((img * 255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR)


if args.video:
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    y0 = cam.yaw
    for k in range(args.frames):
        cam.yaw = y0 + 2 * math.pi * k / args.frames
        vw.write(draw(cam))
        print(f"\r{k + 1}/{args.frames}", end="", flush=True)
    vw.release()
    print("\nvideo ->", args.out)
else:
    state = {"drag": False, "x": 0, "y": 0}

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state.update(drag=True, x=x, y=y)
        elif ev == cv2.EVENT_LBUTTONUP:
            state["drag"] = False
        elif ev == cv2.EVENT_MOUSEMOVE and state["drag"]:
            cam.orbit(x - state["x"], y - state["y"])
            state.update(x=x, y=y)
        elif ev == cv2.EVENT_MOUSEWHEEL:
            cam.zoom(1 if flags > 0 else -1)

    cv2.namedWindow("NeRF")
    cv2.setMouseCallback("NeRF", on_mouse)
    while True:
        cv2.imshow("NeRF", draw(cam))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        v = 0.08
        cam.move(v * ((k == ord("w")) - (k == ord("s"))), v * ((k == ord("d")) - (k == ord("a"))),
                 v * ((k == ord("e")) - (k == ord("q"))))
    cv2.destroyAllWindows()
