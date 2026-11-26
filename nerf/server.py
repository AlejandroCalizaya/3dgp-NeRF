"""Render server: the OpenGL viewer (C++/Docker) requests frames over TCP and this process
renders them with the NeRF on the GPU. It discovers every trained scene (data/*/nerf.pt)
and the viewer chooses which one to load.

Protocol (little-endian):
  server -> client on connect:
      int32 N scenes; N x 32 bytes (name, padded with \0); int32 current index;
      20 x float32 for the current scene = initial c2w (16, column-major), target (3), fovY rad (1)
  client -> server, every message starts with an int32 type:
      0 = render : int32 W, int32 H, float32 focal, 16 x float32 c2w (column-major)
                   reply: W*H*3 bytes RGB8 (row 0 = top)
      1 = scene  : int32 index
                   reply: 20 x float32 (same as above, for the new scene)
"""
import argparse
import glob
import math
import os
import socket
import struct

import numpy as np
import torch

from render import load_model, render_image

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", default="data", help="folder with one subfolder per scene (each with a nerf.pt)")
ap.add_argument("--scene", default=None, help="initial scene (folder name)")
ap.add_argument("--port", type=int, default=5555)
ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 if the viewer runs in Docker (opens the port to the network)")
ap.add_argument("--max_side", type=int, default=2048, help="maximum W and H accepted from a client")
ap.add_argument("--samples", type=int, default=64)
ap.add_argument("--prune", type=float, default=0.5, help="density threshold used to skip empty space")
args = ap.parse_args()

ckpts = sorted(glob.glob(os.path.join(args.data_root, "*", "nerf.pt")))
if not ckpts:
    raise SystemExit(f"No trained models in {args.data_root}/*/nerf.pt")
names = [os.path.basename(os.path.dirname(p)) for p in ckpts]
print("Available scenes:", ", ".join(names), flush=True)

current = {"idx": -1, "model": None, "hello": None}


def load_scene(idx):
    """Load scene idx on the GPU (freeing the previous one) and return the 20 welcome floats."""
    if current["idx"] != idx:
        current["model"] = None
        torch.cuda.empty_cache()
        print(f"loading scene '{names[idx]}'...", flush=True)
        model, ck = load_model(ckpts[idx], prune=args.prune)
        meta = ck["meta"]
        fovy = 2 * math.atan(0.5 * meta["h"] / meta["fl_y"])
        start = np.array(meta["frames"][0]["transform_matrix"], dtype=np.float32)
        # start from the first training camera, looking at the origin (normalized center)
        current.update(idx=idx, model=model,
                       hello=np.concatenate([start.T.reshape(-1), np.zeros(3, np.float32), [fovy]]).astype("<f4"))
    return current["hello"]


def recv_all(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError
        buf += chunk
    return bytes(buf)


first = names.index(args.scene) if args.scene in names else 0
load_scene(first)

srv = socket.socket()
if os.name == "nt":  # on Windows SO_REUSEADDR lets two servers share the port
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
else:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    srv.bind((args.host, args.port))
except OSError:
    raise SystemExit(f"Port {args.port} is already in use: is another NeRF server open? Close it first.")
srv.listen(1)
print(f"NeRF server listening on {args.host}:{args.port}", flush=True)
while True:
    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("client connected", addr, flush=True)
    try:
        msg = struct.pack("<i", len(names)) + b"".join(n.encode()[:31].ljust(32, b"\0") for n in names)
        conn.sendall(msg + struct.pack("<i", current["idx"]) + current["hello"].tobytes())
        while True:
            (kind,) = struct.unpack("<i", recv_all(conn, 4))
            if kind == 0:
                W, H, focal = struct.unpack("<iif", recv_all(conn, 12))
                c2w = np.frombuffer(recv_all(conn, 64), dtype="<f4").reshape(4, 4).T.copy()
                if not (0 < W <= args.max_side and 0 < H <= args.max_side):
                    raise ValueError(f"invalid size {W}x{H}")
                # small image (camera moving) -> fewer samples per ray to stay smooth
                samples = args.samples // 2 if H <= 128 else args.samples
                img = render_image(current["model"], torch.from_numpy(c2w).cuda(), H, W, focal, samples)
                conn.sendall((img * 255).byte().cpu().numpy().tobytes())
            elif kind == 1:
                (idx,) = struct.unpack("<i", recv_all(conn, 4))
                conn.sendall(load_scene(int(np.clip(idx, 0, len(names) - 1))).tobytes())
            else:
                raise ValueError(f"unknown message {kind}")
    except (ConnectionError, OSError):
        print("client disconnected", flush=True)
    except Exception as e:  # e.g. CUDA out of memory or an invalid message: drop this client, keep the server alive
        print("client error:", repr(e), flush=True)
        torch.cuda.empty_cache()
    finally:
        conn.close()
