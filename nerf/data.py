"""Data loading (NeRF/Instant-NGP style transforms.json, OpenGL convention)."""
import json
import os

import cv2
import numpy as np
import torch


def get_rays(c2w, H, W, focal, cx=None, cy=None):
    """Per-pixel rays. c2w: (4,4) tensor. Camera looks along -Z, +Y is up (same as the OpenGL viewer)."""
    cx = W * 0.5 if cx is None else cx
    cy = H * 0.5 if cy is None else cy
    j, i = torch.meshgrid(torch.arange(H, device=c2w.device, dtype=torch.float32),
                          torch.arange(W, device=c2w.device, dtype=torch.float32), indexing="ij")
    dirs = torch.stack([(i + 0.5 - cx) / focal, -(j + 0.5 - cy) / focal, -torch.ones_like(i)], -1)
    d = dirs @ c2w[:3, :3].T
    o = c2w[:3, 3].expand_as(d)
    return o.reshape(-1, 3), torch.nn.functional.normalize(d.reshape(-1, 3), dim=-1)


class Dataset:
    def __init__(self, root, device="cuda", holdout_every=8):
        meta = json.load(open(os.path.join(root, "transforms.json")))
        self.meta = meta
        self.H, self.W = meta["h"], meta["w"]
        self.focal = meta["fl_x"]
        self.aabb = meta["aabb"]
        imgs, poses, names, alphas = [], [], [], []
        for f in meta["frames"]:
            raw = cv2.imread(os.path.join(root, f["file_path"]), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise SystemExit(f"Could not read image {f['file_path']}")
            if raw.ndim == 3 and raw.shape[2] == 4:
                alphas.append(raw[..., 3])
                raw = raw[..., :3]
            imgs.append(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            poses.append(f["transform_matrix"])
            names.append(f["file_path"])
        self.images = torch.from_numpy(np.stack(imgs)).to(device).float() / 255.0  # (N,H,W,3)
        # With alpha (Blender): images = composited over white (for metrics); training uses
        # premultiplied color + alpha and a random background, which forces empty space to stay empty.
        self.has_alpha = len(alphas) == len(imgs)
        if self.has_alpha:
            self.alpha = torch.from_numpy(np.stack(alphas)).to(device).float()[..., None] / 255.0
            self.premult = self.images * self.alpha
            self.images = self.premult + (1.0 - self.alpha)
        self.poses = torch.tensor(poses, dtype=torch.float32, device=device)       # (N,4,4)
        n = len(names)
        idx = np.arange(n)
        self.val_idx = idx[::holdout_every]
        self.train_idx = np.setdiff1d(idx, self.val_idx)
        self.device = device
        # all training rays precomputed
        o, d, c, a = [], [], [], []
        for i in self.train_idx:
            oi, di = get_rays(self.poses[i], self.H, self.W, self.focal)
            o.append(oi); d.append(di); c.append((self.premult[i] if self.has_alpha else self.images[i]).reshape(-1, 3))
            if self.has_alpha:
                a.append(self.alpha[i].reshape(-1, 1))
        self.rays_o, self.rays_d, self.rays_c = torch.cat(o), torch.cat(d), torch.cat(c)
        self.rays_a = torch.cat(a) if self.has_alpha else None

    def sample(self, n):
        i = torch.randint(0, self.rays_o.shape[0], (n,), device=self.device)
        return self.rays_o[i], self.rays_d[i], self.rays_c[i], (self.rays_a[i] if self.has_alpha else None)
