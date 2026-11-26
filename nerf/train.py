import argparse
import math
import os
import time

import cv2
import numpy as np
import torch

from data import Dataset
from model import NeRF
from render import render_image

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/casco")
ap.add_argument("--out", default="data/casco/nerf.pt")
ap.add_argument("--encoding", default="hash", choices=["hash", "freq"])
ap.add_argument("--iters", type=int, default=8000)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--samples", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--eval_every", type=int, default=1000)
args = ap.parse_args()

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ds = Dataset(args.data, dev)
print(f"{len(ds.train_idx)} training images, {len(ds.val_idx)} validation images, {ds.W}x{ds.H}, focal {ds.focal:.1f}")
model = NeRF(args.encoding, aabb=ds.aabb).to(dev)
if ds.has_alpha:  # fixed white background for viewing; training uses a random background
    model.bg.data.fill_(1.0)
    print("dataset with alpha: training with a random background")

lr = args.lr if args.encoding == "hash" else 5e-4
opt = torch.optim.Adam([
    {"params": model.pos_enc.parameters(), "weight_decay": 0.0},
    {"params": list(model.sigma_net.parameters()) + list(model.color_net.parameters()) + [model.bg],
     "weight_decay": 1e-6},
], lr=lr, betas=(0.9, 0.99), eps=1e-15)
warm = 200
sched = lambda it: min(1.0, (it + 1) / warm) * (0.1 ** (it / args.iters))  # decays to 0.1x


def save():
    """Write to a temp file and replace: a cut mid-write does not damage the previous nerf.pt."""
    tmp = args.out + ".tmp"
    torch.save({"state": model.state_dict(), "encoding": args.encoding, "aabb": ds.aabb, "meta": ds.meta}, tmp)
    os.replace(tmp, args.out)


def psnr_val():
    ps = []
    for i in ds.val_idx[:4]:
        img = render_image(model, ds.poses[i], ds.H, ds.W, ds.focal, args.samples)
        ps.append(-10 * math.log10(((img - ds.images[i]) ** 2).mean().item()))
    model.train()
    return float(np.mean(ps))


t0 = time.time()
model.train()
for it in range(args.iters + 1):
    for g in opt.param_groups:
        g["lr"] = lr * sched(it)
    o, d, c, a = ds.sample(args.batch)
    if a is not None:
        bg = torch.rand(args.batch, 3, device=dev)
        c = c + (1.0 - a) * bg
    else:
        bg = "learned"
    rgb, acc, _ = model.render_rays(o, d, args.samples, perturb=True, use_occ=it > 256, bg=bg)
    loss = torch.nn.functional.mse_loss(rgb, c)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if it % 16 == 0 and it > 0:
        model.update_occupancy(thresh=0.01)
    if it % 100 == 0:
        print(f"it {it:6d}  loss {loss.item():.5f}  psnr {-10 * math.log10(loss.item()):.2f}  "
              f"occupancy {model.occ.float().mean().item() * 100:.1f}%  {time.time() - t0:.0f}s", flush=True)
    if it > 0 and it % args.eval_every == 0:
        print(f"   >> validation PSNR: {psnr_val():.2f} dB", flush=True)
        save()

save()
# comparison image on one validation view
i = ds.val_idx[0]
img = render_image(model, ds.poses[i], ds.H, ds.W, ds.focal, args.samples)
cmp = torch.cat([ds.images[i], img], 1).cpu().numpy()
cv2.imwrite(os.path.join(args.data, "val_comparison.png"), cv2.cvtColor((cmp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
print("done ->", args.out)
