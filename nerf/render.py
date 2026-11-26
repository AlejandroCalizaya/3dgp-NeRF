import torch

from data import get_rays


@torch.no_grad()
def render_image(model, c2w, H, W, focal, n_samples=128, chunk=1 << 17, point_chunk=1 << 18):
    """Return an (H,W,3) float image in [0,1] on the GPU."""
    model.eval()
    o, d = get_rays(c2w, H, W, focal)
    out = []
    for oc, dc in zip(o.split(chunk), d.split(chunk)):
        rgb, _, _ = model.render_rays(oc, dc, n_samples=n_samples, perturb=False, chunk=point_chunk)
        out.append(rgb)
    return torch.cat(out).reshape(H, W, 3).clamp(0, 1)


@torch.no_grad()
def prune_occupancy(model, thresh, passes=4):
    """Recompute the occupancy grid with a higher threshold: faster, slightly lower quality."""
    c = model._cell_centers()
    g = torch.zeros_like(model.density_grid)
    for _ in range(passes):
        j = (torch.rand_like(c) - 0.5) * (2 * model.aabb / model.grid_res)
        g = torch.maximum(g, torch.cat([model.density(p) for p in (c + j).split(65536)]))
    model.occ = g > thresh


def load_model(path, device="cuda", prune=None):
    from model import NeRF
    ck = torch.load(path, map_location=device)
    m = NeRF(encoding=ck["encoding"], aabb=ck["aabb"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    if prune:
        prune_occupancy(m, prune)
    return m, ck
