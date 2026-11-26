"""NeRF in pure PyTorch (no CUDA extensions).

- Position encoding: 'hash' (multi-resolution hash grid, Instant-NGP style) or
  'freq' (sinusoidal positional encoding from the original paper).
- Density MLP -> feature vector -> view-direction-conditioned color MLP.
- Occupancy grid to skip empty space while sampling.
- Volume rendering from the paper (alpha compositing with accumulated transmittance).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- encodings
class FreqEncoding(nn.Module):
    def __init__(self, in_dim=3, n_freqs=10):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freqs) * math.pi)
        self.out_dim = in_dim * (1 + 2 * n_freqs)

    def forward(self, x):
        xf = x[..., None] * self.freqs  # (...,3,L)
        return torch.cat([x, torch.sin(xf).flatten(-2), torch.cos(xf).flatten(-2)], -1)


class HashEncoding(nn.Module):
    """Input in [-1,1]^3 (mapped to [0,1]^3). Trilinear interpolation per level."""

    PRIMES = (1, 2654435761, 805459861)

    def __init__(self, n_levels=16, n_feat=2, log2_size=19, base_res=16, max_res=1024):
        super().__init__()
        self.L, self.F, self.T = n_levels, n_feat, 2 ** log2_size
        growth = math.exp((math.log(max_res) - math.log(base_res)) / (n_levels - 1))
        res = [int(base_res * growth ** l) for l in range(n_levels)]
        self.register_buffer("res", torch.tensor(res, dtype=torch.float32))
        self.table = nn.Parameter(torch.empty(n_levels, self.T, n_feat).uniform_(-1e-4, 1e-4))
        self.register_buffer("corners", torch.tensor(
            [[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], dtype=torch.long))
        self.register_buffer("corners32", self.corners.int(), persistent=False)
        self.register_buffer("corners_f", self.corners.float(), persistent=False)
        # primes as int32 with the same two's-complement representation
        self.PRIMES32 = tuple(((p + 2 ** 31) % 2 ** 32) - 2 ** 31 for p in self.PRIMES)
        self.out_dim = n_levels * n_feat
        self.dense = [(r + 1) ** 3 <= self.T for r in res]  # coarse levels: direct indexing

    def forward_reference(self, x):
        """Simple version (loop per level). Kept to verify that forward() gives the same result."""
        x = (x * 0.5 + 0.5).clamp(0, 1 - 1e-6)
        out = []
        for l in range(self.L):
            p = x * self.res[l]
            p0 = p.floor()
            w = p - p0  # (N,3)
            p0 = p0.long()
            pc = p0[:, None, :] + self.corners[None]  # (N,8,3)
            if self.dense[l]:
                s = int(self.res[l]) + 1
                idx = pc[..., 0] + pc[..., 1] * s + pc[..., 2] * s * s
            else:
                idx = ((pc[..., 0] * self.PRIMES[0]) ^ (pc[..., 1] * self.PRIMES[1]) ^
                       (pc[..., 2] * self.PRIMES[2])) % self.T
            f = self.table[l][idx]  # (N,8,F)
            c = self.corners[None].float()
            wc = (w[:, None, :] * c + (1 - w[:, None, :]) * (1 - c)).prod(-1, keepdim=True)  # (N,8,1)
            out.append((f * wc).sum(1))
        return torch.cat(out, -1)

    def forward(self, x):
        """Loop per level with int32 indices: the hash multiplication overflows the same way as int64
        in the low bits, so `& (T-1)` gives exactly the same indices with half the memory."""
        x = (x * 0.5 + 0.5).clamp(0, 1 - 1e-6)
        c = self.corners_f[None]
        out = []
        for l in range(self.L):
            p = x * self.res[l]
            p0 = p.floor()
            w = (p - p0)[:, None, :]
            pc = p0.int()[:, None, :] + self.corners32[None]  # (N,8,3) int32
            if self.dense[l]:
                s = int(self.res[l]) + 1
                idx = pc[..., 0] + pc[..., 1] * s + pc[..., 2] * s * s
            else:
                idx = ((pc[..., 0] * self.PRIMES32[0]) ^ (pc[..., 1] * self.PRIMES32[1]) ^
                       (pc[..., 2] * self.PRIMES32[2])) & (self.T - 1)
            f = self.table[l][idx.long()]  # (N,8,F)
            wc = (w * c + (1 - w) * (1 - c)).prod(-1, keepdim=True)
            out.append((f * wc).sum(1))
        return torch.cat(out, -1)


# ---------------------------------------------------------------- model
class NeRF(nn.Module):
    def __init__(self, encoding="hash", hidden=64, geo_dim=15, aabb=1.0, grid_res=96):
        super().__init__()
        self.aabb = aabb  # the scene lives in [-aabb, aabb]^3
        if encoding == "hash":
            self.pos_enc = HashEncoding()
            width, depth = hidden, 2
        else:  # original paper: PE L=10, 8x256 MLP
            self.pos_enc = FreqEncoding(3, 10)
            width, depth = 256, 8
        self.dir_enc = FreqEncoding(3, 4)
        self.geo_dim = geo_dim if encoding == "hash" else 255

        layers, d = [], self.pos_enc.out_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU(True)]
            d = width
        layers += [nn.Linear(d, 1 + self.geo_dim)]
        self.sigma_net = nn.Sequential(*layers)

        d = self.geo_dim + self.dir_enc.out_dim
        cw = 64 if encoding == "hash" else 128
        self.color_net = nn.Sequential(
            nn.Linear(d, cw), nn.ReLU(True), nn.Linear(cw, cw), nn.ReLU(True), nn.Linear(cw, 3))

        self.bg = nn.Parameter(torch.full((3,), 0.5))  # learned background color (rays that hit nothing)

        # occupancy grid (max density per cell, EMA)
        self.grid_res = grid_res
        self.register_buffer("density_grid", torch.zeros(grid_res ** 3))
        self.register_buffer("occ", torch.ones(grid_res ** 3, dtype=torch.bool))

    # x in world coordinates
    def query(self, x, d):
        h = self.sigma_net(self.pos_enc(x / self.aabb))
        sigma = F.softplus(h[..., 0] - 1.0)  # >=0
        rgb = torch.sigmoid(self.color_net(torch.cat([h[..., 1:], self.dir_enc(d)], -1)))
        return sigma, rgb

    def density(self, x):
        return F.softplus(self.sigma_net(self.pos_enc(x / self.aabb))[..., 0] - 1.0)

    # ---------------- occupancy grid
    def _cell_centers(self):
        r = self.grid_res
        g = (torch.arange(r, device=self.occ.device) + 0.5) / r * 2 - 1
        return torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) * self.aabb

    @torch.no_grad()
    def update_occupancy(self, thresh=0.01, decay=0.95):
        c = self._cell_centers()
        jitter = (torch.rand_like(c) - 0.5) * (2 * self.aabb / self.grid_res)
        sigma = torch.cat([self.density(p) for p in (c + jitter).split(65536)])
        self.density_grid = torch.maximum(self.density_grid * decay, sigma)
        self.occ = self.density_grid > thresh

    def occupied(self, x):
        r = self.grid_res
        i = ((x / self.aabb * 0.5 + 0.5) * r).long().clamp(0, r - 1)
        return self.occ[i[..., 0] * r * r + i[..., 1] * r + i[..., 2]]

    # ---------------- render
    def render_rays(self, o, d, n_samples=128, perturb=False, bg='learned', use_occ=True, chunk=1 << 16):
        """o,d: (N,3). Returns rgb (N,3), accumulated opacity (N,), depth (N,)."""
        N = o.shape[0]
        dev = o.device
        # intersection with the cube [-aabb,aabb]^3
        inv = 1.0 / torch.where(d.abs() < 1e-8, torch.full_like(d, 1e-8), d)
        t0 = (-self.aabb - o) * inv
        t1 = (self.aabb - o) * inv
        tn = torch.minimum(t0, t1).amax(-1).clamp(min=0.0)
        tf = torch.maximum(t0, t1).amin(-1)
        hit = tf > tn
        tn = torch.where(hit, tn, torch.zeros_like(tn))
        tf = torch.where(hit, tf, torch.full_like(tf, 1e-3))

        u = torch.linspace(0, 1, n_samples, device=dev)[None].expand(N, -1)
        if perturb:
            u = u + torch.rand(N, n_samples, device=dev) / n_samples
        else:
            u = u + 0.5 / n_samples
        t = tn[:, None] + (tf - tn)[:, None] * u  # (N,S)
        delta = ((tf - tn) / n_samples)[:, None].expand(-1, n_samples)
        pts = o[:, None, :] + d[:, None, :] * t[..., None]

        sigma = torch.zeros(N, n_samples, device=dev)
        rgb = torch.zeros(N, n_samples, 3, device=dev)
        mask = hit[:, None].expand(-1, n_samples)
        if use_occ:
            mask = mask & self.occupied(pts)
        idx = mask.nonzero(as_tuple=True)
        if idx[0].numel() > 0:
            p = pts[idx]
            dd = d[idx[0]]
            s_out, c_out = [], []
            for pc, dc in zip(p.split(chunk), dd.split(chunk)):
                s, c = self.query(pc, dc)
                s_out.append(s)
                c_out.append(c)
            sigma = sigma.index_put(idx, torch.cat(s_out))
            rgb = rgb.index_put(idx, torch.cat(c_out))

        alpha = 1.0 - torch.exp(-sigma * delta)
        trans = torch.cumprod(torch.cat([torch.ones(N, 1, device=dev), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
        w = alpha * trans
        acc = w.sum(-1)
        color = (w[..., None] * rgb).sum(1)
        if isinstance(bg, str):
            bg = self.bg
        if bg is not None:
            color = color + (1.0 - acc)[:, None] * bg
        depth = (w * t).sum(-1)
        return color, acc, depth
