"""python nerf/test_hash.py  -> checks that HashEncoding.forward (int32) == forward_reference (int64)."""
import torch

from model import HashEncoding

enc = HashEncoding()
for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    e = enc.to(dev)
    with torch.no_grad():
        e.table.uniform_(-1, 1)  # with the table near 0 the test could not tell different indices apart
        x = torch.rand(2000, 3, device=dev) * 2 - 1
        assert torch.allclose(e(x), e.forward_reference(x), atol=1e-6), f"outputs differ on {dev}"
    print("ok", dev)
