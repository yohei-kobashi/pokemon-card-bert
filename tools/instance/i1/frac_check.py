"""Is decision subsampling an unbiased estimate of the same update?

Compares the UPDATE DIRECTION (ckpt - start, flattened over all tensors) between a full-data
run and subsampled runs. If subsampling is unbiased, the directions agree up to noise; if it
biases the update, they diverge systematically. Two different subsample seeds bound the noise
floor: cos(half_a, half_b) is how much two equally-valid samples disagree with each other, so
cos(full, half) should be no worse than that.
"""
import sys
import torch
from safetensors import safe_open


def delta(a, b):
    """flattened (b - a) over the tensors both share, float32"""
    fa, fb = safe_open(a, "pt"), safe_open(b, "pt")
    ks = sorted(set(fa.keys()) & set(fb.keys()))
    return torch.cat([(fb.get_tensor(k).float() - fa.get_tensor(k).float()).reshape(-1)
                      for k in ks])


start = sys.argv[1]
runs = sys.argv[2:]
base = start + "/model.safetensors"
ds = {}
for r in runs:
    name = r.rstrip("/").split("/")[-1]
    d = delta(base, r + "/model.safetensors")
    ds[name] = d
    print("%-22s ||delta|| = %.6e" % (name, d.norm().item()))
print()
names = list(ds)
print("%-22s %-22s %8s" % ("A", "B", "cosine"))
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = ds[names[i]], ds[names[j]]
        c = torch.dot(a, b) / (a.norm() * b.norm())
        print("%-22s %-22s %8.4f" % (names[i], names[j], c.item()))
