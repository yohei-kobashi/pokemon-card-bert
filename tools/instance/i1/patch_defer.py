p = "/root/defer_gate.sh"
s = open(p).read()
old = """# Wait for whatever is on the GPU (the field chain's round-1 gate) rather than fighting it.
for _ in $(seq 1 240); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$u" -le 2000 ] && break
    sleep 60
done"""
new = """# NO GPU WAIT. The field chain runs its rounds back to back, so "wait for an idle GPU" meant
# waiting for a gap that never opens -- the gate sat idle through round 1 and would have sat
# through every later one. DeBERTa-v3-base is ~1.5 GiB and the field gate peaks near 7.6 of the
# card's 24, so the two fit together; they trade throughput, not correctness, and with the
# deadline on 08-16 a slower answer to both beats a fast answer to one."""
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, new))
print("defer_gate: gpu wait removed")
