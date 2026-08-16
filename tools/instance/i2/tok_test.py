import sys, gzip, json
sys.path.insert(0, "/root")
from eval_teacher import index_seqs, n_options, softmax
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("unsloth/Qwen3.5-9B-Base")
seqs = index_seqs(tk, 52)
by = {}
for k, s in enumerate(seqs):
    by.setdefault(len(s), []).append(k)
print("index tokenization:")
for L, v in sorted(by.items()):
    print("  %d token(s): %d values  e.g. %s -> %s" % (L, len(v), v[:4], [seqs[k] for k in v[:3]]))
print("distinct first tokens among 2-token indices:",
      sorted({s[0] for s in seqs if len(s) > 1}))
print("forwards needed: n=10 ->", len({s[:d] for s in seqs[:10] for d in range(len(s))}),
      "| n=26 ->", len({s[:d] for s in seqs[:26] for d in range(len(s))}),
      "| n=52 ->", len({s[:d] for s in seqs for d in range(len(s))}))
# real held-out rows
P, C = [], []
with gzip.open("/root/ptcg/repo/data/sft/teacher_0730_index.jsonl.gz", "rt") as f:
    for i, line in enumerate(f):
        if i >= 4000: break
        d = json.loads(line)
        if d.get("target"): P.append(d["prompt"]); C.append(d["target"])
ns = [n_options(p) for p in P]
big = [n for n in ns if n > 10]
print("\nheld-out %d rows | n_options p50 %d p90 %d max %d | >10: %d (%.1f%%)"
      % (len(P), sorted(ns)[len(ns)//2], sorted(ns)[int(len(ns)*.9)], max(ns), len(big),
         100.0*len(big)/len(ns)))
bad = [i for i,(p,c) in enumerate(zip(P,C)) if not c.isdigit() or int(c) >= n_options(p)]
print("targets not scorable (non-digit or out of range): %d" % len(bad))
tl = [len(tk(p, add_special_tokens=False)["input_ids"]) for p in P[:600]]
tl.sort()
print("prompt tokens p50 %d p99 %d max %d (maxlen 1024)" % (tl[300], tl[594], tl[-1]))
print("softmax sanity:", [round(x,4) for x in softmax([-1.0,-2.0,-3.0])])
