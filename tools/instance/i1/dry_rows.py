"""Why can 300 rows of ONE rule not be memorised?

Two learning rates a decade apart both leave the loss flat and conformance at chance. A 184M
model failing to memorise 300 examples is not an optimisation problem, so look at what the
examples actually are. The two ways a dataset makes memorisation impossible:

  * CONTRADICTION -- the same (prompt, menu) appears with different correct answers. No model
    can fit both, and the best it can do is the average, which is what a flat loss at the
    entropy of the menu looks like.
  * NO SIGNAL -- the correct answer is not a function of anything in the prompt (e.g. the rule
    keys on a board fact the prompt does not render).

The first is visible in the data alone. The second is not, but ruling the first out is what
makes it worth looking for.
"""
import collections, gzip, json, math, random, sys

rows = [json.loads(x) for x in gzip.open(sys.argv[1], "rt")]
random.Random(0).shuffle(rows)
rows = rows[:300]                       # exactly what the probe trains on

key = lambda r: (r["prompt"], tuple(r["cands"]))
by = collections.defaultdict(list)
for r in rows:
    by[key(r)].append(r)

dup = {k: v for k, v in by.items() if len(v) > 1}
contra = 0
for k, v in dup.items():
    best = {max(range(len(r["wc"])), key=lambda j: r["wc"][j]) for r in v}
    if len(best) > 1:
        contra += 1
print("300 rows | %d distinct (prompt,menu) | %d duplicated | %d of those CONTRADICT"
      % (len(by), len(dup), contra))

nc = [len(r["cands"]) for r in rows]
print("candidates/row: min %d p50 %d max %d" % (min(nc), sorted(nc)[len(nc) // 2], max(nc)))

# The loss floor is the target entropy; a row whose weight is spread cannot go to zero.
fl = []
for r in rows:
    w = [x for x in r["wc"] if x > 0]
    s = sum(w)
    fl.append(-sum((x / s) * math.log(x / s) for x in w) if s > 0 else 0.0)
print("target entropy: mean %.4f | rows with a single non-zero weight: %d/300"
      % (sum(fl) / len(fl), sum(1 for x in fl if x < 1e-9)))

# If the answer is a function of the PROMPT at all, prompts that repeat should agree. If every
# prompt is unique the model is being asked to memorise 300 independent facts, which is easy --
# unless the candidate TEXTS carry the answer and the prompt does not, in which case identical
# candidate sets with different answers is the tell.
pk = collections.defaultdict(set)
for r in rows:
    pk[r["prompt"]].add(tuple(r["cands"]))
print("distinct prompts: %d | prompts appearing with >1 menu: %d"
      % (len(pk), sum(1 for v in pk.values() if len(v) > 1)))

ck = collections.defaultdict(set)
for r in rows:
    ck[tuple(r["cands"])].add(max(range(len(r["wc"])), key=lambda j: r["wc"][j]))
amb = sum(1 for v in ck.values() if len(v) > 1)
print("distinct menus: %d | menus whose correct answer varies with the board: %d" % (len(ck), amb))

r = rows[0]
print("\nsample row:\n  rules=%s\n  wc=%s\n  cands=%s\n  prompt head=%r"
      % (r.get("rules"), r["wc"], r["cands"][:6], r["prompt"][:110]))
