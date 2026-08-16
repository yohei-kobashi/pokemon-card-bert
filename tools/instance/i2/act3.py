"""Size the action-token vocabulary and measure the ambiguity it CANNOT resolve.

The check is independent of the token function: `equivalent()` is defined on the raw option
strings, so "the token covers several options and they are not the same act" is a real finding
rather than a restatement of how the token was built.
"""
import gzip, json, re, sys, collections
sys.path.insert(0, ".")
from lm.action_token import action_token, equivalent, ambiguous

RE = re.compile(r"(?:^| )(\d+)=(\S+)")
vocab = collections.Counter()
n = amb = benign = 0
amb_ex = collections.Counter()
kinds = collections.Counter()
with gzip.open("data/sft/v39_dag005.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        t = d.get("target")
        if not t:
            continue
        opts = [o for _, o in RE.findall(d["prompt"].rsplit(":: ", 1)[-1])]
        k = int(t)
        if k >= len(opts):
            continue
        n += 1
        tok = action_token(opts[k])
        vocab[tok] += 1
        hits = [o for o in opts if action_token(o) == tok]
        if len(hits) > 1:
            if all(equivalent(o, opts[k]) for o in hits):
                benign += 1
            else:
                amb += 1
                amb_ex[(opts[k].split(":")[0], hits[0][:22], hits[1][:22])] += 1
        for o in opts:
            kinds[action_token(o).split("|")[1] if "|" in action_token(o) else "?"] += 0
        if n >= 400000:
            break
c = sorted(vocab.values(), reverse=True)
tot = sum(c)
print("decisions %d" % n)
print("distinct action tokens (as LABELS) %d" % len(c))
print("  no collision       %.2f%%" % (100.0 * (n - benign - amb) / n))
print("  benign collision   %.2f%%   same act, different copy -> any is correct" % (100.0 * benign / n))
print("  REAL  collision    %.3f%%  (%d decisions need a tie-break)" % (100.0 * amb / n, amb))
if amb_ex:
    print("\n  what the real ties look like:")
    for kk, v in amb_ex.most_common(8):
        print("    %-9s %-24s vs %-24s %6d" % (kk[0], kk[1], kk[2], v))
print("\nlabel frequency (can every token be learned?)")
for thr in (5, 20, 100):
    rare = sum(1 for x in c if x < thr)
    mass = sum(x for x in c if x < thr)
    print("  seen < %-4d %5d tokens (%2.0f%% of vocab) carrying %.2f%% of the labels"
          % (thr, rare, 100.0 * rare / len(c), 100.0 * mass / tot))
print("  median %d | max %d" % (c[len(c) // 2], c[0]))
