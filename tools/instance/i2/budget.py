"""How much of the data do we actually have to train on?

Throughput is 8.6 samples/s, so the full 2.52M mix is 81 hours -- out of reach. The useful
question is not "how many samples" but "at N samples, what fraction of real DECISIONS have a
correct answer whose token the model has seen enough times to have learned it".

A token is counted as learned at >= T label occurrences INSIDE the training slice. Coverage is
then measured on a DISJOINT tail of the file, so it is a held-out number rather than a
restatement of the training counts.
"""
import gzip, json, re, sys, collections
sys.path.insert(0, ".")
from lm.action_token import action_token

RE = re.compile(r"(?:^| )(\d+)=(\S+)")
HOLD = 200000
rows = []
labels = []
n = 0
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
        labels.append(action_token(opts[k]))
        if n > 2_321_800:                      # the last slice becomes held-out
            rows.append(action_token(opts[k]))
        if len(rows) >= HOLD:
            break
print("labels read %d | held-out %d" % (len(labels), len(rows)))

for N in (100_000, 150_000, 250_000, 400_000, 600_000, 1_000_000, 1_500_000):
    seen = collections.Counter(labels[:N])
    line = "  %8d samples (%4.1f h): " % (N, N / 8.628 / 3600)
    for T in (5, 20, 100):
        cov = sum(1 for t in rows if seen.get(t, 0) >= T)
        line += "  >=%3d seen: %5.1f%%" % (T, 100.0 * cov / len(rows))
    ntok = sum(1 for v in seen.values() if v >= 20)
    print(line + "   | tokens with >=20: %d" % ntok)
print("\n'>=20 seen: X%' = on X% of held-out decisions, the correct answer's token appeared at")
print("least 20 times in the training slice. The rest are answers the model had little or no")
print("chance to learn -- an upper bound on accuracy that no amount of tuning can move.")
