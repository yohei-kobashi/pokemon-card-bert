#!/bin/bash
# Put the deduped pools in place of the originals, so every script that names them keeps working.
# The raw files move into raw/ rather than being renamed in place: the loop's mixer globs
# dagger_r*.jsonl.gz, and a sibling called dagger_r1.raw.jsonl.gz would still match it and quietly
# re-introduce exactly the records this is removing.
set -eu
cd /root/ptcg/repo
cp /tmp/agent.py lm/agent.py
mkdir -p /root/loop_rerank/raw data/rerank/raw
for r in 1 2; do
  mv /root/loop_rerank/dagger_r$r.jsonl.gz /root/loop_rerank/raw/
  mv /root/loop_rerank/dagger_r$r.dd.jsonl.gz /root/loop_rerank/dagger_r$r.jsonl.gz
done
mv data/rerank/v39_0731.rerank.jsonl.gz data/rerank/raw/
mv data/rerank/v39_0731.dd.rerank.jsonl.gz data/rerank/v39_0731.rerank.jsonl.gz
rm -f data/rerank/loop_r*.rerank.jsonl.gz      # stale mixes built from the old pools
echo "--- in place ---"
ls -la /root/loop_rerank/*.gz data/rerank/v39_0731.rerank.jsonl.gz | awk '{print $5, $9}'
echo "--- moved aside ---"
ls /root/loop_rerank/raw data/rerank/raw
python3 - <<'PY'
import gzip, json, sys
sys.path.insert(0, "/root/ptcg/repo")
from lm.action_token import equivalent
n = bad = 0
with gzip.open("data/rerank/v39_0731.rerank.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line); n += 1
        c, k = d["candidates"], d["chosen"]
        if any(equivalent(x, c[k]) for j, x in enumerate(c) if j != k):
            bad += 1
        if n >= 200000:
            break
print("VERIFY: %d records checked, %d still contain an equivalent negative" % (n, bad))
PY
