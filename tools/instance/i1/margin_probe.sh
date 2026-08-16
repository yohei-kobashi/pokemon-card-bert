#!/usr/bin/env bash
# Is there learnable signal in the CONFIDENT pairs, or none anywhere?
#
# The loop's held-out plan-conformance has sat at 46-60% for eight rounds on a TWO-way choice,
# i.e. around chance. Two readings fit that and they lead opposite ways:
#   (a) the playout labels are noise -> nothing to fit, and sharpening targets only makes the
#       model confidently wrong;
#   (b) the labels are fine but diluted -- median margin 0.26 with 24 playouts is about one
#       standard error of the Q estimate itself, so most pairs are coin flips that drown the
#       minority carrying real information.
#
# If (b), training on the high-margin tail should generalise ABOVE chance on held-out data from
# the same tail. If (a), it will not, whatever the threshold. Same model, same rows count, only
# the filter differs.
set -u
cd /root/ptcg/repo
POOL=/tmp/pairs_pool.jsonl.gz
zcat /root/fld_pairs7.jsonl.gz /root/fld_pairs8.jsonl.gz /root/fld_pairs9.jsonl.gz 2>/dev/null | gzip > "$POOL"
echo "pooled $(zcat $POOL | wc -l) pairs"

for TH in 0.0 0.35 0.60; do
    OUT=/tmp/pairs_m$TH.jsonl.gz
    python3 - "$POOL" "$OUT" "$TH" <<'PY'
import gzip, json, sys
src, dst, th = sys.argv[1], sys.argv[2], float(sys.argv[3])
n = k = 0
with gzip.open(dst, "wt") as o:
    for ln in gzip.open(src, "rt"):
        r = json.loads(ln); n += 1
        if (r.get("qw", 0) - r.get("ql", 0)) >= th:
            o.write(ln); k += 1
print("  margin >= %.2f : %d / %d pairs" % (th, k, n))
PY
    ROWS=/tmp/rows_m$TH.jsonl.gz
    # beta 0 and a sharper temperature: with the flattening terms removed, a fit failure is the
    # LABELS failing, not the target construction hiding the signal.
    python3 /root/mrl_convert.py --pairs "$OUT" --out "$ROWS" --beta 0.0 --temp 0.25 >/dev/null
    echo "=== threshold $TH ==="
    PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 python3 -u tools/dusk_plan_train.py \
        --data "$ROWS" --model /root/out/fld_r2a --out /tmp/mp_$TH \
        --lr 1e-5 --epochs 2 --accum 4 --l2sp 0 --eval-frac 0.2 2>&1 \
        | grep -aE "\[data\]|conformance|FINAL"
done
echo "MARGIN_PROBE_DONE"
