#!/usr/bin/env bash
# Build the interim submission (user 2026-08-16: after 21:00 JST, once a round completes).
# Chain proven by dusk_v1 (COMPLETE, 337.0): sweep-reuse -> prune+INT8 -> build -> tarball.
# One addition vs dusk_v1: the keep-id set is UNIONED with the CURRENT deck's domain tokens
# (the list changed twice today; a card token outside the kept set maps to [UNK] silently).
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
say() { echo "[sub3 $(date -u +%m-%d_%H:%M:%S)] $*"; }
W=/root/onnx_sub4
mkdir -p "$W"
POOL=$REPO/data/rerank/v41_dusk11.jsonl.gz

CKPT=$(python3 -c 'from lm import registry as r; t=r.resolve("dragapult_dusknoir")["target"]; print(t.split(":",1)[1])')
case "$CKPT" in /*) ;; *) CKPT=/root/out/$CKPT ;; esac
[ -s "$CKPT/model.safetensors" ] || { say "STOP: no weights at $CKPT"; exit 1; }
WRAP=$(python3 -c 'import json; print(json.load(open("models/adapters.json"))["decks"]["dragapult_dusknoir"]["wrap"])')
say "champion=$CKPT"
say "wrap=$WRAP"

# keep ids: old sweep + every domain token of the CURRENT deck (and its attacks)
python3 - "$CKPT" <<'PY'
import json, sys
from transformers import AutoTokenizer
keep = set(json.load(open('/root/onnx_dusk/keep_ids.json')))
tok = AutoTokenizer.from_pretrained(sys.argv[1])
want_tokens = set()
ids = [int(l) for l in open('decks/dragapult_dusknoir.csv') if l.strip()]
from agents._engine import _CARDS, _ATTACKS
for cid in set(ids):
    want_tokens.add("c%d" % cid)
    c = _CARDS.get(cid)
    for aid in (getattr(c, "attacks", None) or []):
        want_tokens.add("a%d" % aid)
added = 0
missing = []
for t in sorted(want_tokens):
    tid = tok.convert_tokens_to_ids(t)
    unk = tok.unk_token_id
    if tid is not None and tid != unk:
        if tid not in keep:
            keep.add(tid); added += 1
    else:
        missing.append(t)
json.dump(sorted(keep), open('/root/onnx_sub4/keep_ids.json', 'w'))
print("keep=%d (added %d deck tokens); tokens with no vocab entry: %s" % (len(keep), added, missing[:8]))
PY

say "prune + export + weight-only INT8 for $CKPT"
python3 tools/prune_vocab_rerank.py --model "$CKPT" --keep "$W/keep_ids.json" \
    --data "$POOL" --work "$W/pruned" --n 60 --max-len 512 > "$W/prune.log" 2>&1 \
    || { say "STOP: prune failed"; tail -20 "$W/prune.log"; exit 1; }
grep -aE "argmax|BUDGET" "$W/prune.log" | tail -6

REMAP="$W/pruned/model/vocab_remap.npy"
say "building bundle dusk_v4"
python3 tools/build_rerank_submission.py dragapult_dusknoir \
    --onnx "$W/pruned/model_wonly_int8.onnx" \
    --tokenizer "$W/pruned/model" \
    ${REMAP:+--remap "$REMAP"} \
    --wrap "$WRAP" \
    --pfmt dusk --tag dusk_v4 \
    --threads 4 --max-len 512 --time-budget 480 \
    --out /root/subm > /root/subm_dusk_v4.log 2>&1 \
    || { say "BUILD FAILED"; tail -30 /root/subm_dusk_v4.log; exit 1; }
grep -aE "SELFCHECK|selfcheck|MiB|cap|wrap" /root/subm_dusk_v4.log | tail -10
ls -la /root/subm/dusk_v4.tar.gz
say "SUB4_BUILD_DONE champion=$CKPT"
