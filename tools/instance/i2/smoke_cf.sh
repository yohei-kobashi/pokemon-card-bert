#!/bin/bash
# Smoke-test the card-first scorer BEFORE the night depends on it.
#
# The real checkpoint does not exist yet -- domain_embeddings.pt and cardfirst_vocab.json are
# written only at the very end of training. A stand-in is built from the latest intermediate
# checkpoint plus a correctly SHAPED (random) embedding blob: the moves it plays are meaningless,
# but every mechanism the night depends on runs for real -- vocabulary resize, row restore, LoRA
# attach, first-token argmax, and the second forward for tie-breaks.
set -u
cd /root/ptcg/repo
CK=$(ls -d /root/out/qwen3_4b_cf1/checkpoint-* | sort -V | tail -1)
FAKE=/root/out/cf_smoke
rm -rf $FAKE; cp -r "$CK" $FAKE
cp data/cardfirst_v39.json $FAKE/cardfirst_vocab.json
python3 - <<PY
import torch, json
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("$FAKE")
n_added = len(tk) - 151669
g = torch.Generator().manual_seed(0)
rows = torch.randn(n_added, 2560, generator=g) * 0.02
torch.save({"n_base": 151669, "rows": rows}, "$FAKE/domain_embeddings.pt")
print("stand-in: %d added rows on top of 151669 (vocab %d)" % (n_added, len(tk)))
PY
echo "=== scoring 2 games ==="
PYTHONPATH=cg-lib timeout 2400 python3 tools/mirror_match.py --deck crustle_stall \
    --a engine --b "qwen:$FAKE" --max-games 2 --out /root/smoke_cf.json 2>&1 | tail -18
echo "=== result file ==="; cat /root/smoke_cf.json 2>/dev/null | head -c 400; echo
rm -rf $FAKE
echo SMOKE DONE
