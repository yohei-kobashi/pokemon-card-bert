#!/usr/bin/env bash
# Does the SUBMITTABLE artifact win? Everything measured so far about dusk_s1 -- the +4.06pt
# gate, the 91.6% top1 -- was measured on the PyTorch fp32 checkpoint. What gets uploaded is a
# vocab-pruned, weight-only-INT8 ONNX driven by an embedded lm/ through main.py. Argmax agreement
# was 100% over 60 decisions, which is under one game.
#
# Two arms, same seeds, same seats:
#   pure    the model decides everything
#   attach  attach decisions go to engine_v2 instead (the +11.4pt was measured on a DIFFERENT
#           model and a DIFFERENT prompt format, so it is a hypothesis here, not a setting)
#
# One process per opponent so the 61.4 effective cores are used without oversubscribing: 11
# processes x 4 ORT threads = 44. Do not raise either number without re-reading the quota.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
OUT=/root/loop_dusk/gate_onnx
GAMES=${GAMES:-50}
mkdir -p "$OUT"
say() { echo "[gonnx $(date -u +%m-%d_%H:%M:%S)] $*"; }

for TAG in dusk_s1_pure dusk_s1_attach; do
    [ -f "/root/subm/$TAG/main.py" ] || { say "STOP: no staged bundle at /root/subm/$TAG"; exit 1; }
done

i=0
for OPP in marnie_grimmsnarl alakazam_nz alakazam crustle_geco crustle ogerpon_mono \
           dudunsparce_box cynthia_garchomp dragapult mega_lucario_tr slowking; do
    nohup python3 -u tools/gate_protagonist.py \
        --deck dragapult_dusknoir --opp "$OPP" --games "$GAMES" --seed $((3000 + i * 100)) \
        --arm "pure=bundle:/root/subm/dusk_s1_pure@dusk" \
        --arm "attach=bundle:/root/subm/dusk_s1_attach@dusk" \
        --baseline pure \
        --out "$OUT/$OPP.json" > "$OUT/$OPP.log" 2>&1 &
    i=$((i + 1))
    sleep 8      # stagger the ONNX session builds
done
say "launched $i opponent shards x 2 arms x $GAMES games"
wait

say "=== per-opponent ==="
python3 - "$OUT" <<'PY'
import json, glob, math, os, sys
d = sys.argv[1]
cells = {}
for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    try:
        j = json.load(open(p))
    except Exception as e:
        print("unreadable %s: %s" % (os.path.basename(p), e)); continue
    for k, v in j.get("cells", {}).items():
        cells[k] = v
if not cells:
    sys.exit("no cells")
arms = sorted({k.split("|")[0] for k in cells})
opps = sorted({k.split("|")[1] for k in cells})
print("%-22s %8s %8s %8s" % ("opponent", arms[0], arms[1] if len(arms) > 1 else "", "delta"))
tot = {a: [] for a in arms}
for o in opps:
    row = {}
    for a in arms:
        v = cells.get("%s|%s" % (a, o), {}).get("raw") or []
        row[a] = v
        tot[a].extend(v)
    def wr(x):
        return 100.0 * sum(x) / max(1, len(x))
    dl = (wr(row.get(arms[1], [])) - wr(row.get(arms[0], []))) if len(arms) > 1 else 0.0
    print("%-22s %7.1f%% %7.1f%% %+7.1f"
          % (o, wr(row.get(arms[0], [])), wr(row.get(arms[1], [])) if len(arms) > 1 else 0, dl))
print()
for a in arms:
    print("%-8s overall %5.1f%%  (%d games)" % (a, 100.0 * sum(tot[a]) / max(1, len(tot[a])), len(tot[a])))
if len(arms) > 1 and len(tot[arms[0]]) == len(tot[arms[1]]):
    diffs = [x - y for x, y in zip(tot[arms[1]], tot[arms[0]])]
    m = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / max(1, len(diffs) - 1))
    se = sd / math.sqrt(len(diffs))
    print("PAIRED %s - %s: %+.2fpt +- %.2f  t %+.2f  (n=%d)"
          % (arms[1], arms[0], 100 * m, 100 * se, m / se if se else 0, len(diffs)))
PY
say "GATE_ONNX_DONE"
