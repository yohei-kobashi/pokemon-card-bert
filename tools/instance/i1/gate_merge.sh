#!/usr/bin/env bash
# Win rate across the merge sweep. Conformance is cheap and rises smoothly; the whole question
# is where the WIN RATE falls off, and only games answer that.
#
# Screening shape: four opponents chosen to span s1's range rather than to be representative --
# slowking 54.7%, dragapult 31.3%, dudunsparce_box 28.0%, marnie_grimmsnarl 20.0%. A drop shows
# up first where there is something to lose, and a cell already at 6% cannot fall far enough to
# be measured. The full eleven come later, for the winner only.
#
# All arms run inside ONE process per opponent, so every arm sees the same (seed, seat) against
# the same opponent and the differences are paired. That is what makes 120 games per cell enough
# to resolve ~3pt when the unpaired standard error at that count would be ~4.5pt.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
OUT=/root/loop_dusk/gate_merge
GAMES=${GAMES:-120}
ALPHAS="${ALPHAS:-0.10 0.20 0.35 0.50 0.75}"
mkdir -p "$OUT"
say() { echo "[gmerge $(date -u +%m-%d_%H:%M:%S)] $*"; }

ARMS="--arm s1=hf:/root/out/dusk_s1@dusk"
for A in $ALPHAS; do
    [ -f "/root/out/merge/a$A/model.safetensors" ] || { say "STOP: no merge at alpha $A"; exit 1; }
    ARMS="$ARMS --arm a$A=hf:/root/out/merge/a$A@dusk"
done
say "arms: s1 + ${ALPHAS}"

i=0
for OPP in slowking dragapult dudunsparce_box marnie_grimmsnarl; do
    nohup python3 -u tools/gate_protagonist.py \
        --deck dragapult_dusknoir --opp "$OPP" --games "$GAMES" --seed $((5000 + i * 100)) \
        --baseline s1 $ARMS \
        --out "$OUT/$OPP.json" > "$OUT/$OPP.log" 2>&1 &
    i=$((i + 1)); sleep 30
done
say "launched $i opponent shards"; wait

python3 - "$OUT" <<'PY'
import glob, json, math, os, sys
d = sys.argv[1]
cells = {}
for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    try:
        cells.update(json.load(open(p)).get("cells", {}))
    except Exception as e:
        print("unreadable %s: %s" % (os.path.basename(p), e))
if not cells:
    sys.exit("no cells")
arms = sorted({k.split("|")[0] for k in cells}, key=lambda a: (a != "s1", a))
opps = sorted({k.split("|")[1] for k in cells})
print("\n%-20s %s" % ("opponent", " ".join("%8s" % a for a in arms)))
tot = {a: [] for a in arms}
for o in opps:
    row = []
    for a in arms:
        v = cells.get("%s|%s" % (a, o), {}).get("raw") or []
        tot[a].extend(v)
        row.append("%7.1f%%" % (100.0 * sum(v) / max(1, len(v))))
    print("%-20s %s" % (o, " ".join(row)))
print("\n%-8s %8s %10s %8s   %s" % ("arm", "win%", "vs s1", "t", "n"))
base = tot["s1"]
for a in arms:
    v = tot[a]
    wr = 100.0 * sum(v) / max(1, len(v))
    if a == "s1":
        print("%-8s %7.1f%% %10s %8s   %d" % (a, wr, "(baseline)", "", len(v)))
        continue
    dd = [x - y for x, y in zip(v, base)]
    m = sum(dd) / len(dd)
    sd = math.sqrt(sum((x - m) ** 2 for x in dd) / max(1, len(dd) - 1))
    se = sd / math.sqrt(len(dd))
    print("%-8s %7.1f%% %+9.2fpt %+8.2f   %d" % (a, wr, 100 * m, m / se if se else 0, len(v)))
PY
say GATE_MERGE_DONE
