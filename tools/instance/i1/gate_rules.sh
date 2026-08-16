#!/usr/bin/env bash
# Are damage-counter placement and energy allocation better done by RULE than by the model?
#
# Nothing has ever checked the plan's rules against games. The plan was written with the win
# term deliberately removed, and its own docstring says a wrong rule "cannot be caught by
# measurement -- it will simply be learned". This is the measurement.
#
# The rules taken off the model cover those two families plus recon, which is UNCONDITIONAL
# ("every turn it is available") and therefore has nothing to learn -- and s1 is BELOW chance
# on it (41.5% against 49.7%), so the model is actively declining a free ability. They are
# the ones where s1 barely chooses (lift over the chance of picking a conformant option at
# random): energy_line +4.8, energy_focus +11.2, spread_aim +16.4.
#
# Four arms, one process per opponent so all four see the same (seed, seat):
#   s1        the model decides everything                        (baseline)
#   strict    the rule decides those menus outright
#   filter    the rule narrows those menus, the model ranks inside
#   engine    engine_v2 decides THE SAME menus                     (is it the RULE, or just
#                                                                  not-the-model, that helps?)
# The fourth arm matters: without it, a win for `strict` cannot be told apart from "anything
# other than the model is better here".
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
OUT=/root/loop_dusk/gate_rules
GAMES=${GAMES:-120}
RULES=${RULES:-spread_aim,energy_line,energy_focus,recon}
S1=/root/out/dusk_s1
mkdir -p "$OUT"
say() { echo "[grules $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "rules taken off the model: $RULES"
i=0
for OPP in slowking dragapult dudunsparce_box marnie_grimmsnarl; do
    nohup python3 -u tools/gate_protagonist.py \
        --deck dragapult_dusknoir --opp "$OPP" --games "$GAMES" --seed $((7000 + i * 100)) \
        --baseline s1 \
        --arm "s1=hf:$S1@dusk" \
        --arm "strict=planrule:$RULES:hf:$S1@dusk" \
        --arm "filter=planfilter:$RULES:hf:$S1@dusk" \
        --arm "engine=planengine:$RULES:hf:$S1@dusk" \
        --out "$OUT/$OPP.json" > "$OUT/$OPP.log" 2>&1 &
    i=$((i + 1)); sleep 30
done
say "launched $i opponent shards x 4 arms x $GAMES games"; wait

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
order = ["s1", "strict", "filter", "engine"]
arms = [a for a in order if any(k.split("|")[0] == a for k in cells)]
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
print("\n%-8s %8s %11s %8s   %s" % ("arm", "win%", "vs s1", "t", "n"))
base = tot["s1"]
for a in arms:
    v = tot[a]
    wr = 100.0 * sum(v) / max(1, len(v))
    if a == "s1":
        print("%-8s %7.1f%% %11s %8s   %d" % (a, wr, "(baseline)", "", len(v)))
        continue
    dd = [x - y for x, y in zip(v, base)]
    m = sum(dd) / len(dd)
    sd = math.sqrt(sum((x - m) ** 2 for x in dd) / max(1, len(dd) - 1))
    se = sd / math.sqrt(len(dd))
    print("%-8s %7.1f%% %+10.2fpt %+8.2f   %d" % (a, wr, 100 * m, m / se if se else 0, len(v)))
PY
say GATE_RULES_DONE
