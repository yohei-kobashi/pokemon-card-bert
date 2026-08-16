#!/usr/bin/env bash
# One-card A/B: Handheld Fan vs a 2nd Budew, piloted by engine_v2, across the live top decks.
#
# WHY ENGINE_V2 AND NOT THE LM. This is a DECK question, and engine_v2 needs no GPU, so it buys
# ~10x the games for the same wall clock and leaves both cards' cards to be drawn and played by
# the same deterministic pilot. It is a screen, not a verdict: a card whose value depends on
# being played WELL (Budew's Itchy Pollen is an attack the pilot must choose to make) can look
# worse here than it is. If the two land within noise, the honest answer is "this swap is not
# worth a deck change", not "Budew is bad".
#
# WHY THIS PAIR. Handheld Fan moves an Energy off the attacker when our Active is damaged. Read
# against the ladder it is aimed at the wrong decks: marnie is 36.2% of the field, attacks for
# {D}{D}, and refills with Punk Up (5 basic {D} out of the DECK at once); ogerpon_mono runs 22
# energy. It bites alakazam_nz (8 energy, Powerful Hand costs a single {P}) and dudunsparce_box
# -- 22% of the ladder, and both are matchups we already win. Budew's Item lock instead attacks
# marnie's only route around its 2-Morgrem bottleneck: 3 Rare Candy.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
O=/root/loop_dusk/card_ab; mkdir -p $O
GAMES=${GAMES:-600}
OPPS=${OPPS:-marnie_grimmsnarl,alakazam_nz,ogerpon_mono,dudunsparce_box}
say() { echo "[cardab $(date -u +%m-%d_%H:%M:%S)] $*"; }

for D in dragapult_dusknoir dragapult_dusknoir_budew; do
    if [ -s "$O/$D.json" ]; then say "$D already measured"; continue; fi
    say "$D vs $OPPS, $GAMES games per opponent, engine_v2 both sides"
    CUDA_VISIBLE_DEVICES= nice -n 10 python3 -u tools/gate_protagonist.py \
        --deck "$D" --opp "$OPPS" --games "$GAMES" --seed 123000 \
        --baseline eng --arm "eng=engine" \
        --mirror-so "$REPO/data/kaggle_engine_ext/libcg_mirror.so" \
        --out "$O/$D.json" > "$O/$D.log" 2>&1 || say "$D FAILED"
    grep -aE "vs " "$O/$D.log" | tail -6
done

python3 - "$O/dragapult_dusknoir.json" "$O/dragapult_dusknoir_budew.json" <<'PY'
import json, sys, math
def cells(p):
    try:
        j = json.load(open(p))
    except Exception:
        return {}
    return {k.split("|", 1)[1]: v for k, v in (j.get("cells") or {}).items()}
a, b = cells(sys.argv[1]), cells(sys.argv[2])
print("\n  %-22s %10s %10s %9s" % ("opponent", "Fan", "Budew x2", "delta"))
ta = tb = na = nb = 0
for opp in sorted(set(a) | set(b)):
    ca, cb = a.get(opp), b.get(opp)
    if not ca or not cb:
        continue
    wa, ga = ca["win"], ca["games"]
    wb, gb = cb["win"], cb["games"]
    ta += wa; na += ga; tb += wb; nb += gb
    # independent samples: different decklists get different shuffles, so this is not paired
    pa, pb = wa / ga, wb / gb
    se = math.sqrt(pa * (1 - pa) / ga + pb * (1 - pb) / gb)
    print("  %-22s %9.1f%% %9.1f%% %+8.2f  (+-%.2f)"
          % (opp, 100 * pa, 100 * pb, 100 * (pb - pa), 100 * se))
if na and nb:
    pa, pb = ta / na, tb / nb
    se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    d = 100 * (pb - pa)
    print("  %-22s %9.1f%% %9.1f%% %+8.2f  (+-%.2f)  t %+.2f"
          % ("OVERALL", 100 * pa, 100 * pb, d, 100 * se, d / (100 * se) if se else 0))
    print("\nVERDICT:", "swap in the 2nd Budew" if d > 2 * 100 * se
          else "no measurable difference -- keep Handheld Fan")
PY
say "CARD_AB_DONE"
