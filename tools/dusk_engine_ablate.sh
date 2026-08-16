#!/usr/bin/env bash
# Ablation over DuskNoirL2's rules. Every arm plays the SAME seeds against the same
# opponent pilots, so the arms are paired game-for-game and the null (rules off) is the
# measured 3.0% rather than an assumed number.
#
#   bash tools/dusk_engine_ablate.sh 400 ogerpon_mono
set -u
G=${1:-400}
OPP=${2:-ogerpon_mono}
SETS=${SETS:-"off front charge search bench front,charge front,charge,search front,charge,search,bench front,charge,search,bench,cap=2"}
cd "$(dirname "$0")/.."
printf '%-38s %s\n' "rules" "win% vs $OPP ($G games)"
for s in $SETS; do
  if [ "$s" = "off" ]; then unset DUSK_RULES; else export DUSK_RULES="$s"; fi
  out=$(PYTHONPATH=cg-lib python3 tools/gate_protagonist.py --deck dragapult_dusknoir \
        --opp "$OPP" --arm engine=engine@prompt --games "$G" --seed 1 2>&1 \
        | awk '/vs /{print $(NF-2), $NF}')
  printf '%-38s %s\n' "$s" "$out"
done
