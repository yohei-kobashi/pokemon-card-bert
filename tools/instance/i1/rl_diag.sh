#!/usr/bin/env bash
# Seven policy-gradient rounds, all from d41_r8, all lost: -12.9 -8.2 -12.5 -8.0 -4.5 -6.5.
# Same recipe every time, so that spread is the recipe's own reproducibility, not luck -- the
# update makes the model about 9pt worse on the gate.
#
# The training logs point at one suspect. Entropy did not move across a whole round
# (H 1.15 -> 1.17) while KL stayed at 0.053: the update is not sharpening the policy at all.
# And there is a structural mismatch behind that -- rollouts SAMPLE at temperature 1.0 while
# the gate plays ARGMAX, so an entropy bonus that keeps the distribution flat is paying us to
# be worse at the thing we are scored on.
#
# ONE rollout, three arms, so the data is identical and only the update differs.
set -u
say() { echo "[diag $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
ROLL=/root/rl/roll_8.jsonl.gz
BASE=/root/out/d41_r8
[ -s "$ROLL" ] || { say "no rollout at $ROLL"; exit 1; }

run() {   # tag  beta_h  lr
  say "arm $1: beta_h=$2 lr=$3"
  PYTHONPATH=cg-lib python3 tools/rl_pg_train.py --data "$ROLL" --model "$BASE" --ref "$BASE" \
    --out /root/out/rl_$1 --epochs 1 --lr "$3" --beta-kl 0.05 --beta-h "$2" \
    > /root/rl/diag_$1.log 2>&1 || { say "arm $1 train FAILED"; tail -4 /root/rl/diag_$1.log; return; }
  grep -aE "FINAL" /root/rl/diag_$1.log
  PYTHONPATH=cg-lib python3 tools/mirror_match.py --deck dragapult_dusknoir --a engine \
    --b "hf:/root/out/rl_$1" --max-games 400 --mirror --seed 1 --mirror-so "$SO" \
    --out /root/rl/diag_$1.json > /root/rl/diag_gate_$1.log 2>&1
  python3 -c "
import json
d=json.load(open('/root/rl/diag_$1.json'))['decks']['dragapult_dusknoir']
print('  arm $1: %.1f%% (%d-%d)  vs incumbent 45.5%%  delta %+.1fpt' %
      (100*d['p'], d['w'], d['l'], 100*d['p']-45.5))"
  rm -rf /root/out/rl_$1
}

run noent   0.0   5e-6      # entropy bonus off: match the greedy evaluation
run slow    0.0   2e-6      # ...and a quarter of the step, in case it is plain drift
run keep    0.01  5e-6      # the shipped recipe, re-run on THIS rollout as the control
say DIAG_DONE
