#!/usr/bin/env bash
# honchkrow: static card_roles re-tier vs the need-aware _card_need boost, at EQUAL n.
# Both are attempts at the same defect (searches fetching an unplayable evolution while the bench
# is empty). Earlier readings came from different runs at different sample sizes -- 34.8% for the
# re-tier at 3600 games vs 30.7-33.2% for body_need at 1800 -- so which is better is unresolved.
# deck_low=20 is applied in every arm because it is already established (+5.7pt alone).
set -u
R="roles:463=win+473=win+414=win+891=engine+474=engine"
G=5400
W=40
run () {   # label, PROBE_ARM, ENGINE_BODY_NEED
  ENGINE_BODY_NEED="$3" ENGINE_BODY_NEED_CAP=0 \
  PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="$2" CUDA_VISIBLE_DEVICES="" \
    python3 /root/probe3.py rockets_honchkrow "$G" "$W" 2>/dev/null \
    | sed "s/^ARM.*  n/$1  n/"
}
run "A_dl20_only          " "set:deck_low=20"      0
run "B_dl20_retier        " "set:deck_low=20;$R"   0
run "C_dl20_bodyneed      " "set:deck_low=20"      1
run "D_dl20_retier+bodyneed" "set:deck_low=20;$R"  1
echo HTR_DONE
