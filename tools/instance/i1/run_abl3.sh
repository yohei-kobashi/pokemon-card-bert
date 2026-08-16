set -u
R="roles:463=win+473=win+414=win+891=engine+474=engine"
for A in full empty "$R" "no:search_items;$R" "set:deck_low=12;$R" "set:deck_low=20;$R" "set:deck_low=28;$R" "set:deck_low=20" ; do
  PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="$A" CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py rockets_honchkrow 1200 40
done
echo ABL3_DONE
