set -u
for D in rockets_honchkrow rockets_mewtwo; do
  echo "@@@@@ $D  [A] BASELINE profile ON @@@@@"
  PROBE_ROOT=/root/ptcg/repo PROBE_PROFILE=1 CUDA_VISIBLE_DEVICES="" python3 /root/probe2.py "$D" 2700 40 | grep -E "^RESULT|Proton|Ultra Ball|Poke Pad"
  echo "@@@@@ $D  [B] FIXED profile ON + basic_search @@@@@"
  PROBE_ROOT=/root/ptcg/repo_fix PROBE_PROFILE=1 CUDA_VISIBLE_DEVICES="" python3 /root/probe2.py "$D" 2700 40 | grep -E "^RESULT|Proton|Ultra Ball|Poke Pad"
done
echo AB3_DONE
