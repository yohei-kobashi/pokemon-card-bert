set -u
for D in rockets_honchkrow rockets_mewtwo; do
  echo "@@@@@ $D  [C] profile OFF (what I measured before) @@@@@"
  PROBE_ROOT=/root/ptcg/repo PROBE_PROFILE=0 CUDA_VISIBLE_DEVICES="" python3 /root/probe2.py "$D" 900 40
  echo "@@@@@ $D  [A] BASELINE, profile ON (the shipped agent) @@@@@"
  PROBE_ROOT=/root/ptcg/repo PROBE_PROFILE=1 CUDA_VISIBLE_DEVICES="" python3 /root/probe2.py "$D" 900 40
  echo "@@@@@ $D  [B] FIXED, profile ON + basic_search @@@@@"
  PROBE_ROOT=/root/ptcg/repo_fix PROBE_PROFILE=1 CUDA_VISIBLE_DEVICES="" python3 /root/probe2.py "$D" 900 40
done
echo AB2_DONE
