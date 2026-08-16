set -u
PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="full" CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py rockets_mewtwo 7200 40
PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="set:deck_low=20" CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py rockets_mewtwo 7200 40
echo MW_DONE
