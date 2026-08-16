set -u
echo "--- honchkrow ---"
PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="full" CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py rockets_honchkrow 3600 40
PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="set:deck_low=20;roles:463=win+473=win+414=win+891=engine+474=engine" CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py rockets_honchkrow 3600 40
echo "--- mewtwo ---"
PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="full" CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py rockets_mewtwo 3600 40
PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="set:deck_low=20" CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py rockets_mewtwo 3600 40
echo CONF_DONE
