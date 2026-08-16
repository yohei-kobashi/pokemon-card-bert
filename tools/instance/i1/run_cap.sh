set -u
for D in archaludon rockets_honchkrow ceruledge ns_zoroark crustle_geco mega_venusaur crustle alakazam; do
  for C in 0_0 1_0 1_2; do
    B=${C%%_*}; CAP=${C##*_}
    ENGINE_BODY_NEED=$B ENGINE_BODY_NEED_CAP=$CAP PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="set:deck_low=20" \
      CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py "$D" 1800 40 2>/dev/null \
      | sed "s/^ARM.*  n/$D bn=$B cap=$CAP n/"
  done
done
echo CAP_DONE
