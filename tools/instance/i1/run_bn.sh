set -u
for D in rockets_honchkrow ceruledge mega_venusaur ns_zoroark mega_lucario_ctrl slowking trevenant_control crustle alakazam dragapult; do
  for B in 0 1; do
    ENGINE_BODY_NEED=$B PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM=full CUDA_VISIBLE_DEVICES="" \
      python3 /root/probe3.py "$D" 1800 40 2>/dev/null | sed "s/^ARM full */DECK $D  body_need=$B | /"
  done
done
echo BN_DONE
