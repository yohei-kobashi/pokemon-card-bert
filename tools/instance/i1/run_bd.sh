set -u
for D in mega_venusaur ns_zoroark mega_lucario_ctrl; do
  for A in full set:deck_low=20; do
    ENGINE_BODY_NEED=0 PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="$A" CUDA_VISIBLE_DEVICES="" \
      python3 /root/probe3.py "$D" 5400 40 2>/dev/null | sed "s/^ARM /$D  /"
  done
done
echo BD_DONE
