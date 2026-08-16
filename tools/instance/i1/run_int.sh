set -u
for D in ceruledge rockets_honchkrow mega_venusaur ns_zoroark crustle_geco archaludon; do
  for B in 0 1; do
    ENGINE_BODY_NEED=$B PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="set:deck_low=20" CUDA_VISIBLE_DEVICES="" \
      python3 /root/probe3.py "$D" 1800 40 2>/dev/null | sed "s/^ARM .* | n/DECK $D  dl20+bn=$B | n/"
  done
done
echo INT_DONE
