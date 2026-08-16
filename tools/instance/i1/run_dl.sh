set -u
for D in ceruledge archaludon ns_zoroark mega_venusaur mega_absol mega_zygarde crustle_geco; do
  for A in full set:deck_low=20; do
    PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="$A" CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py "$D" 1800 40 2>/dev/null | sed "s/^ARM /DECK $D  ARM /"
  done
done
PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="set:deck_low=12" CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py ceruledge 1800 40 2>/dev/null | sed "s/^ARM /DECK ceruledge  ARM /"
echo DL_DONE
