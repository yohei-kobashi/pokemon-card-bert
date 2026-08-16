set -u
# decks the screen did NOT flag (library outlasts the game) -- regression check
for D in crustle dragapult alakazam mega_lucario_ctrl marnie_grimmsnarl crustle_stall; do
  for A in full set:deck_low=20; do
    PROBE_ROOT=/root/ptcg/repo_fix PROBE_ARM="$A" CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py "$D" 1800 40 2>/dev/null | sed "s/^ARM /DECK $D  ARM /"
  done
done
echo REG_DONE
