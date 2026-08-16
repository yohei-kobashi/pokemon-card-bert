set -u
for A in full empty no:draw_supporters no:search_items no:card_roles no:line no:l2 set:draw_threshold=3 set:draw_threshold=2 no:draw_supporters+search_items; do
  PROBE_ROOT=/root/ptcg/repo PROBE_ARM="$A" CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py rockets_honchkrow 1200 40
done
echo ABL_DONE
