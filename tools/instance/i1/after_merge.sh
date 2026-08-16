set -u
# Queued behind the merge sweep: both need the GPU, and three scorers per process is already
# what the card will hold.
while ! grep -q "GATE_MERGE_DONE" /root/gate_merge.log 2>/dev/null; do sleep 60; done
echo "[queue] merge gate finished; starting the rule gate"
bash /root/gate_rules.sh
