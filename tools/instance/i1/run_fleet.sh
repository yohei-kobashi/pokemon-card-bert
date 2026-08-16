set -u
python3 - <<PY > /root/fleet_decks.txt
import json
print("\n".join(sorted(json.load(open("/root/ptcg/repo/agents/tuning.json")))))
PY
while read D; do
  PROBE_ROOT=/root/ptcg/repo PROBE_ARM=full CUDA_VISIBLE_DEVICES="" python3 /root/probe3.py "$D" 300 40 2>/dev/null | sed "s/^ARM full */DECK $D | /"
done < /root/fleet_decks.txt
echo FLEET_DONE
