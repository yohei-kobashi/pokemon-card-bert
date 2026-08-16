"""Let field_chain branch instance2's generated traces alongside its own.

instance2 now plays dragapult_dusknoir with the DeBERTa champion behind the plan filter against
each opponent deck's own 4B adapter, and instance1 pulls the finished shards into /root/gen_in.
Those games are the same protagonist policy this loop is training, so their branch points are
on-policy for it -- what they add is opponents that are PLAYED rather than held: every gate and
every collection so far has faced engine_v2.

Consumed shards move to /root/gen_used so the next round branches new games rather than
re-branching the same states with a fresh budget.
"""
import os

p = "/root/field_chain.sh"
s = open(p).read()

old = """        CUDA_VISIBLE_DEVICES= nice -n 5 python3 tools/dpo_branch.py \\
            --traces "$TR" --fmt dusk --only-deck "$DECK" --rule-weights --rule-exclude "$WRAP_RULES" \\"""
new = """        # instance2's shards, if its generator has delivered any. Capped per round so one big
        # backlog cannot make a single round's branch run for hours.
        TRALL="$TR"
        GEN=$(ls -1 /root/gen_in/gtr_*.jsonl.gz 2>/dev/null | head -${GEN_MAX:-12} | paste -sd,)
        if [ -n "$GEN" ]; then
            TRALL="$TR,$GEN"
            say "branching with $(echo "$GEN" | tr , '\\n' | wc -l) shard(s) from instance2"
        fi
        CUDA_VISIBLE_DEVICES= nice -n 5 python3 tools/dpo_branch.py \\
            --traces "$TRALL" --fmt dusk --only-deck "$DECK" --rule-weights --rule-exclude "$WRAP_RULES" \\"""
assert s.count(old) == 1, "branch anchor"
s = s.replace(old, new)

old2 = """        grep -aE "^wrote|selected" "$STATE/branch$R.log" | tail -2"""
new2 = """        grep -aE "^wrote|selected" "$STATE/branch$R.log" | tail -2
        # Retire what was just branched: a shard left in place would be re-branched next round,
        # spending the budget on states this round already mined.
        if [ -n "$GEN" ]; then
            mkdir -p /root/gen_used
            echo "$GEN" | tr , '\\n' | xargs -r -I{} mv -f {} /root/gen_used/ 2>/dev/null || true
        fi"""
assert s.count(old2) == 1, "retire anchor"
s = s.replace(old2, new2)

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
os.chmod(p, 0o755)
print("field_chain now branches /root/gen_in shards and retires them after use")
