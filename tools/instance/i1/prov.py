import os
p = "/root/field_chain.sh"
s = open(p).read()
old = '''            say "branching with $(echo "$GEN" | tr , '\\n' | wc -l) shard(s) from instance2"'''
new = '''            # Provenance, not a filter. User directive 08-14: traces from an older champion
            # are used as they are. What matters is that the log SAYS which policies produced
            # this round's pairs, so a surprising round can be read against the mix.
            say "branching with $(echo "$GEN" | tr , '\\n' | wc -l) shard(s) from instance2: $(echo "$GEN" | tr , '\\n' | sed 's/.*_\\(fld_[a-z0-9]*\\)\\.jsonl\\.gz/\\1/' | sort | uniq -c | tr '\\n' ' ')"'''
assert s.count(old) == 1, "say anchor"
s = s.replace(old, new)
open(p + ".n", "w").write(s)
os.replace(p + ".n", p)
os.chmod(p, 0o755)
print("field_chain logs which champions produced the shards it branches")
