import os
p = "/root/gend2.sh"
s = open(p).read()

# User directive 08-14: traces made by an older champion are used as they are, not discarded.
# Nothing in the pipeline drops them today -- the puller takes every shard and field_chain
# branches every shard it is given. What was missing is the PROVENANCE: with the champion moving
# every round or two, a round's pairs can come from two or three different policies and the log
# said nothing about it. Stamping the producer into the filename makes the mix visible, and
# leaves the option of capping staleness later without touching the transport.
old = '--out "$OUT/gen_r${R}s$S.jsonl.gz" --trace-out "$OUT/.gtr_r${R}s${S}.part" \\'
new = '--out "$OUT/gen_r${R}s$S.jsonl.gz" --trace-out "$OUT/.gtr_r${R}s${S}.part" \\'
assert s.count(old) == 1, "out anchor"

old2 = '''        P="$OUT/.gtr_r${R}s${S}.part"
        if [ -s "$P" ] && gzip -t "$P" 2>/dev/null; then
            mv -f "$P" "$OUT/gtr_r${R}s$S.jsonl.gz"'''
new2 = '''        P="$OUT/.gtr_r${R}s${S}.part"
        if [ -s "$P" ] && gzip -t "$P" 2>/dev/null; then
            # the producing champion goes in the name: a round on instance1 can branch shards
            # from two or three different champions and nothing else records which
            mv -f "$P" "$OUT/gtr_r${R}s${S}_$(basename "$CKPT").jsonl.gz"'''
assert s.count(old2) == 1, "rename anchor"
s = s.replace(old2, new2)

old3 = '    say "round $R ready: $(ls -1 $OUT/gtr_r${R}s*.jsonl.gz 2>/dev/null | wc -l) shard(s); $(( (STOP - $(date -u +%s)) / 60 )) min left"'
new3 = '    say "round $R ready: $(ls -1 $OUT/gtr_r${R}s*.jsonl.gz 2>/dev/null | wc -l) shard(s) from $(basename "$CKPT"); $(( (STOP - $(date -u +%s)) / 60 )) min left"'
assert s.count(old3) == 1, "say anchor"
s = s.replace(old3, new3)

open(p + ".n", "w").write(s)
os.replace(p + ".n", p)
os.chmod(p, 0o755)
print("gend2 stamps the producing champion into each shard name")
