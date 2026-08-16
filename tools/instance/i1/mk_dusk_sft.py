"""Re-render instance1's SFT pool in the single-deck prompt format (no DECK[] segment).

Re-rendering, not re-playing: v41_dusk.jsonl.gz already holds 40M dusknoir-piloted decisions,
and the format change only alters how each one is written down. But the pool stores the RENDERED
prompt, not the observation, so the segment is stripped textually -- the segment is a fixed
prefix between "[ACT]\\n" and the "T<turn>." marker, which makes the edit exact rather than a
guess. Verified on 200 rows before the pool is touched.
"""
import gzip, json, re, sys

SRC, DST, LIMIT = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0
DECK_SEG = re.compile(r"^DECK .*?(?=T\d+\.)", re.S)

n = w = miss = 0
with gzip.open(SRC, "rt") as f, gzip.open(DST, "wt") as g:
    for line in f:
        n += 1
        d = json.loads(line)
        st = d.get("state") or ""
        new, k = DECK_SEG.subn("", st, count=1)
        if k == 0:
            miss += 1
            if miss <= 3:
                print("  no DECK segment: %r" % st[:80], file=sys.stderr)
        d["state"] = new
        g.write(json.dumps(d, ensure_ascii=False) + "\n")
        w += 1
        if LIMIT and w >= LIMIT:
            break
print("rewrote %d rows (%d had no DECK segment) -> %s" % (w, miss, DST))
