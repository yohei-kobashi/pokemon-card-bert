"""Which sort makes <s0> right most often?

The tie-breaker is only worth having if its order carries information; if it did not, the answer
would be spread evenly and the model would be back to memorising positions. `menu` is the
baseline -- the order the engine happens to list options in -- so the gap between it and a sorted
variant is exactly what the sort buys.
"""
import gzip, json, re, sys, collections
sys.path.insert(0, ".")
from lm.action_token import equivalent
sys.path.insert(0, "/root")
from cardfirst import first_token, parse_board, RE, SLOT

VARIANTS = {
    "menu (baseline)":        lambda o, b, i: (i,),
    "play,energy,hp":         lambda o, b, i: _k(o, b, ("play", "e", "hp")),
    "play,hp,energy":         lambda o, b, i: _k(o, b, ("play", "hp", "e")),
    "play,energy,hpfrac":     lambda o, b, i: _k(o, b, ("play", "e", "hpf")),
    "active,energy,hp":       lambda o, b, i: _k(o, b, ("act", "e", "hp")),
}


def _slot(o):
    t = o.split("@", 1)[1] if "@" in o else ""
    return t.split("#")[0]


def _k(o, board, keys):
    t = _slot(o)
    m = SLOT.match(t)
    e, hp = board.get(t, (0, 9999))
    mx = board.get(t + "|max", (0, 9999))[1]
    out = []
    for k in keys:
        if k == "play":
            out.append(0 if m else 1)
        elif k == "act":
            out.append(0 if (m and m.group(1) == "ACTIVE") else (1 if m else 2))
        elif k == "e":
            out.append(-e)
        elif k == "hp":
            out.append(hp)
        elif k == "hpf":
            out.append(hp / max(1, mx))
    out.append(o)
    return tuple(out)


res = collections.defaultdict(collections.Counter)
n = need2 = 0
with gzip.open("data/sft/v39_dag005.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        t = d.get("target")
        if not t:
            continue
        opts = [o for _, o in RE.findall(d["prompt"].rsplit(":: ", 1)[-1])]
        k = int(t)
        if k >= len(opts):
            continue
        n += 1
        ft = first_token(opts[k])
        same = [(i, o) for i, o in enumerate(opts) if first_token(o) == ft]
        if len(same) < 2 or all(equivalent(o, opts[k]) for _, o in same):
            continue
        need2 += 1
        board = parse_board(d["prompt"])
        for name, fn in VARIANTS.items():
            order = sorted(same, key=lambda io: fn(io[1], board, io[0]))
            res[name][[o for _, o in order].index(opts[k])] += 1
        if n >= 400000:
            break

print("decisions %d | tie-break needed %d (%.2f%%)\n" % (n, need2, 100.0 * need2 / n))
print("%-22s %8s %8s %8s" % ("sort", "<s0>", "<=s1>", "<=s2>"))
for name in VARIANTS:
    c = res[name]
    tot = sum(c.values())
    print("%-22s %7.1f%% %7.1f%% %7.1f%%"
          % (name, 100.0 * c[0] / tot, 100.0 * (c[0] + c[1]) / tot,
             100.0 * (c[0] + c[1] + c[2]) / tot))
