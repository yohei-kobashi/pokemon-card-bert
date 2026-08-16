"""Residual label noise in the card-first tie-breaker.

The sub-index sorts options by (in play, active/bench, energy, remaining HP) and falls back to
the option string. When two board slots hold Pokemon that are IDENTICAL in everything the prompt
shows -- same card, same HP, same attached energy, same tools -- that fallback is arbitrary, and
the label it produces is noise: the model is asked to prefer BENCH0 over BENCH1 on evidence that
does not exist. That is the same defect just measured in the reranker (17.17% of its records).

Counted here only among decisions that actually need a tie-break, since the rest never look at
the order at all.
"""
import gzip, json, re, sys, collections
sys.path.insert(0, ".")
from lm.action_token import first_token, tie_group, equivalent, sort_key, parse_board

RE = re.compile(r"(?:^| )(\d+)=(\S+)")
SLOT = re.compile(r"^(ACTIVE|BENCH)(\d+)$")


def slot_text(prompt):
    """slot -> its full rendered description, so two slots can be compared exactly as the model
    sees them (card, damage, energy, tools, need, retreat)."""
    m = re.search(r" ME (A\[[^\]]*\])(?: (B\[[^\]]*\]))?", prompt)
    out = {}
    if not m:
        return out
    out["ACTIVE0"] = m.group(1)[2:-1]
    if m.group(2):
        for i, s in enumerate(m.group(2)[2:-1].split(",")):
            out["BENCH%d" % i] = s
    return out


def tgt(o):
    return (o.split("@", 1)[1] if "@" in o else "").split("#")[0]


n = need = noisy = 0
kinds = collections.Counter()
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
        grp = tie_group(opts, first_token(opts[k]))
        if len(grp) < 2 or all(equivalent(opts[i], opts[k]) for i in grp):
            continue
        need += 1
        st = slot_text(d["prompt"])
        board = parse_board(d["prompt"])
        order = sorted(grp, key=lambda i: sort_key(opts[i], board))
        r = order.index(k)
        # is the option ranked next to it indistinguishable from the chosen one?
        twin = False
        for j in (r - 1, r + 1):
            if not 0 <= j < len(order):
                continue
            a, b = opts[order[j]], opts[k]
            if a.split(":")[0] != b.split(":")[0]:
                continue
            ta, tb = tgt(a), tgt(b)
            if SLOT.match(ta or "") and SLOT.match(tb or "") and st.get(ta) == st.get(tb) \
                    and st.get(ta) is not None:
                twin = True
        if twin:
            noisy += 1
            kinds[opts[k].split(":")[0]] += 1
        if n >= 400000:
            break

print("decisions %d | need a tie-break %d (%.2f%%)" % (n, need, 100.0 * need / n))
print("of those, the neighbour in the sorted order is INDISTINGUISHABLE in the prompt: "
      "%d (%.2f%% of tie-breaks, %.3f%% of all decisions)"
      % (noisy, 100.0 * noisy / max(1, need), 100.0 * noisy / n))
print("kinds:", ", ".join("%s %d" % kv for kv in kinds.most_common(6)) or "-")
