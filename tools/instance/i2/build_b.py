"""Price scheme B and freeze its vocabulary."""
import gzip, json, re, sys, statistics, collections
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib")
from transformers import AutoTokenizer
from lm.vocab import special_tokens
from lm.action_token import to_scheme_b, label_b, first_token, second_token, groups

RE = re.compile(r"(?:^| )(\d+)=(\S+)")
tk0 = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Base")
firsts, seconds = collections.Counter(), collections.Counter()
cur_menu, new_menu, cur_all, new_all = [], [], [], []
two = n = 0
with gzip.open("data/sft/v39_dag005.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        t = d.get("target")
        if not t:
            continue
        p = d["prompt"]
        opts = [o for _, o in RE.findall(p.rsplit(" :: ", 1)[-1])]
        k = int(t)
        if k >= len(opts):
            continue
        n += 1
        for tok, os_ in groups(opts):
            firsts[tok] += 1
            if len(os_) > 1:
                for o in os_:
                    seconds[second_token(o)] += 1
        a, b = label_b(None, k, opts)
        if b is not None:
            two += 1
        if n <= 4000:
            cur_menu.append(p.rsplit(" :: ", 1)[-1])
            new_menu.append(to_scheme_b(p).rsplit(" :: ", 1)[-1])
            cur_all.append(p); new_all.append(to_scheme_b(p))
        if n >= 2521800:
            break

known = set(special_tokens())
new_first = sorted(t for t in firsts if t not in known)
sec = sorted(seconds)
json.dump({"scheme": "b", "first_tokens": sorted(firsts), "new_tokens": new_first,
           "second_tokens": sec, "decisions": n,
           "counts": {t: firsts[t] for t in firsts},
           "second_counts": {t: seconds[t] for t in seconds}},
          open("data/cardfirst_b_v39.json", "w"))

tk = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Base")
tk.add_tokens(list(special_tokens()) + new_first + sec)
L = lambda s: len(tk(s, add_special_tokens=False)["input_ids"])
print("decisions %d" % n)
print("first tokens  %d (%d new)   second tokens %d" % (len(firsts), len(new_first), len(sec)))
print("second token needed on %d decisions (%.2f%%) -> %.3f forwards each"
      % (two, 100.0 * two / n, 1 + two / n))
c = sorted(seconds.values())
if c:
    print("  second-token frequency: median %d  min %d  seen<20 %d of %d"
          % (c[len(c) // 2], c[0], sum(1 for x in c if x < 20), len(c)))
mc = statistics.mean(L(s) for s in cur_menu)
mn = statistics.mean(L(s) for s in new_menu)
pc = statistics.mean(L(s) for s in cur_all[:1500])
pn = statistics.mean(L(s) for s in new_all[:1500])
print("\nmenu tokens   %.1f -> %.1f  (%+.1f)" % (mc, mn, mn - mc))
print("prompt tokens %.1f -> %.1f  (%.1f%% shorter)" % (pc, pn, 100.0 * (pc - pn) / pc))
print("\nexamples:")
for i in (0, 5, 11):
    print("  A: %s\n  B: %s" % (cur_menu[i][:110], new_menu[i][:110]))
