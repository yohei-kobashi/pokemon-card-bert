"""Token cost of each segment of the v39 prompt, measured on real records.

Any redesign has to start from where the tokens actually are. Segments are split on the markers
lm/serialize.py emits, and counted with the real tokenizer plus the domain vocabulary, because a
card renders as ONE token (c1227) and counting characters would mislead by an order of magnitude.
"""
import gzip, json, re, sys, collections, statistics
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib")
from transformers import AutoTokenizer
from lm.vocab import special_tokens

tk = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Base")
c = json.load(open("data/cardfirst_v39.json"))
tk.add_tokens(list(special_tokens()) + list(c["new_tokens"]) + list(c["sub_tokens"]))
L = lambda s: len(tk(s, add_special_tokens=False)["input_ids"])

seg = collections.defaultdict(list)
menu_opts = []
sample = None
n = 0
with gzip.open("data/sft/v39_dag005.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        if not d.get("target"):
            continue
        p = d["prompt"]
        n += 1
        if sample is None and 6 <= len(re.findall(r"(?:^| )\d+=", p.rsplit(":: ", 1)[-1])) <= 8:
            sample = p
        # markers, in order: [ACT]\n DECK ... T<n>.<n> ME ... | OP ... ID ... || SEL ... :: menu
        head, _, rest = p.partition(" ME ")
        me, _, opp = rest.partition(" | OP ")
        opp_body, _, tail = opp.partition(" || SEL ")
        sel, _, menu = tail.partition(" :: ")
        deck = head.split(" T", 1)[0]
        turn = " T" + head.split(" T", 1)[1] if " T" in head else ""
        for k, v in (("DECK", deck), ("turn", turn), ("ME(board+hand)", me),
                     ("OP(+ID)", opp_body), ("SEL header", sel), ("menu", menu)):
            seg[k].append(L(v))
        menu_opts.append(len(re.findall(r"(?:^| )\d+=", menu)))
        if n >= 4000:
            break

tot = sum(statistics.mean(v) for v in seg.values())
print("mean prompt %.0f tokens over %d records\n" % (tot, n))
print("%-16s %8s %8s %8s" % ("segment", "mean", "p90", "share"))
for k in ("DECK", "turn", "ME(board+hand)", "OP(+ID)", "SEL header", "menu"):
    v = sorted(seg[k])
    print("%-16s %8.1f %8.0f %7.1f%%"
          % (k, statistics.mean(v), v[int(len(v) * 0.9)], 100.0 * statistics.mean(v) / tot))
print("\nmenu: %.2f options on average, %.1f tokens each"
      % (statistics.mean(menu_opts), statistics.mean(seg["menu"]) / statistics.mean(menu_opts)))
print("\n---------------- a real prompt ----------------")
print(sample)
