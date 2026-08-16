"""The token set that must ship with the model, and how often it would miss on unseen data.

The vocabulary has to cover every token that can appear as an OPTION, not merely the ones the
engine chose: at inference the model scores all legal options, and an option whose token is
absent simply cannot be picked. So the set is built from the option side, and the miss rate is
estimated the only honest way -- build on one slice, measure on a later, disjoint one.
"""
import gzip, json, re, sys, collections
sys.path.insert(0, ".")
from lm.action_token import action_token

RE = re.compile(r"(?:^| )(\d+)=(\S+)")
BUILD, TEST = 1_200_000, 300_000
vocab = collections.Counter()
rows = []
n = 0
with gzip.open("data/sft/v39_dag005.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        if not d.get("target"):
            continue
        opts = [o for _, o in RE.findall(d["prompt"].rsplit(":: ", 1)[-1])]
        n += 1
        if n <= BUILD:
            for o in opts:
                vocab[action_token(o)] += 1
        else:
            rows.append((opts, int(d["target"])))
            if len(rows) >= TEST:
                break
print("built on %d decisions -> %d distinct option tokens" % (BUILD, len(vocab)))
known = set(vocab)
miss_opt = miss_lab = dec_any = dec_all = 0
tot_opt = 0
for opts, k in rows:
    toks = [action_token(o) for o in opts]
    unk = [t for t in toks if t not in known]
    tot_opt += len(toks)
    miss_opt += len(unk)
    if unk:
        dec_any += 1
    if len(unk) == len(toks):
        dec_all += 1
    if k < len(toks) and toks[k] not in known:
        miss_lab += 1
m = len(rows)
print("\nheld-out %d decisions / %d options" % (m, tot_opt))
print("  options with an unknown token      %d (%.3f%%)" % (miss_opt, 100.0 * miss_opt / tot_opt))
print("  decisions with ANY unknown option  %d (%.3f%%)  <- model cannot pick that option"
      % (dec_any, 100.0 * dec_any / m))
print("  decisions where EVERY option is unknown %d (%.4f%%)  <- must defer to engine_v2"
      % (dec_all, 100.0 * dec_all / m))
print("  decisions where the CORRECT option is unknown %d (%.3f%%)" % (miss_lab, 100.0 * miss_lab / m))
c = sorted(vocab.values(), reverse=True)
print("\noption-side frequency: median %d | seen<5 %d (%.0f%%) | seen<20 %d (%.0f%%)"
      % (c[len(c) // 2], sum(1 for x in c if x < 5), 100.0 * sum(1 for x in c if x < 5) / len(c),
         sum(1 for x in c if x < 20), 100.0 * sum(1 for x in c if x < 20) / len(c)))
