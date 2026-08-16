"""Prototype the next-generation menu and price it against the current one.

Three changes, each of which follows from the answer being a CARD rather than a position:

  1. drop the `N=` indices     the model never emits a position, and a numbered list is exactly
                               the cue that would let it learn one
  2. dedup by equivalence      the current menu shows `play:c1227` twice while the answer space
                               contains `c1227` once -- an inconsistency the model has to resolve
                               silently, and the same defect just removed from the reranker
  3. group by card, targets in the SORT ORDER
                               the answer is (card, then which target), so the menu is rendered
                               in that shape. The tie-break order stops being something the model
                               must re-derive from the board segment and becomes something it
                               reads: the first target listed IS <s0>.

Only the menu changes; DECK / board / ID are untouched, so this stays comparable with v39.
"""
import gzip, json, re, sys, statistics, collections
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib")
from transformers import AutoTokenizer
from lm.vocab import special_tokens
from lm.action_token import first_token, equivalent, sort_key, parse_board

tk = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Base")
_c = json.load(open("data/cardfirst_v39.json"))
tk.add_tokens(list(special_tokens()) + list(_c["new_tokens"]) + list(_c["sub_tokens"]))
L = lambda s: len(tk(s, add_special_tokens=False)["input_ids"])
RE = re.compile(r"(?:^| )(\d+)=(\S+)")


def render_v41(opts, board):
    """Grouped by first token; within a group, the equivalence-deduped targets in sort order."""
    groups = collections.OrderedDict()
    for o in opts:
        groups.setdefault(first_token(o), []).append(o)
    out = []
    for tok, os_ in groups.items():
        keep = []
        for o in os_:
            if not any(equivalent(o, k) for k in keep):
                keep.append(o)
        keep.sort(key=lambda o: sort_key(o, board))
        kind = keep[0].split(":", 1)[0]
        tgts = [(o.split("@", 1)[1].split("#")[0] if "@" in o else "") for o in keep]
        if len(keep) == 1:
            out.append("%s>%s%s" % (tok, kind, (" " + tgts[0]) if tgts[0] else ""))
        else:
            out.append("%s>%s %s" % (tok, kind, " ".join(t or "-" for t in tgts)))
    return " | ".join(out)


cur, new, nopt_cur, nopt_new = [], [], [], []
samples = []
n = 0
with gzip.open("data/sft/v39_dag005.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        if not d.get("target"):
            continue
        p = d["prompt"]
        menu = p.rsplit(" :: ", 1)[-1]
        opts = [o for _, o in RE.findall(menu)]
        if not opts:
            continue
        n += 1
        board = parse_board(p)
        v41 = render_v41(opts, board)
        cur.append(L(menu)); new.append(L(v41))
        nopt_cur.append(len(opts))
        nopt_new.append(len(set(first_token(o) for o in opts)))
        if len(samples) < 3 and 5 <= len(opts) <= 9 and len(opts) != nopt_new[-1]:
            samples.append((menu, v41))
        if n >= 4000:
            break

mc, mn = statistics.mean(cur), statistics.mean(new)
print("menu tokens   current %.1f -> proposed %.1f   (%+.1f, %.1f%% of a 352-token prompt)"
      % (mc, mn, mn - mc, 100.0 * (mn - mc) / 352))
print("answer space  options %.2f -> distinct cards %.2f"
      % (statistics.mean(nopt_cur), statistics.mean(nopt_new)))
print("whole prompt  352 -> %.0f tokens (%.1f%% shorter)" % (352 + mn - mc, 100.0 * (mc - mn) / 352))
for a, b in samples:
    print("\n--- current ---\n%s\n--- proposed ---\n%s" % (a, b))
