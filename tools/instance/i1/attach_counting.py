"""Is the reranker's attach weakness a COUNTING problem?

`attach-decisions-at-chance`: energy-attach top1 is 29.1% (v37) against a 14.1% chance level,
while every other decision kind runs +25 to +79pt above chance. Today's tie audit raised the
honest ceiling from 100% to 78.9%, which shrank the deficit but did not remove it.

One candidate explanation: attached energy renders as one letter per copy (`|GGGGGGGGGGGG`) and
attack costs render the same way (`[FFC]`), so deciding whether a target can pay for its attack
means COUNTING two repeated-character strings and comparing them -- the classic weak spot for a
tokenizer-based model. If that is the cause, accuracy must fall as the counting load rises.

Design. Bucket attach decisions by the LOAD, defined as the largest number of energies already
attached to any candidate target, then report per bucket:

  * top1 of the v37 reranker,
  * the CEILING, because candidates that render identically in the prompt are indistinguishable
    to the model by construction: ceiling = E[1 / size of the tie group holding the label],
  * accuracy as a fraction of that ceiling -- the only comparable number across buckets.

Counting predicts a falling accuracy/ceiling as load grows. A flat profile refutes it and points
elsewhere (target identity, or the label itself being arbitrary).

Run:  python attach_counting.py <model_dir> [n_records]
"""
import collections
import gzip
import json
import os
import re
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/root/out/rerank_gte_v37"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400000
MAXPAIRS = int(os.environ.get("MAXPAIRS", "6000"))

RE_MENU = re.compile(r"(?:^| )(\d+)=(\S+)")
RE_ME = re.compile(r" ME A\[([^\]]*)\](?: B\[([^\]]*)\])?")
RE_TARGET = re.compile(r"@(ACTIVE|BENCH)(\d*)$")
RE_TURN = re.compile(r" T(\d+)\.")


def board_entries(prompt):
    """The rendered ME active + bench entries, exactly as the model sees them."""
    m = RE_ME.search(prompt)
    if not m:
        return None, []
    act = m.group(1)
    bench = [x for x in (m.group(2) or "").split(",") if x]
    return act, bench


def energy_count(entry):
    """How many energies are attached, from the rendered `|LLL` suffix."""
    if entry is None or "|" not in entry:
        return 0
    return len(re.sub(r"[^A-Z]", "", entry.rsplit("|", 1)[-1]))


def target_entry(enc, act, bench):
    m = RE_TARGET.search(enc)
    if not m:
        return None
    if m.group(1) == "ACTIVE":
        return act
    i = int(m.group(2) or 0)
    return bench[i] if i < len(bench) else None


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = []
    n = 0
    for line in gzip.open("data/sft/teacher_0730.jsonl.gz", "rt"):
        d = json.loads(line)
        n += 1
        if n > N:
            break
        p = d["prompt"]
        ents = RE_MENU.findall(p.rsplit(":: ", 1)[-1])
        if len(ents) < 2:
            continue
        try:
            tgt = int(d["target"])
        except (TypeError, ValueError):
            continue
        enc = dict((int(i), e) for i, e in ents)
        if tgt not in enc or not enc[tgt].startswith("attach:"):
            continue
        att = {i: e for i, e in enc.items() if e.startswith("attach:")}
        if len(att) < 2:
            continue
        act, bench = board_entries(p)
        te = {i: target_entry(e, act, bench) for i, e in att.items()}
        if any(v is None for v in te.values()):
            continue
        load = max(energy_count(v) for v in te.values())
        # tie group = candidates whose TARGET RENDERS IDENTICALLY (indistinguishable in-prompt)
        groups = collections.defaultdict(list)
        for i, v in te.items():
            groups[(att[i].split("@")[0], v)].append(i)
        gsize = next(len(v) for v in groups.values() if tgt in v)
        mt = RE_TURN.search(p)
        turn = int(mt.group(1)) if mt else -1
        rows.append((p, att, tgt, load, gsize, turn))
        if len(rows) >= MAXPAIRS:
            break

    print("attach decisions with >=2 attach candidates: %d (scanned %d records)"
          % (len(rows), n - 1), flush=True)
    if len(rows) < 200:
        print("too few")
        return

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()

    buckets = collections.defaultdict(lambda: [0, 0, 0.0])     # ok, n, ceiling_sum
    with torch.no_grad():
        for p, att, tgt, load, gsize, turn in rows:
            idx = sorted(att)
            pairs = [[p, att[i]] for i in idx]
            e = tok(pairs, padding=True, truncation="only_first", max_length=1024,
                    return_tensors="pt").to("cuda")
            s = model(**e).logits.squeeze(-1).float().tolist()
            if not isinstance(s, list):
                s = [s]
            best = idx[max(range(len(idx)), key=lambda j: s[j])]
            tb = 0 if turn <= 3 else (1 if turn <= 6 else (2 if turn <= 10 else 3))
            b = (min(load, 3), tb)
            buckets[b][0] += int(best == tgt)
            buckets[b][1] += 1
            buckets[b][2] += 1.0 / gsize

    TL = ("T1-3", "T4-6", "T7-10", "T11+")
    print("\n  accuracy AS A FRACTION OF THE CEILING, load x turn")
    print("  %-6s %s" % ("load", "".join("%14s" % t for t in TL)))
    for L in range(4):
        cells = []
        for T in range(4):
            ok, nn, ce = buckets.get((L, T), [0, 0, 0.0])
            if nn < 40:
                cells.append("%14s" % ("n=%d" % nn))
            else:
                cells.append("%12.0f%% " % (100.0 * (ok / nn) / max(1e-9, ce / nn)))
        print("  %-6s %s" % (">=3" if L == 3 else str(L), "".join(cells)))
    print("\n  counts per cell:")
    for L in range(4):
        print("  %-6s %s" % (">=3" if L == 3 else str(L),
              "".join("%14d" % buckets.get((L, T), [0, 0, 0])[1] for T in range(4))))
    print("\n  READ DOWN a turn column: if load still degrades WITHIN a turn band, counting survives. Flat down the columns means the earlier gradient was game phase, not counting.")


if __name__ == "__main__":
    main()
