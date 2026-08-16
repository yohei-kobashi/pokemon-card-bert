"""Which decision kinds are actually weak, once the tie ceiling is removed?

`attach-decisions-at-chance` ranked kinds by RAW top1, and today's audits showed that number is
biased: candidates that render identically in the prompt are indistinguishable to the model, so a
kind whose menus are full of duplicates looks bad without being bad. Attach turned out to be
genuinely weak (43.7% of its ceiling, and the cause is now isolated to energy-vs-cost reading),
but the same correction has never been applied to the other kinds -- `play` at 56.4% raw could be
the next attach or could be almost entirely ties.

Per kind: raw top1, the in-prompt tie ceiling E[1/|tie group holding the label|], and accuracy as
a fraction of that ceiling. Ranked by the last column, which is the only comparable one.

Two candidates count as tied when their rendered option strings are IDENTICAL -- that is exactly
what the model sees, so it cannot prefer one over the other by any means.

Run:  python kind_ceiling.py <model_dir> [n_records]
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
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000000
PER_KIND = int(os.environ.get("PER_KIND", "2200"))

RE_MENU = re.compile(r"(?:^| )(\d+)=(\S+)")


def kind_of(enc):
    head = enc.split(":")[0].split("@")[0]
    return head if head else "?"


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = collections.defaultdict(list)
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
        enc = {int(i): e for i, e in ents}
        if tgt not in enc:
            continue
        k = kind_of(enc[tgt])
        if len(rows[k]) >= PER_KIND:
            if all(len(v) >= PER_KIND for v in rows.values()) and len(rows) > 6:
                break
            continue
        # tie group among ALL candidates that render identically to the labelled one
        same = [i for i, e in enc.items() if e == enc[tgt]]
        rows[k].append((p, enc, tgt, len(same)))

    print("scanned %d records | kinds %s" % (n - 1, {k: len(v) for k, v in rows.items()}),
          flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()

    out = []
    with torch.no_grad():
        for k, rs in rows.items():
            if len(rs) < 200:
                continue
            ok = 0
            ce = 0.0
            nc = 0.0
            rnd = 0.0
            for p, enc, tgt, gsz in rs:
                idx = sorted(enc)
                pairs = [[p, enc[i]] for i in idx]
                e = tok(pairs, padding=True, truncation="only_first", max_length=1024,
                        return_tensors="pt").to("cuda")
                s = model(**e).logits.squeeze(-1).float().tolist()
                if not isinstance(s, list):
                    s = [s]
                best = idx[max(range(len(idx)), key=lambda j: s[j])]
                ok += int(best == tgt)
                ce += 1.0 / gsz
                nc += len(idx)
                rnd += gsz / len(idx)
            nn = len(rs)
            out.append((100.0 * (ok / nn) / max(1e-9, ce / nn), k, nn, 100.0 * ok / nn,
                        100.0 * ce / nn, nc / nn, 100.0 * rnd / nn))

    out.sort()
    print("\n  %-10s %6s %8s %9s %10s %7s %8s"
          % ("kind", "n", "top1", "ceiling", "of ceiling", "cands", "chance"))
    for frac, k, nn, t1, cl, cd, ch in out:
        print("  %-10s %6d %7.1f%% %8.1f%% %9.1f%% %7.2f %7.1f%%" % (k, nn, t1, cl, frac, cd, ch))
    print("\n  RANKED WORST-FIRST by 'of ceiling'. attach is the known case at ~44%;")
    print("  anything else near it is a candidate for the same kind of diagnosis.")


if __name__ == "__main__":
    main()
