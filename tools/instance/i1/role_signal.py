"""Does a card's ROLE predict which option engine_v2 picks? Measure before changing the format.

A prompt change costs a full re-SFT plus a redone deployment vocab sweep, so the feature has to
earn it first. This reads the SFT records built from the current engine (3,088,497 rows, index
targets) and asks, per decision:

  * how often the candidates even differ in role -- if they are all the same role, the feature
    cannot discriminate and the decision is excluded;
  * mutual information between "this option was chosen" and its role;
  * the accuracy of a ROLE-ONLY predictor: knowing nothing but each candidate's role, pick the
    role with the best empirical selection rate. That number is directly comparable to the
    reranker's 69.7% top1 and to chance.

Card ids are parsed out of the rendered menu itself (`0=play:c1086 1=attach:c18@ACTIVE0 ...`),
and the deck comes from the prompt's own `ID ME d_<deck>` segment, so nothing has to be joined
back to the game logs.

Roles come from lm/roles.resolve, i.e. `prompt_roles` if a deck defines it and `card_roles`
otherwise.

Run:  python role_signal.py [n_records]
"""
import collections
import gzip
import json
import math
import os
import re
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

from lm.roles import resolve, UNLABELLED               # noqa: E402

DATA = "data/sft/teacher_0730.jsonl.gz"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
CARD_ONLY = os.environ.get("CARD_ONLY", "0") == "1"
CARD_ROLES = {"win", "engine", "line", "fuel", "tech", "filler", UNLABELLED}

RE_MENU = re.compile(r"(?:^| )(\d+)=(\S+)")
RE_DECK = re.compile(r"ID ME (d_\S+)")
RE_CARD = re.compile(r"c(\d+)")


def opt_role(enc, roles):
    """Role of the card an option acts on, or a kind label when it names no card."""
    m = RE_CARD.search(enc)
    if not m:
        return enc.split(":")[0].split("@")[0]      # attack / retreat / end / num / yes ...
    return roles.get(int(m.group(1)), UNLABELLED)


def main():
    prof = json.load(open("agents/tuning.json"))
    roles_by_deck = {d: resolve(p) for d, p in prof.items()}

    n = used = 0
    chosen = collections.Counter()
    offered = collections.Counter()
    # role-only predictor is fitted and scored on DISJOINT halves
    fit_ch, fit_of = collections.Counter(), collections.Counter()
    rows = []
    for line in gzip.open(DATA, "rt"):
        d = json.loads(line)
        n += 1
        if n > N:
            break
        p = d["prompt"]
        md = RE_DECK.search(p)
        if not md:
            continue
        deck = md.group(1)[2:]
        roles = roles_by_deck.get(deck)
        if not roles:
            continue
        menu = p.rsplit(":: ", 1)[-1]
        ents = RE_MENU.findall(menu)
        if len(ents) < 2:
            continue
        try:
            tgt = int(d["target"])
        except (TypeError, ValueError):
            continue
        rs = {int(i): opt_role(e, roles) for i, e in ents}
        if tgt not in rs:
            continue
        if len(set(rs.values())) < 2:               # role cannot discriminate here
            continue
        if CARD_ONLY:
            # Keep only options that NAME A CARD. attack/retreat/end/yes/no/stop are option
            # KINDS, already spelled out in the menu text, so counting them measures information
            # the model has regardless of roles. This asks what the role adds ON TOP.
            rs = {i: v for i, v in rs.items() if v in CARD_ROLES}
            if len(rs) < 2 or tgt not in rs or len(set(rs.values())) < 2:
                continue
        used += 1
        rows.append((rs, tgt))
        if used % 2:                                 # odd -> fit half
            fit_ch[rs[tgt]] += 1
            for v in rs.values():
                fit_of[v] += 1
        else:
            chosen[rs[tgt]] += 1
            for v in rs.values():
                offered[v] += 1

    print("records read %d | usable (>=2 roles among candidates) %d (%.1f%%)"
          % (n - 1, used, 100.0 * used / max(1, n - 1)))
    if used < 1000:
        print("too few to measure")
        return

    print("\n  %-12s %8s %8s %8s %7s" % ("role", "offered", "chosen", "rate", "lift"))
    base = sum(chosen.values()) / max(1, sum(offered.values()))
    for r, o in offered.most_common(14):
        c = chosen.get(r, 0)
        rate = c / max(1, o)
        print("  %-12s %8d %8d %7.1f%% %6.2fx" % (r, o, c, 100.0 * rate, rate / max(1e-9, base)))
    print("  %-12s %8d %8d %7.1f%%" % ("ALL", sum(offered.values()), sum(chosen.values()),
                                       100.0 * base))

    # mutual information I(chosen ; role) in bits, over the offered-option population
    tot = sum(offered.values())
    mi = 0.0
    for r, o in offered.items():
        c = chosen.get(r, 0)
        for k, cnt in ((1, c), (0, o - c)):
            if cnt <= 0:
                continue
            pxy = cnt / tot
            px = o / tot
            py = (sum(chosen.values()) if k else tot - sum(chosen.values())) / tot
            mi += pxy * math.log2(pxy / (px * py))
    hy = 0.0
    for k in (sum(chosen.values()), tot - sum(chosen.values())):
        if k > 0:
            q = k / tot
            hy -= q * math.log2(q)
    print("\n  I(chosen ; role) = %.4f bits   H(chosen) = %.4f bits   -> %.1f%% of the label"
          % (mi, hy, 100.0 * mi / max(1e-9, hy)))

    # ROLE-ONLY predictor, fitted on the odd half, scored on the even half
    score = {r: fit_ch.get(r, 0) / max(1, fit_of.get(r, 0)) for r in fit_of}
    ok = tot_ev = 0
    rnd = 0.0
    for i, (rs, tgt) in enumerate(rows):
        if i % 2 == 0:
            continue
        best = max(rs, key=lambda k: (score.get(rs[k], 0.0), -k))
        ok += int(best == tgt)
        tot_ev += 1
        rnd += 1.0 / len(rs)
    print("  ROLE-ONLY predictor: %d/%d = %.1f%%   (random over the same menus %.1f%%)"
          % (ok, tot_ev, 100.0 * ok / max(1, tot_ev), 100.0 * rnd / max(1, tot_ev)))
    print("  reference: the reranker's held-out top1 is 69.7%% on all decisions")


if __name__ == "__main__":
    main()
