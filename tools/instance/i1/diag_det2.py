"""Is the determinization deficit exactly the number of FACE-DOWN board cards?

A face-down Pokemon is a real card off that player's decklist, but the observation gives it
no id, so subtraction-by-id cannot see it. If deficit == face-down count on every decision,
the accounting rule is settled and the fix is mechanical.
"""
import collections
import os
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import library                                                   # noqa: E402
from cg.game import battle_start, battle_select, battle_finish    # noqa: E402
from lm.agent import make_lm_agent                                # noqa: E402
import rl_branch                                                  # noqa: E402


def facedown_count(pl):
    """Board entries that exist but expose no card id."""
    n = 0
    for z in ("active", "bench"):
        for x in (pl.get(z) or []):
            if x is None:
                n += 1
            elif rl_branch._cid(x) is None:
                n += 1
    return n


def main():
    pairs = [("alakazam", "dragapult"), ("crustle_stall", "alakazam")]
    agree = collections.Counter()
    detail = collections.Counter()
    for gi, (pilot, opp_name) in enumerate(pairs):
        d_me = library.read_deck(pilot)
        d_op = library.read_deck(opp_name)
        a_me = make_lm_agent(pilot, None, None)
        a_op = make_lm_agent(opp_name, None, None)
        obs, _ = battle_start(d_me, d_op)
        try:
            for _ in range(4000):
                cur = obs.get("current")
                if cur is None or cur.get("result", -1) != -1:
                    break
                if obs.get("select") is None:
                    break
                yi = cur["yourIndex"]
                me, opp = cur["players"][yi], cur["players"][1 - yi]

                used_me = (rl_branch._board_ids(me)
                           + [rl_branch._cid(h) for h in (me.get("hand") or [])])
                used_me = [i for i in used_me if i is not None]
                used_opp = [i for i in rl_branch._board_ids(opp) if i is not None]

                have_me = 60 - len(used_me)
                want_me = int(me.get("deckCount", 0)) + len(me.get("prize") or [])
                have_opp = 60 - len(used_opp)
                want_opp = (int(opp.get("deckCount", 0)) + int(opp.get("handCount", 0))
                            + len(opp.get("prize") or []))

                d_me_def = have_me - want_me
                d_opp_def = have_opp - want_opp
                fd_me = facedown_count(me)
                fd_opp = facedown_count(opp)

                agree["me_match" if d_me_def == fd_me else "me_MISMATCH"] += 1
                agree["opp_match" if d_opp_def == fd_opp else "opp_MISMATCH"] += 1
                if d_me_def != fd_me:
                    detail["me deficit=%d facedown=%d" % (d_me_def, fd_me)] += 1
                if d_opp_def != fd_opp:
                    detail["opp deficit=%d facedown=%d" % (d_opp_def, fd_opp)] += 1

                obs = battle_select((a_me if yi == 0 else a_op)(obs))
        finally:
            battle_finish()

    print("=== deficit == face-down count? ===")
    for k, v in sorted(agree.items()):
        print("  %-14s %d" % (k, v))
    tot = agree["me_match"] + agree["me_MISMATCH"]
    print("  agreement: me %.1f%%, opp %.1f%%"
          % (100.0 * agree["me_match"] / max(1, tot),
             100.0 * agree["opp_match"] / max(1, tot)))
    if detail:
        print("\n  mismatches:")
        for k, v in detail.most_common(12):
            print("    %4d  %s" % (v, k))


if __name__ == "__main__":
    main()
