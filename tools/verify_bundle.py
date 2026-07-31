"""Verify a reranker submission bundle before it is sent to Kaggle.

Checks, in the order they can silently fail:

1. The tarball extracts and `main.py` imports from a CLEAN directory (Kaggle unpacks into
   /kaggle_simulations/agent, not into the repo, so a bundle that only works next to the repo
   passes nothing).
2. **The LM is actually the decision-maker.** main.py wraps scorer construction in
   `except Exception: make_lm_agent(..., model=None)`, so a broken ONNX file degrades to pure
   engine_v2 with no error. Under the LM-only directive that would mean shipping the heuristic
   without knowing it, so this asserts the scorer object exists and that scoring is actually
   called during play.
3. The rendered prompt carries `ID ME` and `DECK[` -- the segments a previous bundle dropped
   silently because deck ids/name were not threaded through.
4. Real decisions: play games against engine_v2 and confirm every returned pick is a legal
   index into the offered options.

Run:  python verify_bundle.py <bundle.tar.gz> <deck> [games]
"""
import json
import os
import subprocess
import sys
import tempfile

TAR = sys.argv[1]
DECK = sys.argv[2]
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 2
REPO = "/root/ptcg/repo"


def main():
    work = tempfile.mkdtemp(prefix="verify_")
    subprocess.run(["tar", "xzf", TAR, "-C", work], check=True)
    inner = work
    entries = os.listdir(work)
    if len(entries) == 1 and os.path.isdir(os.path.join(work, entries[0])):
        inner = os.path.join(work, entries[0])
    print("extracted -> %s" % inner)
    assert os.path.exists(os.path.join(inner, "main.py")), "no main.py"

    # import main.py from the extracted dir ONLY
    sys.path.insert(0, inner)
    os.chdir(inner)
    import importlib.util
    spec = importlib.util.spec_from_file_location("subm_main", os.path.join(inner, "main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    print("main.py imported | DECK_NAME=%r TIER=%r PROMPT_FMT=%r"
          % (m.DECK_NAME, getattr(m, "TIER", "?"), m.PROMPT_FMT))
    assert m.DECK_NAME == DECK, "bundle is for %r, expected %r" % (m.DECK_NAME, DECK)

    # THE call Kaggle makes FIRST, reproduced from the failed episode's own observation.
    # The local game harness starts with battle_start(deck_me, deck_op) and therefore NEVER
    # issues it -- which is exactly why three LM submissions ERRORed while this script passed.
    init_obs = {"current": None, "logs": [], "remainingOverageTime": 600.0,
                "search_begin_input": None, "select": None, "step": 1}
    try:
        d0 = m.agent(init_obs)
    except Exception as e:
        print("FAIL: deck-selection call raised %r" % (e,))
        raise SystemExit(1)
    if not (isinstance(d0, list) and len(d0) == 60 and all(isinstance(x, int) for x in d0)):
        print("FAIL: deck-selection returned %r (want a list of 60 ints)" % (str(d0)[:120],))
        raise SystemExit(1)
    print("deck-selection call OK: 60 card ids, first 5 %s" % (d0[:5],))

    # TIER is main.py's explicit contract; `_scorer` only exists on the top tier, so checking
    # TIER names the actual failure instead of reporting a missing attribute.
    tier = getattr(m, "TIER", None)
    scorer = getattr(m, "_scorer", None)
    if tier != "reranker" or scorer is None:
        print("FAIL: tier=%r scorer=%r -- this bundle would play as engine_v2 and its live "
              "rating would be recorded as the LM's" % (tier, type(scorer).__name__))
        raise SystemExit(1)
    print("scorer live: %s (tier %s)" % (type(scorer).__name__, tier))

    # count real scoring calls, and capture one prompt
    calls = {"n": 0}
    seen = {}
    orig = scorer.score

    def wrapped(prompt, cands, obs=None):
        calls["n"] += 1
        seen.setdefault("prompt", prompt)
        return orig(prompt, cands, obs)
    scorer.score = wrapped

    # play against engine_v2 using the REPO's engine (the bundle ships its own lm/ + agents/)
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "cg-lib"))
    import library
    from cg.game import battle_start, battle_select, battle_finish
    from lm.agent import make_lm_agent
    tuning = json.load(open(os.path.join(REPO, "agents", "tuning.json")))

    d_me = library.read_deck(DECK)
    d_op = library.read_deck("alakazam" if DECK != "alakazam" else "crustle")
    opp = make_lm_agent("alakazam" if DECK != "alakazam" else "crustle",
                        tuning.get("alakazam" if DECK != "alakazam" else "crustle"), None)
    bad = 0
    done = 0
    for g in range(GAMES):
        obs, _ = battle_start(d_me, d_op)
        if obs is None:
            continue
        try:
            for _ in range(4000):
                cur = obs.get("current")
                if cur is None or cur.get("result", -1) != -1:
                    done += 1
                    break
                sel = obs.get("select")
                if sel is None:
                    break
                n = len(sel.get("option") or [])
                if cur["yourIndex"] == 0:
                    pick = m.agent(obs)
                    if (not isinstance(pick, list) or not pick
                            or any((not isinstance(x, int)) or x < 0 or x >= n for x in pick)):
                        bad += 1
                        print("ILLEGAL pick %r for %d options" % (pick, n))
                        break
                else:
                    pick = opp(obs)
                obs = battle_select(pick)
        finally:
            try:
                battle_finish()
            except Exception:
                pass

    print("games completed %d/%d | illegal picks %d | scorer.score calls %d"
          % (done, GAMES, bad, calls["n"]))
    # Which segments to expect is DERIVED from the format, not hard-coded. `ID ME` used to be
    # required here; under identify="op" it is deliberately absent, so the old check would have
    # failed a correct v39 bundle -- and "fixing" that by deleting the check would give up the
    # very guard that caught the bundle which dropped the ID segment entirely.
    fmt = m.PROMPT_FMT
    p = seen.get("prompt", "")
    want, forbid = ["DECK["], []
    idf = fmt.get("identify", "both")
    # lm/identify.render emits ONE `ID` segment holding both halves: `ID ME <deck> <arch> OP
    # <deck>:<p> ...`. So the literal "ID OP" appears only when identify="op" drops the ME half.
    # Checking for "ID OP" unconditionally failed a correct bundle -- read render() before
    # trusting a substring. The OP half is unconditional (it prints "OP ?" when the posterior is
    # empty), so one captured prompt is enough; it is not a turn-dependent segment.
    _i = p.find("ID ")
    idseg = p[_i:_i + 90] if _i >= 0 else ""
    (want if idf in ("both", "me") else forbid).append("ID ME")
    if fmt.get("deck_mode") == "roles":
        want.append("win[")           # role-grouped DECK[]; a collapsed list has only oth[
    if fmt.get("board_facts"):
        want.append("rt:")            # retreat cost -- one of the v39 board facts
    miss = [s for s in want if s not in p]
    if idf != "me" and "OP" not in idseg:
        miss.append("<OP half of the ID segment>")
    extra = [s for s in forbid if s in p]
    print("prompt len=%d | ID segment %r" % (len(p), idseg[:70]))
    print("  expected %r | MISSING %r | SHOULD-NOT-BE-THERE %r" % (want, miss, extra))
    print("  head: %s" % p[:150].replace("\n", " / "))
    ok = (calls["n"] > 0) and bad == 0 and not miss and not extra and done > 0
    print("VERDICT: %s" % ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
