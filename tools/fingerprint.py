"""Per-deck MECHANISM fingerprint — the ledger that protects deck diversity.

Why this exists: engine unification (legacy -> engine_v2) and the removal of bespoke
L2 classes both risk HOMOGENISING the decks — every deck still "works", every win rate
still lands inside the ±8pt self-play noise, and nobody notices that a deck stopped
playing its own game. A win rate cannot catch that. A mechanism count can.

So: snapshot what each deck actually DOES, migrate, and require the migrated deck to
reproduce its own fingerprint. A deck whose fingerprint cannot be reproduced KEEPS its
bespoke class — the refactor serves the decks, not the other way round.

Measured through whatever engine the deck SHIPS with: since tools/generate_agents.py
grew a V2_TEMPLATE, load_agent() builds the same wrapper the submission bundles, so
this measures production by construction.

Decisions are read from the agent's own choices on the main menu (select type/ctx
(0,0)) — never from obs['logs'], which is EMPTY in the live loop:
    optType 7  {'index'}                  -> play hand[index]
    optType 8  {'area':2,'index',...}     -> attach hand[index] (energy / tool)
    optType 9  {'area':2,'index',...}     -> evolve using hand[index]
    optType 13 {'attackId'}               -> attack
    optType 12 -> retreat, 14 -> end turn

Usage:
    python tools/fingerprint.py --games 60 --out fingerprints/baseline.json
    python tools/fingerprint.py --decks slowking --games 60 --out /tmp/after.json
    python tools/fingerprint.py --compare fingerprints/baseline.json /tmp/after.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import _sample        # noqa: E402
import arena          # noqa: E402
import library        # noqa: E402
from battle_log import load_agent   # noqa: E402

# A TRULY FROZEN panel: the opponents are self-contained SUBMISSION BUNDLES under
# panel_frozen/, not live decks. A fingerprint only means something against the same
# opponents, so the panel must never move -- and "pick legacy decks, they're immune to
# engine_v2 edits" only held until we migrated the legacy decks themselves. A bundle's
# main.py inlines the whole engine (0 repo imports), so NOTHING in this repo can change
# it, ever. That also preserves the free control group: `--compare` needs a cohort the
# change provably cannot touch, and once every deck is on engine_v2 the panel bundles are
# the only such cohort left.
# Archetype spread, frozen 2026-07-17 from their last legacy build: bulk/race, spread+
# snipe, damage-prevention wall, item-lock control.
PANEL_DIR = os.path.join(ROOT, "panel_frozen")
PANEL = ("archaludon", "dragapult", "crustle_stall", "trevenant_control")
_PANEL_CACHE = {}


def _panel_agent(name):
    """Import a frozen bundle's main.py with ITS OWN deck.csv bound.

    cwd must be inside the bundle at import time: load_deck() tries `decks/<name>.csv`
    FIRST, so importing from the repo root silently binds TODAY's list and you measure a
    frozen engine piloting a live deck.
    """
    if name in _PANEL_CACHE:
        return _PANEL_CACHE[name]
    import importlib.util
    path = os.path.join(PANEL_DIR, name)
    cwd = os.getcwd()
    try:
        os.chdir(path)
        spec = importlib.util.spec_from_file_location(f"_panel_{name}",
                                                      os.path.join(path, "main.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        os.chdir(cwd)
    own = [int(x) for x in open(os.path.join(path, "deck.csv")) if x.strip()]
    assert sorted(m.DECK) == sorted(own), f"panel {name} bound the wrong deck (cwd trap)"
    _PANEL_CACHE[name] = (m.agent, own)
    return _PANEL_CACHE[name]


def _fingerprint(args):
    name, games = args
    try:
        agent, deck = load_agent(name), library.read_deck(name)
    except Exception as e:                        # noqa: BLE001
        return name, {"error": f"{type(e).__name__}: {e}"}
    atk = Counter()
    play = Counter()
    n = wins = draws = 0
    first_turns = []
    attacked = 0
    prizes_taken = []
    turns = []
    for opp in PANEL:
        try:
            oa, od = _panel_agent(opp)            # frozen bundle, never the live deck
        except Exception:                         # noqa: BLE001
            continue
        for g in range(games):
            state = {"first": None, "atk": 0, "turn": 0}

            def wrap(obs):
                c = obs["current"]
                me = c["yourIndex"]
                ps = c["players"][me]
                hand = [h.get("id") if isinstance(h, dict) else h
                        for h in (ps.get("hand") or [])]
                state["turn"] = max(state["turn"], c.get("turn", 0))
                sel = agent(obs)
                s = obs["select"]
                if (s.get("type"), s.get("context")) == (0, 0):
                    opt = s["option"]
                    for i in (sel or []):
                        if not (0 <= i < len(opt)):
                            continue
                        o = opt[i]
                        t, idx = o.get("type"), o.get("index")
                        if t == 13:
                            atk[o.get("attackId")] += 1
                            state["atk"] += 1
                            if state["first"] is None:
                                state["first"] = c.get("turn", 0)
                        elif t in (7, 8, 9) and idx is not None and idx < len(hand):
                            play[hand[idx]] += 1
                opz = c["players"][1 - me].get("prize")
                state["opz"] = len(opz) if isinstance(opz, list) else opz
                return sel

            w = arena.play(wrap, oa, deck, od) if g % 2 == 0 else arena.play(oa, wrap, od, deck)
            mine = 0 if g % 2 == 0 else 1
            if w == mine:
                wins += 1
            elif w not in (0, 1):
                draws += 1        # engine result 2 / None: a timed-out non-game
            n += 1
            turns.append(state["turn"])
            if state["first"] is not None:
                first_turns.append(state["first"])
                attacked += 1
            if state.get("opz") is not None:
                prizes_taken.append(6 - state["opz"])
    if not n:
        return name, {"error": "no games"}
    tot_atk = sum(atk.values())
    try:
        from agents import engine_v2 as _e2
        _nm = lambda i: getattr(_e2._ATTACKS.get(i), "name", None) or str(i)   # noqa: E731
    except Exception:                             # noqa: BLE001
        _nm = str
    return name, {
        "games": n,
        "winrate": round(100 * wins / n, 1),
        # a deck that mostly DRAWS is not playing a game at all (slowking vs a wall
        # times out at 400+ turns); win rate alone hides this completely.
        "draw_pct": round(100 * draws / n, 1),
        "attacks_per_game": round(tot_atk / n, 3),
        # THE identity signal: which attacks this deck actually uses, as shares.
        "attack_mix": {f"{k}:{_nm(k)}": round(v / tot_atk, 3)
                       for k, v in atk.most_common(8)} if tot_atk else {},
        "first_attack_turn": round(sum(first_turns) / len(first_turns), 2) if first_turns else None,
        "attacked_pct": round(100 * attacked / n, 1),
        "plays_per_game": {str(k): round(v / n, 3) for k, v in play.most_common(12)},
        "prizes_taken": round(sum(prizes_taken) / len(prizes_taken), 2) if prizes_taken else None,
        "game_turns": round(sum(turns) / len(turns), 1),
    }


def compare(a_path, b_path, tol_share=0.15, tol_rate=0.35):
    """Flag decks whose IDENTITY moved, not decks whose win rate moved."""
    a = json.load(open(a_path))["decks"]
    b = json.load(open(b_path))["decks"]
    print(f"{'deck':24} {'verdict':10} detail")
    bad = 0
    flags = {}
    for name in sorted(set(a) & set(b)):
        fa, fb = a[name], b[name]
        if "error" in fa or "error" in fb:
            continue
        notes = []
        # 1. does it still use the same attacks, in the same proportions?
        keys = set(fa["attack_mix"]) | set(fb["attack_mix"])
        for k in keys:
            va, vb = fa["attack_mix"].get(k, 0.0), fb["attack_mix"].get(k, 0.0)
            if abs(va - vb) > tol_share:
                notes.append(f"attack {k} share {va:.2f}->{vb:.2f}")
        # 2. does it still attack as often, and start as fast?
        ra, rb = fa["attacks_per_game"], fb["attacks_per_game"]
        if ra and abs(ra - rb) / ra > tol_rate:
            notes.append(f"attacks/game {ra:.2f}->{rb:.2f}")
        ta, tb = fa.get("first_attack_turn"), fb.get("first_attack_turn")
        if ta and tb and abs(ta - tb) > 1.5:
            notes.append(f"first attack turn {ta}->{tb}")
        if abs(fa["attacked_pct"] - fb["attacked_pct"]) > 15:
            notes.append(f"attacked% {fa['attacked_pct']}->{fb['attacked_pct']}")
        verdict = "IDENTITY!" if notes else "ok"
        bad += bool(notes)
        flags[name] = bool(notes)
        print(f"{name:24} {verdict:10} " + ("; ".join(notes) if notes else
              f"wr {fa['winrate']}->{fb['winrate']}  atk/g {ra:.2f}->{rb:.2f}"))
    print(f"\n{bad} deck(s) changed IDENTITY (win-rate moves are NOT flagged: ±8pt is noise)")
    _control_report(flags)
    return bad


def _control_report(flags):
    """The BUILT-IN PLACEBO: decks the change provably cannot touch.

    A raw "N decks changed identity" is meaningless without knowing how many the
    comparison flags when NOTHING changed. For any engine_v2 edit, every `engine:
    "legacy"` deck is a control group by construction -- it runs _engine.py+policies.py
    and cannot be affected. Whatever fraction of THOSE gets flagged is the false-positive
    floor, for free, in the same run. Only a v2 flag rate ABOVE that floor is evidence.

    Measured 2026-07-17 on the typed-each-count change at 40 games/opponent: legacy
    4/15 = **26.7% false positives**, v2 8/45 = 17.8% -- i.e. the v2 rate was BELOW the
    floor, so the change showed no detectable identity damage AND the gate was too noisy
    to be trusted per-deck. Raise --games until the control floor is near zero before
    treating any single deck's flag as real.
    """
    try:
        import json as _j
        tun = _j.load(open(os.path.join(ROOT, "agents", "tuning.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return
    ctl = [d for d in flags if (tun.get(d) or {}).get("engine") == "legacy"]
    exp = [d for d in flags if (tun.get(d) or {}).get("engine") == "v2"]
    if not ctl or not exp:
        return
    cf = sum(flags[d] for d in ctl)
    ef = sum(flags[d] for d in exp)
    cr, er = 100 * cf / len(ctl), 100 * ef / len(exp)
    print(f"\nCONTROL (legacy decks — an engine_v2 edit CANNOT touch them):")
    print(f"   false-positive floor  {cf}/{len(ctl)} = {cr:.1f}%")
    print(f"   engine_v2 flag rate   {ef}/{len(exp)} = {er:.1f}%")
    print("   -> " + ("NO evidence of identity damage: the v2 rate is at/below the "
                      "noise floor. Raise --games to resolve anything finer."
                      if er <= cr else
                      f"v2 flags exceed the floor by {er - cr:.1f}pt — investigate the "
                      f"v2-only decks above."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60, help="games per panel opponent")
    ap.add_argument("--decks", type=str, default="", help="comma-separated subset")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "NEW"))
    a = ap.parse_args()
    if a.compare:
        sys.exit(1 if compare(*a.compare) else 0)

    decks = library.list_decks()
    if a.decks:
        want = set(a.decks.split(","))
        decks = [d for d in decks if d in want]
    print(f"fingerprinting {len(decks)} deck(s) x {len(PANEL)} panel x {a.games} games "
          f"on {a.workers} workers")
    t = time.time()
    out = {}
    with Pool(a.workers) as pool:
        for i, (name, fp) in enumerate(pool.imap_unordered(
                _fingerprint, [(d, a.games) for d in decks]), 1):
            out[name] = fp
            if i % 10 == 0 or i == len(decks):
                print(f"  {i}/{len(decks)} ({time.time() - t:.0f}s)", flush=True)
    print(_sample.banner(a.games, "games/opponent"), end="")
    blob = _sample.stamp({"panel": list(PANEL), "games_per_opponent": a.games,
                          "decks": out}, a.games, "games/opponent")
    path = a.out or os.path.join(ROOT, "fingerprints", "baseline.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(blob, open(path, "w"), indent=1)
    errs = [d for d, f in out.items() if "error" in f]
    print(f"saved -> {path}" + (f"   {len(errs)} errored: {errs}" if errs else ""))


if __name__ == "__main__":
    main()
