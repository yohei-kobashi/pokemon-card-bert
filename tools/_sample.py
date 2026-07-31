"""Sample-size credibility for tools that produce a VERDICT.

Every wrong call in this project's history traces to one thing: a number that was too
small to mean anything, quoted as if it did. The floor is not a guess — it is measured:

  * **160 games/pair still carries ~±8pt.** Re-running the *identical* (reverted) code
    moved mega_lucario's matchups by up to 8pt (archaludon 89->81, crustle_stall 28->23).
  * **40 games/pair is worthless**: the same mirror read 60-40 one way, then 41-59 the
    other.
  * **Mechanism counts are not immune.** At 30 games/matchup an A/B reported "-14%
    attacks" and "crustle 13->5 wins" — all noise; the *unchanged* baseline's own bench
    count moved 6.13 vs 5.47 between runs. At 60 games, identical code moved
    attacks/game by 1.33 and wins by 14/60.
  * At 40 games/opponent, tools/fingerprint.py flags **26.7%** of decks it provably
    cannot have affected (see its `--compare` control report).

So: below the floor a run is a SMOKE TEST — legitimate for "does it crash / 60 decks
complete", never evidence for "X is better than Y". This module refuses to let that
distinction stay implicit: it prints a banner AND stamps the saved artifact, so a JSON
that was never a measurement cannot later be mistaken for one.

Deliberately does NOT hard-block: `--games 2` fleet smokes are a real, useful thing.
"""

SAMPLE_FLOOR = 150          # games per pair / per matchup / per panel opponent


def is_trustworthy(games):
    return games >= SAMPLE_FLOOR


def banner(games, unit="games/pair"):
    """A loud line for stdout when a run cannot support a verdict (else '')."""
    if is_trustworthy(games):
        return ""
    return (f"\n{'!' * 78}\n"
            f"!! SMOKE TEST, NOT A MEASUREMENT: {games} {unit} is below the {SAMPLE_FLOOR} floor.\n"
            f"!! Identical code moves these numbers by ~8pt at 160 {unit}; at {games} the\n"
            f"!! output cannot distinguish a real effect from noise. Use it for 'does it\n"
            f"!! run', never for 'is it better'. Re-run with --games {SAMPLE_FLOOR}+ to judge.\n"
            f"{'!' * 78}\n")


def stamp(blob, games, unit="games/pair"):
    """Record credibility INSIDE the saved artifact, so it travels with the data."""
    blob["sample_floor"] = SAMPLE_FLOOR
    blob["trustworthy"] = bool(is_trustworthy(games))
    if not blob["trustworthy"]:
        blob["warning"] = (f"SMOKE TEST ONLY: {games} {unit} < {SAMPLE_FLOOR} floor; "
                           f"differences here are indistinguishable from noise.")
    return blob
