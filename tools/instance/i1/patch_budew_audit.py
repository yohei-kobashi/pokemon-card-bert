"""Add the Budew-gate measurement to tools/dusk_ogerpon_audit.py.

The question: `lock_early` only exists while `turn <= 6 and PULT not in play`.  Last night's A/B
added a second Budew without touching that gate, so it may have measured how often we DRAW Budew
rather than whether the lock is worth playing.  Split the denominator.

Counted per TURN, never per menu -- the same artifact that made phantom_dive read 39% per menu
and 94.7% per turn, and that has now been hit three times.

Nothing here touches dusk_plan.py or plan_filter.py, so rules_fp is unchanged and the round in
flight survives.  The gate condition is not reimplemented either: `opportunities()` is called and
asked whether it produced the rule, so what is measured is the shipped rule, not my reading of it.
"""
import os

p = "/root/ptcg/repo/tools/dusk_ogerpon_audit.py"
s = open(p).read()

# --- constants ------------------------------------------------------------------------------
old = "JUDGE, STAMP, BOSS, FAN = 1213, 1080, 1182, 1161"
new = old + """
ITCHY_POLLEN = 323                 # Budew's lock: no energy cost, no Item cards for them
BUDEW, PULT = 235, 121"""
assert s.count(old) == 1, "const anchor"
s = s.replace(old, new)

# --- measurement, inside watch(), right after the disruption per-turn block ------------------
old2 = """            # Was a zero-energy body sitting on their bench for Boss's Orders to drag up?"""
new2 = """            # --- Budew's lock, and what the plan's gate does to it ---------------------
            # Three numbers, not one: how often the lock is even AVAILABLE (do we draw and
            # promote Budew at all), how often the pilot takes it, and -- of the turns where it
            # was available -- how many `lock_early` refuses to teach because of its two
            # clauses.  If the refusals dominate, the A/B added copies of a card the plan had
            # already decided not to talk about.
            pol_ = [i for i, o in enumerate(opts)
                    if isinstance(o, dict) and o.get("attackId") == ITCHY_POLLEN]
            if pol_:
                T["itchy_turns"].add(key)
                if picked & set(pol_):
                    T["itchy_played"].add(key)
                try:
                    live = "lock_early" in _plan.opportunities(obs)
                except Exception:
                    live = None
                if live:
                    T["itchy_rule_live"].add(key)
                elif live is False:
                    T["itchy_rule_dead"].add(key)
                    # which clause did it -- both can be true at once
                    if isinstance(turn, int) and turn > 6:
                        T["itchy_dead_late"].add(key)
                    if PULT in [(x or {}).get("id") for x in ma + mb]:
                        T["itchy_dead_pult"].add(key)
            # Budew in hand but not Active is a different miss: the lock needs it promoted.
            if any(("c%d" % BUDEW) in t for t in texts):
                T["budew_in_menu"].add(key)

            # Was a zero-energy body sitting on their bench for Boss's Orders to drag up?"""
assert s.count(old2) == 1, "watch anchor"
s = s.replace(old2, new2)

# --- import the plan module the audit is measuring -------------------------------------------
old3 = "    import mirror_match as mm\n"
new3 = "    import mirror_match as mm\n    import dusk_plan as _plan\n"
assert s.count(old3) == 1, "import anchor"
s = s.replace(old3, new3)
s = s.replace("    def watch(obs):", "    def watch(obs, _plan=None):", 0)   # no-op guard

# `_plan` is bound in main()'s scope and watch() closes over it -- but only if it is imported
# before watch is defined, which the anchor above guarantees.

# --- report ----------------------------------------------------------------------------------
old4 = '''    print("  boss with an EMPTY body on their bench to drag: %d turns"'''
new4 = '''    ig = len(T["itchy_turns"])
    print("\\n-- Budew's lock, and the gate `lock_early` puts on it --")
    print("  turns Itchy Pollen was legal          %6d   (%.2f per game)"
          % (ig, ig / max(1, a.games)))
    print("  ... of which the pilot played it      %6d   %5.0f%%"
          % (len(T["itchy_played"]), 100.0 * len(T["itchy_played"]) / max(1, ig)))
    print("  ... `lock_early` LIVE (rule teaches)  %6d   %5.0f%%"
          % (len(T["itchy_rule_live"]), 100.0 * len(T["itchy_rule_live"]) / max(1, ig)))
    print("  ... gated OUT                         %6d   %5.0f%%"
          % (len(T["itchy_rule_dead"]), 100.0 * len(T["itchy_rule_dead"]) / max(1, ig)))
    print("        by turn > 6                     %6d" % len(T["itchy_dead_late"]))
    print("        by Dragapult ex in play         %6d" % len(T["itchy_dead_pult"]))
    print("  turns a Budew option appeared at all  %6d   (%.2f per game)"
          % (len(T["budew_in_menu"]), len(T["budew_in_menu"]) / max(1, a.games)))
    if ig == 0:
        print("  -> the lock is never AVAILABLE: Budew is not reaching the Active slot, so")
        print("     neither the gate nor a second copy can matter. NO TRIGGER.")
    elif len(T["itchy_rule_dead"]) > len(T["itchy_rule_live"]):
        print("  -> the gate refuses most of the turns the lock was available on. Relaxing it")
        print("     is a one-line change and the A/B did not test it.")
    print("  boss with an EMPTY body on their bench to drag: %d turns"'''
assert s.count(old4) == 1, "report anchor"
s = s.replace(old4, new4)

t = p + ".new"
open(t, "w").write(s)
os.replace(t, p)
print("patched")
