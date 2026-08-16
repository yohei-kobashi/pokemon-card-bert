"""Where do the Phantom Dives go?

The human model for THIS deck (the Dusknoir bomb variant, not the general list) is explicit:
turn 2 evolve to Drakloak as the top priority, Dragapult ex on turn 3-4, then "three Phantom
Dives and one Cursed Bomb usually ends the game". Measured against it, our evolution is nearly on
schedule -- Dragapult ex arrives on our turn 3.8 -- and the offence is not: 0.37 Phantom Dives per
game against ogerpon_mono, 1.30 against marnie, where a human expects 3.

So the question is not "why is the first one late" alone; it is where each game stops on the
chain. Attacking with Phantom Dive needs FOUR things to be true at once, and the chain localises
which one fails:

    Dragapult ex exists  ->  a line body can pay {R}{P}  ->  that body is ACTIVE
                         ->  and we choose the attack

Everything below is counted per GAME (did it ever happen, and on which of our turns), because a
rate per menu answers a question nobody asked -- an error this project has now made three times.

Deliberately broad, because the cause is not established: the alternatives we take instead, the
turns games last, whether the attacker gets knocked out, whether energy is being stripped, and
whether the accelerators we already own (Crispin x3, Night Stretcher x2) are being used and where
their energy lands. No deck changes are involved; every lever here is one we already hold.
"""
import os

p = "/root/ptcg/repo_sb/tools/dusk_ogerpon_audit.py"
s = open(p).read()

old = "    first_ready = {}                # game -> first turn a body could PAY Phantom Dive\n"
new = ("    first_ready = {}                # game -> first turn a body could PAY Phantom Dive\n"
       "    first_active_ready = {}         # game -> first turn a PAYABLE Dragapult ex was ACTIVE\n"
       "    pd_uses = collections.Counter()      # game -> how many Phantom Dives\n"
       "    atk_uses = collections.Counter()     # game -> how many attacks of any kind\n"
       "    last_turn = {}                  # game -> our last turn number\n"
       "    pult_seen = collections.Counter()    # game -> most Dragapult ex ever in play at once\n"
       "    pult_lost = collections.Counter()    # game -> times that count went DOWN (a KO)\n"
       "    e_drop = collections.Counter()       # game -> turns our line energy fell\n"
       "    benched_ready = collections.Counter()  # turns a payable pult sat on the BENCH\n"
       "    benched_promoted = collections.Counter()  # ... and we took a promote/retreat option\n"
       "    accel = collections.Counter()   # Crispin / Night Stretcher opportunity and use\n"
       "    alt_atk = collections.Counter()      # what we attacked with when PD was also legal\n"
       "    _prev = {}                      # game -> (pult count, line energy) last seen\n")
assert s.count(old) == 1, "state anchor"
s = s.replace(old, new)

old2 = """                if any(isinstance(o, dict) and o.get("attackId") == PHANTOM_DIVE
                       for i, o in enumerate(opts) if i in picked):
                    first_pd.setdefault(cur[0], turn)"""
new2 = """                g = cur[0]
                last_turn[g] = turn
                _line = [x for x in ma + mb if isinstance(x, dict) and x.get("id") in (DREEPY, DRAKLOAK, PULT)]
                _np = sum(1 for x in _line if x.get("id") == PULT)
                _ne = sum(_energy(x) for x in _line)
                pult_seen[g] = max(pult_seen[g], _np)
                pp, pe = _prev.get(g, (0, 0))
                if _np < pp:
                    pult_lost[g] += 1
                if _ne < pe:
                    e_drop[g] += 1
                _prev[g] = (_np, _ne)

                _act = ma[0] if ma else None
                if isinstance(_act, dict) and _act.get("id") == PULT and _plan._can_pd(_act):
                    first_active_ready.setdefault(g, turn)
                # a payable Dragapult ex sitting on the BENCH is the promotion question:
                # Phantom Dive cannot be used from there, and retreat_energy forbids retreating
                # a body that carries {R}/{P}
                if not (isinstance(_act, dict) and _act.get("id") == PULT):
                    if any(x.get("id") == PULT and _plan._can_pd(x) for x in mb if isinstance(x, dict)):
                        benched_ready[key] = 1
                        if picked & {i for i, t in enumerate(texts)
                                     if t == "retreat" or t.startswith("card:")}:
                            benched_promoted[key] = 1

                _pd_legal = [i for i, o in enumerate(opts)
                             if isinstance(o, dict) and o.get("attackId") == PHANTOM_DIVE]
                for i in picked:
                    if isinstance(i, int) and 0 <= i < len(opts) and isinstance(opts[i], dict):
                        _aid = opts[i].get("attackId")
                        if _aid:
                            atk_uses[g] += 1
                            if _aid == PHANTOM_DIVE:
                                pd_uses[g] += 1
                            elif _pd_legal:
                                alt_atk["a%s" % _aid] += 1
                for _cid, _nm in ((1198, "crispin"), (1097, "stretcher")):
                    _o = [i for i, t in enumerate(texts) if ("c%d" % _cid) in t]
                    if _o:
                        accel["%s_turns" % _nm] += 1
                        if picked & set(_o):
                            accel["%s_used" % _nm] += 1
                if any(isinstance(o, dict) and o.get("attackId") == PHANTOM_DIVE
                       for i, o in enumerate(opts) if i in picked):
                    first_pd.setdefault(cur[0], turn)"""
assert s.count(old2) == 1, "pd anchor"
s = s.replace(old2, new2)

old3 = '''    print("\\n-- when could we have attacked, and when did we? (our-turn ordinals) --")'''
new3 = '''    print("\\n=== PHANTOM DIVE FORENSICS (human model: 3 uses per game) ===")
    NG = max(1, a.games)
    def _ord_of(d):
        v = [rank_of.get((g, t)) for g, t in d.items() if rank_of.get((g, t))]
        return (sum(v) / len(v) - 1) if v else 0.0
    print("  the chain, per GAME -- each step needs the one above it")
    print("  %-42s %6s %8s" % ("", "games", "our turn"))
    for lbl, d in (("Dragapult ex ever in play", first_pult),
                   ("... a line body could ever PAY {R}{P}", first_ready),
                   ("... a payable Dragapult ex was ever ACTIVE", first_active_ready),
                   ("... Phantom Dive was actually used", first_pd)):
        print("  %-42s %5d (%3.0f%%) %7.2f" % (lbl, len(d), 100.0 * len(d) / NG, _ord_of(d)))
    tot_pd = sum(pd_uses.values()); tot_atk = sum(atk_uses.values())
    print("  Phantom Dives per game  %.2f   (human: 3)   |  all attacks per game %.2f"
          % (tot_pd / NG, tot_atk / NG))
    import collections as _c
    hist = _c.Counter(pd_uses.get(g, 0) for g in range(a.games))
    print("  games by Phantom Dive count: %s"
          % " ".join("%d:%d" % (k, hist[k]) for k in sorted(hist)))
    print("  our turns per game (mean) %.1f   |  games where our Dragapult ex was KO'd %d"
          % (sum(last_turn.values()) / max(1, len(last_turn)) / 2.0, sum(1 for v in pult_lost.values() if v)))
    print("  turns our line energy FELL: %.2f per game (hammers, retreat costs, knockouts)"
          % (sum(e_drop.values()) / NG))
    print("  a payable Dragapult ex sat on the BENCH on %d turns; we took a promote/retreat "
          "option on %d of them" % (len(benched_ready), len(benched_promoted)))
    print("  attacked with something else while Phantom Dive was legal: %s"
          % (dict(alt_atk.most_common(6)) or "never"))
    print("  accelerators we already own, per TURN they were offered:")
    for nm in ("crispin", "stretcher"):
        t_, u_ = accel["%s_turns" % nm], accel["%s_used" % nm]
        print("     %-10s offered %4d turns, used %4d (%3.0f%%)  = %.2f uses per game"
              % (nm, t_, u_, 100.0 * u_ / max(1, t_), u_ / NG))

    print("\\n-- when could we have attacked, and when did we? (our-turn ordinals) --")'''
assert s.count(old3) == 1, "report anchor"
s = s.replace(old3, new3)

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
print("patched: Phantom Dive forensics")
import subprocess
print(subprocess.run(["python3", "-c", "import ast;ast.parse(open(%r).read());print('parses OK')" % p],
                     capture_output=True, text=True).stdout.strip())
