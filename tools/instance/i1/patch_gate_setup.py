"""Make the gate report SETUP SPEED next to the win rate.

The human template for dragapult_dusknoir is explicit -- turn 1: three Dreepy (two at minimum)
and a Duskull on the bench; turn 2: evolve to Drakloak and draw with Recon Directive -- and we
open on 1.45 Dreepy and 0.39 Drakloak with the first Phantom Dive on our turn 9, in 13 of 60
games.  A round that moves the win rate without moving those numbers moved something else, and a
round that reaches the template without moving the win rate is still worth knowing about.  So
every round prints both from now on.

Keyed by OUR turn ordinal rather than the engine's counter: the counter is shared between the
seats, so its "turn 1" is our first turn only when we go first.  The first table written this way
had exactly 30 of 60 games at t1 and 46 at t3, which is how the shared counter announces itself.
"""
import os

p = "/root/ptcg/repo_sb/tools/gate_protagonist.py"
s = open(p).read()

old = "def parse_arm(s):"
new = '''SETUP_IDS = ((119, "dreepy"), (120, "drakloak"), (121, "pult"), (131, "duskull"))
PHANTOM_DIVE = 154


def _n_energy(body):
    for k in ("energy", "attachedEnergy", "energies"):
        v = body.get(k)
        if isinstance(v, list):
            return len(v)
    return 0


def setup_watch(agent, acc, pd_acc):
    """Wrap an arm's pilot and record how fast its board comes up.

    `acc` collects one Counter per (game, our-Nth-turn); `pd_acc` collects the ordinal of the
    first Phantom Dive, or nothing when the game never got one.  Costs one dict update per
    decision and cannot change the pick -- the wrapper returns the agent's answer untouched, and
    every read is inside a try so an observation shape it does not expect degrades to no data
    rather than to a failed gate.
    """
    st = {}
    seen_pd = [None]

    def flush():
        for i, t in enumerate(sorted(st), 1):
            acc.setdefault(i, []).append(st[t])
            if seen_pd[0] == t:
                pd_acc.append(i)
        st.clear()
        seen_pd[0] = None

    def w(obs):
        cur = obs.get("current") or {}
        if not cur:
            flush()                      # episode boundary: fold the game that just ended
            return agent(obs)
        pick = agent(obs)
        try:
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            if not opts:
                return pick
            t = cur.get("turn")
            if not isinstance(t, int):
                return pick
            yi = cur.get("yourIndex", 0)
            me = (cur.get("players") or [{}])[yi] or {}
            bodies = [x for x in ([(me.get("active") or [None])[0]] + list(me.get("bench") or []))
                      if isinstance(x, dict)]
            ids = [b.get("id") for b in bodies]
            d = st.setdefault(t, collections.Counter())
            for cid, nm in SETUP_IDS:
                d[nm] = max(d[nm], ids.count(cid))
            d["bodies"] = max(d["bodies"], len(ids))
            d["energy"] = max(d["energy"], sum(_n_energy(b) for b in bodies))
            if seen_pd[0] is None:
                for i in (pick if isinstance(pick, (list, tuple)) else [pick]):
                    if (isinstance(i, int) and 0 <= i < len(opts)
                            and isinstance(opts[i], dict)
                            and opts[i].get("attackId") == PHANTOM_DIVE):
                        seen_pd[0] = t
        except Exception:               # noqa: BLE001 -- measurement must never break the gate
            pass
        return pick

    return w, flush


def parse_arm(s):'''
assert s.count(old) == 1, "helper anchor"
s = s.replace(old, new, 1)

old = """        arms.append((label, spec, fmt, agent))"""
new = """        setup_acc[label], pd_acc[label] = {}, []
        agent, flusher = setup_watch(agent, setup_acc[label], pd_acc[label])
        flushers[label] = flusher
        arms.append((label, spec, fmt, agent))"""
assert s.count(old) == 1, "arm anchor"
s = s.replace(old, new)

old = "    arms = []\n"
new = "    arms = []\n    setup_acc, pd_acc, flushers = {}, {}, {}\n"
assert s.count(old) == 1, "state anchor"
s = s.replace(old, new)

old = '''    print("\\n%-8s %8s %8s %s" % ("arm", "win%", "delta", "vs " + a.baseline))'''
new = '''    for _f in flushers.values():
        _f()                            # the last game of the run has no episode start after it

    print("\\n-- setup speed, by OUR turn (human template: t1 = Dreepy x3 + Duskull x1-2) --")
    print("  %-8s %9s %9s %9s %11s %11s %10s %14s"
          % ("arm", "t1 dreepy", "t1 bodies", "t2 dreepy", "t2 drakloak", "t3 drakloak",
             "t3 energy", "1st PhantomDive"))
    for label, _s, _f, _ag in arms:
        acc = setup_acc[label]
        m = lambda i, k: (sum(r[k] for r in acc.get(i, [])) / len(acc[i])) if acc.get(i) else 0.0
        pds = pd_acc[label]
        ngames = len(acc.get(1, []))
        pd_s = ("t%.1f in %d/%d" % (sum(pds) / len(pds), len(pds), ngames)) if pds \\
            else ("never in %d" % ngames)
        print("  %-8s %9.2f %9.2f %9.2f %11.2f %11.2f %10.2f %14s"
              % (label, m(1, "dreepy"), m(1, "bodies"), m(2, "dreepy"), m(2, "drakloak"),
                 m(3, "drakloak"), m(3, "energy"), pd_s))

    print("\\n%-8s %8s %8s %s" % ("arm", "win%", "delta", "vs " + a.baseline))'''
assert s.count(old) == 1, "report anchor"
s = s.replace(old, new)

old = '''        out["arms"][label] = {"spec": spec, "fmt": fmt, "win_rate": wr,
                              "delta_vs_baseline": d, "se": se, "games": len(allg)}'''
new = '''        _acc, _pds = setup_acc[label], pd_acc[label]
        _m = lambda i, k: (sum(r[k] for r in _acc.get(i, [])) / len(_acc[i])) if _acc.get(i) else 0.0
        out["arms"][label] = {"spec": spec, "fmt": fmt, "win_rate": wr,
                              "delta_vs_baseline": d, "se": se, "games": len(allg),
                              "setup": {
                                  "t1_dreepy": _m(1, "dreepy"), "t1_bodies": _m(1, "bodies"),
                                  "t2_dreepy": _m(2, "dreepy"), "t2_drakloak": _m(2, "drakloak"),
                                  "t3_drakloak": _m(3, "drakloak"), "t3_energy": _m(3, "energy"),
                                  "pd_turn": (sum(_pds) / len(_pds)) if _pds else None,
                                  "pd_games": len(_pds), "games_seen": len(_acc.get(1, []))}}'''
assert s.count(old) == 1, "json anchor"
s = s.replace(old, new)

if "\nimport collections\n" not in s:
    s = s.replace("\nimport json\n", "\nimport collections\nimport json\n", 1)
assert "\nimport collections\n" in s, "import anchor"

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
print("patched gate_protagonist")
