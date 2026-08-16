"""Label the pairs the playouts cannot label, using a board fact instead of throwing them away.

WHAT THE MEASUREMENT SAID.  Adding the setup potential to Q as a small additive term did nothing:
against a same-condition control (gamma 0 vs gamma 0), the winner flipped 27.2% of the time from
playout noise alone; with gamma 0.10 it flipped 29.1% and 26.0%.  Indistinguishable.  The reason
is arithmetic -- 24 playouts give each Q an SE of ~0.2, so the pairwise gap carries a ~0.22 noise
floor while the whole potential spans 0.10.  Raising the weight until it wins would also override
Q where Q has real signal, so the additive form is simply the wrong insertion point.

WHERE IT BELONGS INSTEAD.  qmin drops every pair with |qw - ql| < 0.35 -- 965 of 1511 in round 23,
64% of the data -- because those labels are coin flips.  That is precisely where a fact with zero
estimation error is worth the most.  So: Q keeps every pair it can actually decide, and the pairs
it cannot are labelled by which candidate leaves us closer to a payable Phantom Dive.

    |qw - ql| >= qmin        Q decides, exactly as before
    |qw - ql| <  qmin        Phi decides, IF the two differ by a full useful energy
                             (and the row is written with a weak weight, not a confident one)

THE RISK, STATED.  This teaches the same preference that `energy_line`/`energy_focus` encode, and
FORCING those cost -2.25pt.  The mechanism differs -- a soft label on decisions the playouts call
indifferent, against a hard override on every attach menu -- but the below-qmin set is a mixture
of "truly indifferent" and "real difference hidden by noise", and on the second kind this teaches
the same thing that lost.  Hence the weak weight (0.65/0.35, not 1/0) and the gate afterwards.

Default OFF: --phi-min 0 reproduces today's converter exactly.
"""
import os

# ---------------------------------------------------------------- rl_branch: return the potential
p = "/root/ptcg/repo/tools/rl_branch.py"
s = open(p).read()

old = """def branch_values(obs, my_deck, opp_deck, pilot_i, selections,
                  agent_me, agent_opp, n_playouts=1, rng=None):"""
new = """def branch_values(obs, my_deck, opp_deck, pilot_i, selections,
                  agent_me, agent_opp, n_playouts=1, rng=None, with_potential=False):"""
assert s.count(old) == 1, "sig anchor"
s = s.replace(old, new)

old = """    shape = bool(SETUP_GAMMA) and _PULT in (my_deck or ())
    vals = [[] for _ in selections]"""
new = """    shape = bool(SETUP_GAMMA) and _PULT in (my_deck or ())
    # The potential is RECORDED whenever asked for, independently of whether it is added to the
    # reward: the downstream converter uses it to label the pairs Q cannot, and that path wants
    # the number even when SETUP_GAMMA is 0.
    want_phi = bool(with_potential) and _PULT in (my_deck or ())
    vals = [[] for _ in selections]
    phis = [[] for _ in selections]"""
assert s.count(old) == 1, "vals anchor"
s = s.replace(old, new)

old = """                v = _playout(step["state"], pilot_i, agent_me, agent_opp)
                if v is not None:
                    if shape:"""
new = """                if want_phi:
                    phis[i].append(_setup_potential(step["state"], pilot_i))
                v = _playout(step["state"], pilot_i, agent_me, agent_opp)
                if v is not None:
                    if shape:"""
assert s.count(old) == 1, "phi record anchor"
s = s.replace(old, new)

old = """    return [sum(v) / len(v) if v else None for v in vals]"""
new = """    out = [sum(v) / len(v) if v else None for v in vals]
    if with_potential:
        return out, [sum(v) / len(v) if v else None for v in phis]
    return out"""
assert s.count(old) == 1, "return anchor"
s = s.replace(old, new)

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
print("rl_branch: branch_values(..., with_potential=True) -> (values, potentials)")

# ---------------------------------------------------------------- dpo_branch: record it
q = "/root/ptcg/repo/tools/dpo_branch.py"
t = open(q).read()
old = "                            q = rl_branch.branch_values(obs, IDS[yi], IDS[1 - yi], yi, sels,"
i = t.index(old)
j = t.index("\n", t.index(")", i))
call = t[i:j]
new_call = (call.replace("q = rl_branch.branch_values(", "q, phi = rl_branch.branch_values(")
                .rstrip())
assert new_call.endswith(")"), "call shape: %r" % new_call[-40:]
new_call = new_call[:-1] + ", with_potential=True)"
t = t[:i] + new_call + t[j:]

old2 = '''                            "pl": playouts,'''
new2 = '''                            "pl": playouts,
                            # The setup potential of each candidate's SUCCESSOR: how close that
                            # choice leaves our best line body to paying Phantom Dive, in [0, 1].
                            # Recorded on every pair; only the converter decides whether to use
                            # it, and only on the pairs the playouts could not separate.
                            "phi_w": (round(phi[best], 3) if phi[best] is not None else None),
                            "phi_l": (round(phi[1 - best], 3) if phi[1 - best] is not None else None),'''
assert t.count(old2) == 1, "record anchor"
t = t.replace(old2, new2)
open(q + ".new", "w").write(t)
os.replace(q + ".new", q)
print("dpo_branch: pairs now carry phi_w / phi_l")

# ---------------------------------------------------------------- mrl_convert: use it
r = "/root/mrl_convert.py"
u = open(r).read()
old3 = """ap.add_argument("--qmin", type=float, default=0.0,"""
new3 = """ap.add_argument("--phi-min", type=float, default=0.0,
                help="rescue below-qmin pairs whose setup potentials differ by at least this. "
                     "Phi steps in halves (one useful energy = 0.5), so 0.5 means a full "
                     "energy of difference. 0 disables and the converter is unchanged.")
ap.add_argument("--phi-wc", type=float, default=0.65,
                help="the winner's weight on a phi-labelled row. Deliberately weak: this is a "
                     "prior about setup, not a measured outcome, and forcing the same "
                     "preference at inference cost -2.25pt.")
ap.add_argument("--qmin", type=float, default=0.0,"""
assert u.count(old3) == 1, "arg anchor"
u = u.replace(old3, new3)

old4 = """        if abs(qw - ql) < a.qmin:
            n_weak += 1
            continue"""
new4 = """        if abs(qw - ql) < a.qmin:
            pw, pl_ = d.get("phi_w"), d.get("phi_l")
            if (a.phi_min > 0 and pw is not None and pl_ is not None
                    and abs(pw - pl_) >= a.phi_min):
                # Q could not separate these; the board can. Order the candidates by the
                # potential and write a weak label rather than dropping the row.
                hi, lo = ((cw, cl) if pw > pl_ else (cl, cw))
                out.write(json.dumps({"prompt": d["prompt"], "cands": [hi, lo],
                                      "wc": [round(a.phi_wc, 4), round(1 - a.phi_wc, 4)]}) + "\\n")
                n_phi += 1
                n_out += 1
                continue
            n_weak += 1
            continue"""
assert u.count(old4) == 1, "qmin anchor"
u = u.replace(old4, new4)

u = u.replace("n_same, n_flat, n_weak", "n_same, n_flat, n_weak, n_phi", 1) if "n_same, n_flat, n_weak" in u else u
old5 = '''print("[mrl] %d pairs -> %d rows (same-text %d, flat %d, below-qmin %d) | beta %.2f temp %.2f qmin %.2f"
      % (n_in, n_out, n_same, n_flat, n_weak, a.beta, a.temp, a.qmin))'''
new5 = '''print("[mrl] %d pairs -> %d rows (same-text %d, flat %d, below-qmin %d, phi-labelled %d) "
      "| beta %.2f temp %.2f qmin %.2f phi-min %.2f"
      % (n_in, n_out, n_same, n_flat, n_weak, n_phi, a.beta, a.temp, a.qmin, a.phi_min))'''
assert u.count(old5) == 1, "print anchor"
u = u.replace(old5, new5)

# the counter has to exist
import re
m = re.search(r"^n_same[^\n]*=.*$", u, re.M)
assert m, "counter init"
u = u[:m.end()] + "\nn_phi = 0" + u[m.end():]

open(r + ".new", "w").write(u)
os.replace(r + ".new", r)
print("mrl_convert: --phi-min / --phi-wc added")
