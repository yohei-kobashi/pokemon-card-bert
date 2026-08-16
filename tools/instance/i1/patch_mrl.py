import os, re

r = "/root/mrl_convert.py"
u = open(r).read()
assert "phi_min" not in u, "already patched"

old = '''ap.add_argument("--qmin", type=float, default=0.0,'''
new = '''ap.add_argument("--phi-min", type=float, default=0.0,
                help="rescue below-qmin pairs whose setup potentials differ by at least this. "
                     "Phi steps in halves (one useful energy = 0.5). 0 disables.")
ap.add_argument("--phi-wc", type=float, default=0.65,
                help="the winner's weight on a phi-labelled row. Deliberately weak: this is a "
                     "prior about setup, not a measured outcome, and forcing the same "
                     "preference at inference cost -2.25pt.")
ap.add_argument("--qmin", type=float, default=0.0,'''
assert u.count(old) == 1, "arg anchor"
u = u.replace(old, new)

u = u.replace("n_in = n_out = n_same = n_flat = n_weak = 0",
              "n_in = n_out = n_same = n_flat = n_weak = n_phi = 0", 1)

old4 = """        if abs(qw - ql) < a.qmin:
            n_weak += 1
            continue"""
new4 = """        if abs(qw - ql) < a.qmin:
            pw, pl_ = d.get("phi_w"), d.get("phi_l")
            if (a.phi_min > 0 and pw is not None and pl_ is not None
                    and abs(pw - pl_) >= a.phi_min):
                # Q could not separate these; the board can. Order by the potential and write a
                # WEAK label rather than dropping the row -- 64% of pairs land here.
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

old5 = '''      % (n_in, n_out, n_same, n_flat, n_weak, a.beta, a.temp, a.qmin))'''
new5 = '''      % (n_in, n_out, n_same, n_flat, n_weak, n_phi, a.beta, a.temp, a.qmin, a.phi_min))'''
assert u.count(old5) == 1, "print args anchor"
u = u.replace(old5, new5)
old6 = '''print("[mrl] %d pairs -> %d rows (same-text %d, flat %d, below-qmin %d) | beta %.2f temp %.2f qmin %.2f"'''
new6 = '''print("[mrl] %d pairs -> %d rows (same-text %d, flat %d, below-qmin %d, phi-labelled %d)"
      " | beta %.2f temp %.2f qmin %.2f phi-min %.2f"'''
assert u.count(old6) == 1, "print fmt anchor"
u = u.replace(old6, new6)

open(r + ".new", "w").write(u)
os.replace(r + ".new", r)
print("mrl_convert patched")
