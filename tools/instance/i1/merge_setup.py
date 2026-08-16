"""Merge the setup work into the live tree.  Run ONLY at a round boundary: rules_fp is the md5 of
dusk_plan.py + plan_filter.py, and field_chain discards a round whose fingerprint moved under it.

WHAT IS BEING MERGED, and what each part is worth (3 arms x 8 opponents x 250 games, paired):

    cur   28.9%                          the shipped six rules
    sb    33.0%   +4.10 +- 1.09  t 3.75   + search_bottom
    ss    35.5%   +6.55 +- 1.13  t 5.82   + setup_search as well

Seven of the eight opponents improve; the eighth (ethan_hooh under sb alone) gives up 0.8pt and
recovers to +9.6 under ss.  The setup metrics move monotonically with the win rate across the
three arms -- Dreepy on our turn 1: 1.15 / 1.23 / 1.69, Drakloak on our turn 2: 0.40 / 0.50 /
0.72, and a first Phantom Dive in 41% / 45% / 53% of games -- so the thing the rules were
written to fix is the thing that moved.

Three files:
  tools/dusk_plan.py       search_bottom (prohibition), setup_search (positive), bench_line no
                           longer scores a Budew as a perfect opening while the line is short
  lm/plan_filter.py        search_bottom in PROHIBITIONS, and the up-to-1 fix -- the filter used
                           to hand every "choose up to 1" menu straight back to the model, which
                           is nearly every deck search, so a search rule could not act at all.
                           Default ON here (PLAN_UPTO1=0 reverts): measured neutral on its own
                           (the fix-only arm was byte-identical to the shipped arm) and it is
                           what lets the two new rules do anything.
  tools/gate_protagonist.py  reports setup speed beside the win rate, so every future round is
                           judged on both.
"""
import os
import shutil
import subprocess

SRC, DST = "/root/ptcg/repo_sb", "/root/ptcg/repo"
BAK = "/root/merge_backup"
os.makedirs(BAK, exist_ok=True)

files = ["tools/dusk_plan.py", "lm/plan_filter.py", "tools/gate_protagonist.py"]
for f in files:
    shutil.copy2(os.path.join(DST, f), os.path.join(BAK, f.replace("/", "_")))
    shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))
print("copied %d files (backups in %s)" % (len(files), BAK))

# the up-to-1 fix ships ON
p = os.path.join(DST, "lm/plan_filter.py")
s = open(p).read()
old = '_upto1 = os.environ.get("SB_UPTO1", "") not in ("", "0")'
new = '_upto1 = os.environ.get("PLAN_UPTO1", "1") not in ("", "0")'
assert s.count(old) == 1, "upto1 anchor"
open(p, "w").write(s.replace(old, new))
print("plan_filter: up-to-1 handling defaults ON (PLAN_UPTO1=0 reverts)")

# and the loop trains with the new rules in the wrapper
q = "/root/field_chain.sh"
t = open(q).read()
old = "WRAP_RULES=${WRAP_RULES:-lethal_now,$PROH}"
new = ("# search_bottom (prohibition) and setup_search (positive) added 08-14 on +4.10 and\n"
       "# +6.55 paired points over 6,000 games; both read only our own board.\n"
       "WRAP_RULES=${WRAP_RULES:-lethal_now,$PROH,search_bottom,setup_search}")
assert t.count(old) == 1, "WRAP_RULES anchor"
open(q + ".new", "w").write(t.replace(old, new))
os.replace(q + ".new", q)
os.chmod(q, 0o755)
print("field_chain.sh: WRAP_RULES now carries search_bottom,setup_search")

# ---- verify -----------------------------------------------------------------------------
os.chdir(DST)
env = dict(os.environ, PYTHONPATH="cg-lib:tools")
r = subprocess.run(["python3", "-c",
                    "import sys; sys.path.insert(0,'tools')\n"
                    "import dusk_plan as d, lm.plan_filter as f\n"
                    "assert 'search_bottom' in d.RULES and 'setup_search' in d.RULES, d.RULES\n"
                    "assert 'search_bottom' in f.PROHIBITIONS\n"
                    "print('rules ok:', len(d.RULES), 'rules,', len(f.PROHIBITIONS), 'prohibitions')"],
                   env=env, capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip()[-500:])
assert r.returncode == 0, "verification failed"
print(subprocess.run(["bash", "-n", q], capture_output=True, text=True).returncode == 0
      and "field_chain.sh parses" or "field_chain.sh DOES NOT PARSE")
print("new rules_fp:", subprocess.run(
    "md5sum tools/dusk_plan.py lm/plan_filter.py | awk '{print $1}' | md5sum | cut -d' ' -f1",
    shell=True, capture_output=True, text=True).stdout.strip())
