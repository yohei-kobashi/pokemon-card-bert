"""build_rerank: take the whole prompt format from rl_config, not from five separate flags.

The format already lives in build_rerank, build_sft, lm/agent and rl_rollout, and its own comment
says a mismatch diverges train and deploy SILENTLY. Five independent CLI flags is five chances to
get that wrong, so `--pfmt current` (the default) pulls the entire dict from
`rl_config.PROMPT_FMT`; `--pfmt v37` reproduces the old format from `PROMPT_FMT_V37` for A/B.
"""
import os
import re

P = os.path.join(os.getcwd(), "tools/build_rerank.py")
s = open(P).read()
if "--pfmt" in s:
    print("already patched")
    raise SystemExit(0)

# 1) worker: unpack two more fields and use them
OLD = ("    idx, path, out_dir, tag, glossary, skip_stale, deck_mode, label, sides, dshuf = job")
NEW = ("    (idx, path, out_dir, tag, glossary, skip_stale, deck_mode, label, sides, dshuf,\n"
       "     board_facts, identify) = job")
assert s.count(OLD) == 1, "job unpack anchor"
s = s.replace(OLD, NEW)

OLD_SER = ('''    _ser = lambda o, p: serialize_stateless(  # noqa: E731
        o, deck_ids=gd.get(p), glossary=glossary, deck_name=dn.get(p),
        deck_mode=deck_mode, deck_shuffle=dshuf)''')
NEW_SER = ('''    _ser = lambda o, p: serialize_stateless(  # noqa: E731
        o, deck_ids=gd.get(p), glossary=glossary, deck_name=dn.get(p),
        deck_mode=deck_mode, deck_shuffle=dshuf,
        board_facts=board_facts, identify=identify)''')
assert s.count(OLD_SER) == 1, "serializer anchor"
s = s.replace(OLD_SER, NEW_SER)

# 2) deck-mode choices gain "roles"
s = s.replace('ap.add_argument("--deck-mode", default="static", choices=("static", "remaining"),',
              'ap.add_argument("--deck-mode", default="static",\n'
              '                    choices=("static", "remaining", "roles"),')

# 3) the --pfmt switch
m = re.search(r'(\s+)ap\.add_argument\("--tag", required=True', s)
assert m, "tag arg"
ind = m.group(1)
s = (s[:m.start()] + ind + 'ap.add_argument("--pfmt", default="current", choices=("current", "v37"),\n'
     + ind + '                help="prompt format: \'current\' = rl_config.PROMPT_FMT (the single '
     'source of truth all four renderers read), \'v37\' = PROMPT_FMT_V37 for A/B. Overrides the '
     'individual format flags.")' + s[m.start():])

# 4) resolve the format and build the jobs with it
m = re.search(r"\n(\s+)jobs = \[", s)
assert m, "jobs anchor"
ind = m.group(1)
inject = ("\n" + ind + "import rl_config\n"
          + ind + "_fmt = dict(rl_config.PROMPT_FMT if args.pfmt == \"current\"\n"
          + ind + "            else rl_config.PROMPT_FMT_V37)\n"
          + ind + "args.glossary = _fmt.get(\"glossary\", args.glossary)\n"
          + ind + "args.deck_mode = _fmt.get(\"deck_mode\", args.deck_mode)\n"
          + ind + "_dshuf = _fmt.get(\"deck_shuffle\", args.deck_shuffle)\n"
          + ind + "_bf = _fmt.get(\"board_facts\", False)\n"
          + ind + "_idf = _fmt.get(\"identify\", \"both\")\n"
          + ind + "print(\"prompt format (%s): %s\" % (args.pfmt, _fmt), flush=True)\n")
s = s[:m.start()] + inject + s[m.start():]

# 5) every job tuple gets the two new fields
s = re.sub(r"(args\.deck_mode, args\.label, args\.sides, )args\.deck_shuffle\)",
           r"\1_dshuf, _bf, _idf)", s)
s = re.sub(r"(args\.deck_mode, args\.label, args\.sides, )_dshuf\)",
           r"\1_dshuf, _bf, _idf)", s)

open(P, "w").write(s)
print("patched", P)
