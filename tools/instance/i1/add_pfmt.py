#!/usr/bin/env python3
"""Give build_rerank the prompt format the models are actually trained on.

`tools/instance/run_v39_pipeline.sh` calls it with `--pfmt current`, and that flag does not
exist -- so the committed builder cannot reproduce the pool it is supposed to have built. It
passes neither `board_facts` nor `identify` nor `menu_dedup`, and its `--deck-mode` choices do
not include `roles`, which is what PROMPT_FMT specifies. Anything it built today would differ
from what `lm/agent` renders at inference, in exactly the way this project has been burned by
before.

`--pfmt current` takes the whole format from `tools/rl_config.PROMPT_FMT` -- one source of
truth, so the builder cannot drift from the agent by forgetting a field.
"""
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

old_ser = '''    _ser = lambda o, p: serialize_stateless(  # noqa: E731
        o, deck_ids=gd.get(p), glossary=glossary, deck_name=dn.get(p),
        deck_mode=deck_mode, deck_shuffle=dshuf)'''
new_ser = '''    _ser = lambda o, p: serialize_stateless(  # noqa: E731
        o, deck_ids=gd.get(p), deck_name=dn.get(p), **fmt)'''
assert old_ser in s, "the _ser lambda is not where it was"
s = s.replace(old_ser, new_ser)

old_sig = ("    idx, path, out_dir, tag, glossary, skip_stale, deck_mode, label, sides, dshuf "
           "= job")
new_sig = "    idx, path, out_dir, tag, fmt, skip_stale, label, sides = job"
assert old_sig in s, "the shard job signature is not where it was"
s = s.replace(old_sig, new_sig)

old_job = ('    jobs = [(i, p, args.out, shard_tag, args.glossary, args.skip_stale, '
           'args.deck_mode, args.label, args.sides, args.deck_shuffle)')
new_job = '    jobs = [(i, p, args.out, shard_tag, fmt, args.skip_stale, args.label, args.sides)'
assert old_job in s, "the job tuple is not where it was"
s = s.replace(old_job, new_job)

anchor = '    ap.add_argument("--tag", required=True'
assert anchor in s
s = s.replace(anchor, '''    ap.add_argument("--pfmt", default="legacy", choices=("legacy", "current"),
                    help="'current' takes the ENTIRE prompt format from rl_config.PROMPT_FMT "
                         "(glossary, deck_mode, board_facts, identify, menu_dedup) so the pool "
                         "cannot drift from what lm/agent renders. The individual flags below "
                         "are ignored when it is set.")
''' + anchor, 1)

# build `fmt` right after the args are parsed
marker = "    args = ap.parse_args()"
assert marker in s, "cannot find the arg parse site"
s = s.replace(marker, marker + '''

    if args.pfmt == "current":
        from tools import rl_config
        fmt = dict(rl_config.PROMPT_FMT)
        print("[pfmt] current: %s" % fmt, flush=True)
    else:
        fmt = dict(glossary=args.glossary, deck_mode=args.deck_mode,
                   deck_shuffle=args.deck_shuffle)
        print("[pfmt] legacy: %s" % fmt, flush=True)''', 1)

p.write_text(s)
print("patched", p)
