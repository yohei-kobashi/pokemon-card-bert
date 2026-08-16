#!/usr/bin/env python3
"""Read and edit the deck -> adapter registry (models/adapters.json).

    tools/adapters.py list                        # every deck, resolved, with presence here
    tools/adapters.py resolve dragapult_dusknoir  # the spec to paste into --arm
    tools/adapters.py set dragapult_dusknoir --target hf:mrl_r2 --fmt dusk
    tools/adapters.py set alakazam --target qwen:lora_alakazam_r1 --defer attach
    tools/adapters.py gate dragapult_dusknoir --win 61.5 --games 600 --opp dragapult_dusknoir
    tools/adapters.py rm alakazam
    tools/adapters.py check                       # non-zero exit if an entry is unusable HERE

`resolve` prints a concrete spec, so a command line never names an adapter directly:

    python3 tools/gate_protagonist.py --deck $D --opp $D --games 600 \\
        --arm "reg=$(tools/adapters.py resolve $D --with-fmt)"
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lm import registry as R          # noqa: E402


def cmd_list(a):
    reg = R.load(a.file)
    root = R.models_root(reg)
    print("registry %s" % R.registry_path(a.file))
    print("root     %s" % root)
    d = reg.get("default")
    if d:
        print("default  %s (fmt %s)" % (d.get("target"), d.get("fmt", "prompt")))
    print()
    print("%-22s %-34s %-6s %-6s %s" % ("deck", "spec", "fmt", "here", "gate"))
    for deck in sorted(reg.get("decks") or {}):
        try:
            r = R.resolve(deck, reg, require_exists=False)
        except R.RegistryError as ex:
            print("%-22s %s" % (deck, ex))
            continue
        g = r["entry"].get("gate") or {}
        gs = ("%.1f%% n=%d vs %s" % (g["win"], g.get("games", 0), g.get("opp", "?"))
              if g.get("win") is not None else "-")
        print("%-22s %-34s %-6s %-6s %s"
              % (deck, r["spec"], r["fmt"], "yes" if r["exists"] else "NO", gs))
    return 0


def cmd_resolve(a):
    try:
        s = R.spec_for(a.deck, with_fmt=a.with_fmt, require_exists=not a.allow_missing)
    except R.RegistryError as ex:
        print(str(ex), file=sys.stderr)
        return 2
    print(s)
    return 0


def cmd_show(a):
    r = R.resolve(a.deck, require_exists=False)
    print(json.dumps(r, indent=2, sort_keys=True, default=str))
    return 0


def cmd_set(a):
    e = R.set_deck(a.deck, target=a.target, fmt=a.fmt,
                   defer=[x for x in a.defer.split(",") if x] if a.defer is not None else None,
                   wrap=a.wrap, note=a.note, path=a.file)
    print("%s -> %s" % (a.deck, json.dumps(e, sort_keys=True)))
    return 0


def cmd_gate(a):
    """Record the measurement an entry was chosen on. Kept next to the adapter because
    'which checkpoint won' is exactly what gets lost between a gate run and a submission."""
    g = {"win": a.win, "games": a.games, "opp": a.opp, "vs": a.vs, "date": a.date}
    e = R.set_deck(a.deck, gate={k: v for k, v in g.items() if v is not None}, path=a.file)
    print("%s gate -> %s" % (a.deck, json.dumps(e.get("gate"), sort_keys=True)))
    return 0


def cmd_rm(a):
    gone = R.remove_deck(a.deck, path=a.file)
    print("removed %s" % a.deck if gone else "no entry for %s" % a.deck)
    return 0


def cmd_check(a):
    rows = R.check(R.load(a.file))
    bad = 0
    for deck, ok, detail in rows:
        print("%-4s %-22s %s" % ("ok" if ok else "MISS", deck, detail))
        bad += 0 if ok else 1
    if not rows:
        print("registry has no deck entries")
    print("%d entr%s, %d missing here" % (len(rows), "y" if len(rows) == 1 else "ies", bad))
    bad += _check_remote(a.file)
    return 1 if bad else 0


def _check_remote(file=None):
    """`resolve` cannot test a remote target -- the weights are on another machine -- so the
    equivalent check happens here: ask each server whether it really holds that adapter.

    This is the check that matters most, because it is the one whose absence is silent. A deck
    pointed at an adapter the server does not serve gets an HTTP 400 per decision, and
    lm/agent.py turns every one of those into an engine_v2 move: the run finishes, the log says
    `reg`, and the opponent was never the 4B."""
    import json as _j
    import urllib.request
    reg = R.load(file)
    want = {}
    for deck in sorted(reg.get("decks") or {}):
        try:
            t = R.resolve(deck, reg, require_exists=False)["target"]
        except R.RegistryError:
            continue
        kind, _, rest = t.partition(":")
        if kind == "remote":
            url, _, name = rest.partition("|")
            want.setdefault(url, []).append((deck, name))
    bad = 0
    for url, pairs in sorted(want.items()):
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=10) as r:
                have = set(_j.loads(r.read().decode()).get("adapters") or [])
        except Exception as ex:                                          # noqa: BLE001
            print("DOWN %s unreachable (%s) -- %d deck(s) would play as engine_v2"
                  % (url, ex, len(pairs)))
            bad += len(pairs)
            continue
        for deck, name in pairs:
            ok = (not name) or name in have
            print("%-4s %-22s remote %s|%s" % ("ok" if ok else "MISS", deck, url, name))
            bad += 0 if ok else 1
    return bad


def cmd_to_remote(a):
    """Repoint decks from local qwen: adapters at a score server, or back again.

    One command each way on purpose. The handover flips eight opponents at once, and the revert
    has to be as fast as the flip: if instance2 goes down mid-round there is no time to hand-edit
    eight entries while a gate is measuring the wrong pilot."""
    reg = R.load(a.file)
    decks = [d for d in a.decks.split(",") if d] or sorted(reg.get("decks") or {})
    n = 0
    for deck in decks:
        e = (reg.get("decks") or {}).get(deck)
        if not e:
            print("skip %-22s no entry" % deck)
            continue
        t = R._abs_target(R._entry_target(e), reg)
        kind, _, rest = t.partition(":")
        if a.back:
            if kind != "remote":
                continue
            _url, _, name = rest.partition("|")
            new = "qwen:%s" % name
        else:
            if kind == "remote":
                continue
            if kind != "qwen":
                print("skip %-22s target is %s, not a qwen adapter" % (deck, kind))
                continue
            new = "remote:%s|%s" % (a.url.rstrip("/"), os.path.basename(rest.rstrip("/")))
        R.set_deck(deck, target=new, path=a.file)
        print("%-22s %s -> %s" % (deck, t, new))
        n += 1
    print("%d entr%s rewritten" % (n, "y" if n == 1 else "ies"))
    if not a.back and n:
        print("\nnow verify the server actually serves them:")
        print("    tools/adapters.py check")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="", help="registry path (default models/adapters.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(fn=cmd_list)

    p = sub.add_parser("resolve")
    p.add_argument("deck")
    p.add_argument("--with-fmt", action="store_true", help="append '@dusk' for gate_protagonist")
    p.add_argument("--allow-missing", action="store_true")
    p.set_defaults(fn=cmd_resolve)

    p = sub.add_parser("show")
    p.add_argument("deck")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("set")
    p.add_argument("deck")
    p.add_argument("--target", help="qwen:<dir> | hf:<dir> | rerank:<dir> | engine")
    p.add_argument("--fmt", choices=R.FMTS)
    p.add_argument("--defer", help="comma-separated action kinds handed to engine_v2")
    p.add_argument("--wrap", help="spec prefix, e.g. 'planengine:recon'")
    p.add_argument("--note")
    p.set_defaults(fn=cmd_set)

    p = sub.add_parser("gate")
    p.add_argument("deck")
    p.add_argument("--win", type=float, required=True)
    p.add_argument("--games", type=int)
    p.add_argument("--opp")
    p.add_argument("--vs", default="engine_v2")
    p.add_argument("--date")
    p.set_defaults(fn=cmd_gate)

    p = sub.add_parser("rm")
    p.add_argument("deck")
    p.set_defaults(fn=cmd_rm)

    sub.add_parser("check").set_defaults(fn=cmd_check)

    p = sub.add_parser("to-remote", help="point qwen: decks at a score server (or --back)")
    p.add_argument("--url", default="http://127.0.0.1:8077")
    p.add_argument("--decks", default="", help="default: every deck with a qwen: target")
    p.add_argument("--back", action="store_true", help="revert remote: entries to qwen:")
    p.set_defaults(fn=cmd_to_remote)

    a = ap.parse_args()
    a.file = a.file or None
    raise SystemExit(a.fn(a))


if __name__ == "__main__":
    main()
