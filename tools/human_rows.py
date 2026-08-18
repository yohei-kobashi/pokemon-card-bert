#!/usr/bin/env python3
"""Human play_server games -> listwise training rows for the DeBERTa dusk loop.

play_server saves every battle to logs/ as the heroz visualize array; each entry is the
selecting player's obs (select/current/logs) PLUS a ``selected`` list -- the indices actually
chosen. That makes a Human-vs-AI log a fully labelled decision stream: the human seat's
entries are expert labels in the exact matchups we lose (ogerpon_mono, mega_abomasnow).

Rows come out in dusk_plan_train's listwise shape {"prompt": ACT+state, "cands": [menu...],
"wc": soft target}: one row per human decision, the whole deduped menu as candidates, most of
the mass on the human's pick. Rendering mirrors tools/dpo_branch.py --fmt dusk exactly
(serialize_stateless under rl_config.DUSK_FMT, canon_key matching into the deduped menu) so
these rows are drop-in additions to a round's rows file. The visualize obs has no
search_begin_input blob, so hidden dmg:+N facts are absent -- serialize degrades gracefully.

    python3 tools/human_rows.py --logs 'logs/2026081*Human*human-dragapult_dusknoir*.json' \
        --deck dragapult_dusknoir --out /tmp/human_rows.jsonl.gz
    python3 tools/human_rows.py --logs logs --since 2026-08-16 --until 2026-08-18 \
        --out /tmp/human_rows.jsonl.gz          # a date range instead of a hand-written glob
"""
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ACT = "[ACT]\n"          # must match dpo_branch / valued_to_sft / the SFT pool


def _enum_val(cls, s):
    if isinstance(s, int) or s is None:
        return s
    key = str(s).replace("_", "").casefold()
    for m in cls:
        if m.name.replace("_", "").casefold() == key:
            return m.value
    return s


def to_kaggle_obs(e):
    """visualize entry -> kaggle-obs shape.

    The heroz visualize array stores enums as C# NAMES ("ToHand", "Card"); the kaggle env
    hands the same structs over with NUMERIC values, and everything downstream (vocab.ctx_name,
    encode_option, dusk_plan's context tests) matches on the numbers. Rendering the strings
    raw produced out-of-format prompts -- "SEL ToHand"/"optCard#2" where training data says
    "SEL TO_HAND"/"card:c121@HAND3" -- which is a silent distribution shift, not an error."""
    import copy
    from cg.api import SelectContext, OptionType, SelectType
    sel = copy.deepcopy(e.get("select") or {})
    sel["type"] = _enum_val(SelectType, sel.get("type"))
    sel["context"] = _enum_val(SelectContext, sel.get("context"))
    for op in sel.get("option") or []:
        if isinstance(op, dict):
            op["type"] = _enum_val(OptionType, op.get("type"))
    return {"select": sel, "current": e.get("current"), "logs": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", default=["logs"],
                    help="dirs or globs of HumanvAI visualize logs (.json / .json.gz)")
    ap.add_argument("--since", default="", help="only games from this date, e.g. 2026-08-16. "
                    "The date comes from the FILENAME stamp, so no file is opened to filter")
    ap.add_argument("--until", default="", help="only games up to this date (inclusive)")
    ap.add_argument("--opp", default="", help="keep only games against this opponent deck "
                    "(e.g. ogerpon_mono); empty = every opponent")
    ap.add_argument("--deck", default="dragapult_dusknoir", help="the HUMAN's deck name")
    ap.add_argument("--seat", type=int, default=0, help="human's player index in the log")
    ap.add_argument("--w-win", type=float, default=0.85,
                    help="soft-target mass on the human pick in games the human WON")
    ap.add_argument("--w-loss", type=float, default=0.75,
                    help="same for lost games (still expert play, slightly discounted)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import library
    import rl_config
    from lm.serialize import serialize_stateless
    from lm.actions import encode_option
    from lm.action_token import canon_key, slot_map_from_state
    from valued_to_sft import menu_of
    from kenkyu.common import list_logs, open_log, opponent_of, parse_date

    dids = [int(x) for x in open(library.deck_path(a.deck)) if x.strip()]
    fmt = dict(rl_config.DUSK_FMT)

    # Same selector the aggregator (tools/kenkyu/log_stats.py) uses, so the rows built here
    # always correspond to the games the研究 just reported statistics for.
    files = list_logs(a.logs, parse_date(a.since), parse_date(a.until, end=True),
                      human_deck="human-" + a.deck, opp=a.opp or None)
    if not files:
        raise SystemExit("no logs matched %r (since=%r until=%r)" % (a.logs, a.since, a.until))

    st = {"games": 0, "wins": 0, "dec": 0, "multi": 0, "no_menu": 0,
          "menu_match": 0, "trivial": 0, "render_err": 0, "rows": 0}
    with gzip.open(a.out, "wt", encoding="utf-8") as out:
        for f in files:
            d = open_log(f)          # .json or .json.gz
            st["games"] += 1
            # winner: the Result log entry (result == player index of the winner)
            res = None
            for e in reversed(d):
                for lg in e.get("logs", []):
                    if lg.get("type") == "Result":
                        res = lg["result"]
                        break
                if res is not None:
                    break
            won = (res == a.seat)
            st["wins"] += 1 if won else 0
            w = a.w_win if won else a.w_loss
            opp = opponent_of(f)
            for e in d:
                cur = e.get("current") or {}
                if cur.get("yourIndex") != a.seat:
                    continue
                raw = (e.get("select") or {}).get("option") or []
                sel = e.get("selected")
                if len(raw) < 2 or not isinstance(sel, list):
                    continue
                st["dec"] += 1
                if len(sel) != 1 or not (0 <= sel[0] < len(raw)):
                    st["multi"] += 1
                    continue
                ko = to_kaggle_obs(e)
                kraw = ko["select"]["option"]
                try:
                    state = serialize_stateless(ko, deck_ids=dids, deck_name=a.deck, **fmt)
                    menu = menu_of(state)
                    if menu is None:
                        st["no_menu"] += 1
                        continue
                    if len(menu) < 2:
                        st["trivial"] += 1          # dedup collapsed the menu: nothing to rank
                        continue
                    slots = slot_map_from_state(state)
                    mkeys = [canon_key(x, slots) for x in menu]
                    want = canon_key(encode_option(kraw[sel[0]], ko), slots)
                    ih = next((i for i, k in enumerate(mkeys) if k == want), None)
                except Exception:                  # noqa: BLE001
                    st["render_err"] += 1
                    continue
                if ih is None:
                    st["menu_match"] += 1
                    continue
                n = len(menu)
                wc = [round(w if i == ih else (1 - w) / (n - 1), 4) for i in range(n)]
                out.write(json.dumps({"prompt": ACT + state, "cands": menu, "wc": wc,
                                      "src": "human", "opp": opp, "won": won, "hi": ih,
                                      "t": cur.get("turn")}, ensure_ascii=False) + "\n")
                st["rows"] += 1
    print("[human] %(games)d games (%(wins)d won) %(dec)d decisions -> %(rows)d rows | "
          "skipped: multi %(multi)d, no_menu %(no_menu)d, menu_match %(menu_match)d, "
          "trivial %(trivial)d, render_err %(render_err)d" % st)


if __name__ == "__main__":
    main()
