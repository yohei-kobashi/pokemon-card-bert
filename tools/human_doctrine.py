#!/usr/bin/env python3
"""User-directed doctrine rows from the human play_server games (2026-08-17 directive).

Three behaviors the user wants HEAVILY rewarded, mined from the human's own decisions so no
row can contradict the human (only menus where the human actually TOOK the option emit):
  1. fez_early     bench Fezandipiti ex (c140) early (turn <= --fez-turn)
  2. stadium_over  play our Stadium while the OPPONENT's Stadium is up
  3. double_pult   evolve into a second Dragapult ex while one is already in play

Rows are the listwise {prompt, cands, wc} shape with --w (default 0.95) on the human's pick
and --copies duplicates (repeat-training, per the directive). Rendering identical to
tools/human_rows.py.
"""
import argparse
import glob
import gzip
import json
import sys

from human_rows import to_kaggle_obs, ACT  # noqa: E402  (tools/ on sys.path when run as script)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--deck", default="dragapult_dusknoir")
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--fez-turn", type=int, default=4)
    ap.add_argument("--w", type=float, default=0.95)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import library
    import rl_config
    import dusk_plan
    from lm.serialize import serialize_stateless
    from lm.actions import encode_option
    from lm.action_token import canon_key, slot_map_from_state
    from valued_to_sft import menu_of

    FEZ, PULT = 140, 121
    OUR_STADS = set(dusk_plan._OUR_STADIUMS)
    dids = [int(x) for x in open(library.deck_path(a.deck)) if x.strip()]
    fmt = dict(rl_config.DUSK_FMT)

    n = {"fez_early": 0, "stadium_over": 0, "double_pult": 0}
    rows = 0
    with gzip.open(a.out, "wt", encoding="utf-8") as out:
        for f in sorted(glob.glob(a.logs)):
            if "_vs_agent-" in f:
                continue
            d = json.load(open(f))
            for e in d:
                cur = e.get("current") or {}
                if cur.get("yourIndex") != a.seat:
                    continue
                raw = (e.get("select") or {}).get("option") or []
                sel = e.get("selected")
                if len(raw) < 2 or not isinstance(sel, list) or len(sel) != 1:
                    continue
                if not (0 <= sel[0] < len(raw)):
                    continue
                ko = to_kaggle_obs(e)
                kraw = ko["select"]["option"]
                try:
                    txt = encode_option(kraw[sel[0]], ko)
                except Exception:                              # noqa: BLE001
                    continue
                t = cur.get("turn") or 0
                me = cur["players"][a.seat]
                my_board = [(b or {}).get("id") for b in
                            (me.get("active") or []) + (me.get("bench") or [])]
                which = None
                if txt == "play:c%d" % FEZ and t <= a.fez_turn:
                    which = "fez_early"
                elif txt.startswith("play:c"):
                    try:
                        cid = int(txt[6:].split("@")[0])
                    except ValueError:
                        cid = None
                    if cid in OUR_STADS:
                        stad = {(s or {}).get("id") for s in (cur.get("stadium") or [])}
                        if stad and not (stad & OUR_STADS):
                            which = "stadium_over"
                if which is None and txt.startswith("evolve:c%d" % PULT) \
                        and my_board.count(PULT) >= 1:
                    which = "double_pult"
                if which is None:
                    continue
                try:
                    state = serialize_stateless(ko, deck_ids=dids, deck_name=a.deck, **fmt)
                    menu = menu_of(state)
                    if menu is None or len(menu) < 2:
                        continue
                    slots = slot_map_from_state(state)
                    mkeys = [canon_key(x, slots) for x in menu]
                    want = canon_key(txt, slots)
                    ih = next((i for i, k in enumerate(mkeys) if k == want), None)
                except Exception:                              # noqa: BLE001
                    continue
                if ih is None:
                    continue
                m = len(menu)
                wc = [round(a.w if i == ih else (1 - a.w) / (m - 1), 4) for i in range(m)]
                row = json.dumps({"prompt": ACT + state, "cands": menu, "wc": wc,
                                  "src": "doctrine", "why": which},
                                 ensure_ascii=False) + "\n"
                for _ in range(a.copies):
                    out.write(row)
                n[which] += 1
                rows += a.copies
    print("[doctrine] menus matched:", n, "-> %d rows (x%d copies, w=%.2f)"
          % (rows, a.copies, a.w))


if __name__ == "__main__":
    main()
