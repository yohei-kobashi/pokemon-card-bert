"""Gating test for the LM shared foundation (component E). Run:
    python lm/roundtrip_test.py

Checks, over real self-play selections:
 (1) serialize_stateless runs on every selection and is deterministic,
 (2) every option's semantic encoding decodes back to a legal, equivalent index,
 (3) the model=None LM-agent plays full games legally (== heuristic, no forfeit),
 (4) a MOCK model exercises the LM query path -- serialize_stateless -> predict ->
     decode_action -> a legal move -- on every real choice (MAIN + sub-select), with
     forced selects auto-resolved by the heuristic, and the game completes,
 (5) the tokenizer special-token list builds.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

import json                                      # noqa: E402
import library                                   # noqa: E402
from battle_log import load_agent                # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from lm import vocab                             # noqa: E402
from lm.serialize import serialize_stateless, STOP        # noqa: E402
from lm.actions import encode_option, decode_action        # noqa: E402
from lm.agent import make_lm_agent, _real_choice  # noqa: E402
from agents.engine_v2 import make_policy         # noqa: E402


def _profile(name):
    tun = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    return tun.get(name, {})


class MockModel:
    """Stand-in LM: scores candidates so the argmax reproduces engine_v2's chosen set,
    exercising the scoring path (single-pick argmax AND sequential multi-pick, incl.
    the STOP candidate) and guaranteeing legal moves. Records how often it scored."""
    def __init__(self, deck, profile):
        self.policy = make_policy(deck, profile)
        self.scored = 0

    def score(self, prompt, candidates, obs=None):
        self.scored += 1
        want = set()
        if obs is not None:
            sel = obs["select"]
            opts = sel.get("option") or []
            try:
                want = {encode_option(opts[j], obs)
                        for j in self.policy.act(obs) if 0 <= j < len(opts)}
            except Exception:
                want = set()
        # in engine's set -> highest; STOP -> middle (wins once all wanted are picked);
        # anything else -> lowest
        return [1.0 if c in want else 0.5 if c == STOP else 0.0 for c in candidates]


def check_shipped_format(deck, deck_name, max_steps=400):
    """Check [6]: the SHIPPED prompt format, not just serialize_stateless's defaults.

    Passes 1-4 call serialize_stateless with no arguments, which renders neither `DECK[...]`
    nor `ID ME d_x a_y` -- so they stayed green through both format bugs found on 2026-07-27:
    build_rerank_submission's main.py never passed deck_mode/deck_shuffle, and lm/identify's
    `import library` made the ID segment vanish from every bundled prompt. Both are silent:
    no exception, no size change, only a worse pilot. A gate that cannot see the failure mode
    that has actually shipped twice is not a gate.

    So assert the segments are PRESENT under the flags the submission bakes in, and that
    deck_shuffle is (a) deterministic for one obs -- train and inference derive the order from
    the same seed, so a mismatch silently changes the prompt -- and (b) actually permuting
    across decisions, which is the whole point of shuffling.
    """
    import re
    from lm.serialize import render_my_deck
    d1 = library.read_deck("crustle_stall" if deck_name != "crustle_stall" else "alakazam")
    prof = _profile(deck_name)
    agent = make_lm_agent(deck, profile=prof, model=None)
    opp = load_agent("crustle_stall" if deck_name != "crustle_stall" else "alakazam")

    kw = dict(deck_ids=deck, glossary="none", deck_name=deck_name,
              deck_mode="remaining", deck_shuffle=True)
    n = n_id = n_deck = n_det = 0
    orders = set()
    re_deck = re.compile(r"^DECK\[([^\]]*)\]")
    obs, _ = battle_start(deck, d1)
    try:
        for _ in range(max_steps):
            cur = obs.get("current")
            if cur is None or cur.get("result", -1) != -1 or obs.get("select") is None:
                break
            if cur["yourIndex"] == 0:
                s = serialize_stateless(obs, **kw)
                n += 1
                n_id += bool(re.search(r" ID ME d_\S+", s))
                n_deck += s.startswith("DECK[")
                n_det += (s == serialize_stateless(obs, **kw))
                m = re_deck.match(s)
                if m and m.group(1):
                    orders.add(m.group(1))
            obs = battle_select((agent, opp)[cur["yourIndex"]](obs))
    finally:
        battle_finish()

    # a deck list with no obs must still render, and 'remaining' must shrink it
    static = render_my_deck(deck, None, "static", False)
    ok = (n > 0 and n_id == n and n_deck == n and n_det == n and len(orders) > 1
          and static.startswith("DECK[") and len(orders) >= max(2, int(0.5 * n)))
    return ok, [
        "[6] shipped format (glossary=none, deck_name, remaining, shuffle) over %d decisions:" % n,
        "      ID ME present   %d/%d   DECK[ head %d/%d   deterministic %d/%d"
        % (n_id, n, n_deck, n, n_det, n),
        "      distinct DECK[] orders %d/%d (shuffle must permute, and must not be stuck)"
        % (len(orders), n),
    ]


def run(me_deck="mega_lucario", opp_deck="crustle_stall", max_steps=3000):
    d0 = library.read_deck(me_deck)
    d1 = library.read_deck(opp_deck)
    prof = _profile(me_deck)
    opp = load_agent(opp_deck)

    # --- passes 1-3: model=None agent, serializer + encode/decode round-trip ---
    me_agent = make_lm_agent(d0, profile=prof, model=None)
    n_sel = det_fail = real_choice = 0
    rt_ok = rt_total = 0
    sample = None
    obs, _ = battle_start(d0, d1)
    try:
        for _ in range(max_steps):
            cur = obs.get("current")
            if cur is None or cur.get("result", -1) != -1:
                break
            sel = obs.get("select")
            if sel is None:
                break
            yi = cur["yourIndex"]
            if yi == 0:
                n_sel += 1
                real_choice += _real_choice(sel)
                if serialize_stateless(obs) != serialize_stateless(obs):
                    det_fail += 1
                if sample is None:
                    sample = serialize_stateless(obs)
                single_ok = sel["minCount"] <= 1 <= sel["maxCount"]
                for i, o in enumerate(sel["option"]):
                    rt_total += 1
                    if not single_ok:
                        rt_ok += 1
                        continue
                    enc = encode_option(o, obs)
                    dec = decode_action(enc, obs)
                    if dec and len(dec) == 1 and encode_option(sel["option"][dec[0]], obs) == enc:
                        rt_ok += 1
            obs = battle_select((me_agent, opp)[yi](obs))
        result = obs.get("current", {}).get("result", -1)
    finally:
        battle_finish()

    # --- pass 4: mock-model agent (exercises the SCORING path incl. multi-pick) ---
    mock = MockModel(d0, prof)
    lm_agent = make_lm_agent(d0, profile=prof, model=mock)
    obs, _ = battle_start(d0, d1)
    mock_result = -1
    multipick = 0
    try:
        for _ in range(max_steps):
            cur = obs.get("current")
            if cur is None or cur.get("result", -1) != -1:
                break
            sel = obs.get("select")
            if sel is None:
                break
            yi = cur["yourIndex"]
            if yi == 0 and _real_choice(sel) and (sel.get("maxCount") or 1) >= 2:
                multipick += 1
            obs = battle_select((lm_agent, opp)[yi](obs))
        mock_result = obs.get("current", {}).get("result", -1)
    finally:
        battle_finish()

    print(f"[1] selections(me)={n_sel}  real-choice(LM-queried)={real_choice}  "
          f"forced(heuristic)={n_sel - real_choice}  determinism_fail={det_fail}")
    print(f"[2] option round-trip: {rt_ok}/{rt_total} "
          f"({100 * rt_ok / rt_total:.1f}%)" if rt_total else "[2] no options seen")
    print(f"[3] model=None game completed, result={result}")
    print(f"[4] MOCK-model game completed, result={mock_result}, scored {mock.scored} "
          f"times, handled {multipick} multi-pick selects (all legal or it would forfeit)")
    print(f"[5] special tokens: {len(vocab.special_tokens())}")
    fmt_ok, fmt_lines = check_shipped_format(d0, me_deck)
    for ln in fmt_lines:
        print(ln)
    print("\n--- sample serialize_stateless (first selection) ---")
    print(sample[:700] if sample else "(none)")
    ok = (det_fail == 0 and rt_total and rt_ok >= int(0.98 * rt_total)
          and mock.scored > 0 and mock_result != -1 and fmt_ok)
    print("\n" + ("RESULT: OK" if ok else "RESULT: CHECK FAILURES"))
    return ok


if __name__ == "__main__":
    run()
