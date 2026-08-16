"""Add a need-gated Basic-Pokemon search rule to engine_v2, plus the two per-deck opt-ins.

Run from the repo root you want to modify (use the isolated copy while RL is running).
"""
import json
import os
import sys

ROOT = os.getcwd()
ENG = os.path.join(ROOT, "agents/engine_v2.py")
TUN = os.path.join(ROOT, "agents/tuning.json")

HELPER = '''
    def _basic_search_play(self, ctx, opt):
        """Search the deck for a Basic Pokemon when the board is thin and the hand holds none.

        Opt-in `basic_search`, ORDERED by preference (fetch 3 bodies before fetching 1).

        MEASURED HOLE (2026-07-30, /root/loss_causes.py + /root/probe_honchkrow.py):
        rockets_honchkrow loses 152 of 300 games to bench-out, and **0 of those losses had a
        Basic in hand** -- it was never declining to bench, it had nothing to bench. The answer
        is in the deck: Team Rocket's Proton searches for up to THREE Basic Team Rocket's
        Pokemon, 4 copies. But `agents/tuning.json` registers Proton in `draw_supporters`, and
        the draw branch fires only while `hand_count <= draw_threshold` (5). Team Rocket decks
        inflate the hand (Ariana draws to 5-8, Roto-Stick adds Supporters), so the gate blocks
        Proton exactly when the bench is empty and the hand is full: Proton sat in hand at 470
        of 939 one-KO-from-death decisions, and 71.1% of the bench-out losses never played a
        single copy of four.

        Gate: board at/below bench_target AND **no Basic Pokemon in hand at all**. The second
        clause is what stops this churning -- with a Basic in hand `decide_bench` already puts
        it down, so this rule only fires when the deck genuinely cannot produce a body. It is
        also self-limiting: a successful fetch puts a Basic in hand, which closes the gate until
        that Basic is benched. Deck-low still blocks it, because searching a near-empty deck is
        the deckout the DECK-LOW GUARD above exists to prevent.

        Not a bucket move. Putting Proton in `search_items` would make it fire every turn with
        no gate (engine_v2.py's _SEARCH_ITEMS branch has none), burning 4 copies early and
        eating the Supporter slot Ariana needs -- the "fixing is not automatically a win" trap
        from the Dawn measurement (fired 3x more, moved winrate 38 -> 38)."""
        cards = list(self.profile.get("basic_search") or ())
        if not cards or ctx.me.deck_count <= self.deck_low:
            return None
        if len(ctx.me.inplay()) > self.bench_target:
            return None
        for x in (getattr(ctx.me_ps, "hand", None) or []):
            c = _CARDS.get(getattr(x, "id", None))
            if c is not None and c.basic:
                return None                     # can bench without searching
        for cid in cards:                       # config order = priority
            for i in ctx.plays:
                c = ctx.hand_card(opt[i])
                if c is None or c.cardId != cid:
                    continue
                if c.cardType == CardType.SUPPORTER and ctx.state.supporterPlayed:
                    continue
                return [i]
        return None
'''

CALL_ANCHOR = """        deck_low = ctx.me.deck_count <= self.deck_low
        # Supporters (one per turn)
"""
CALL_NEW = """        deck_low = ctx.me.deck_count <= self.deck_low
        # BASIC-SEARCH (2026-07-30): before the supporter/item ladder, because the card that
        # fixes an empty bench is often registered as a draw supporter and would otherwise be
        # gated on hand size. Fires only when the hand holds NO Basic -- see the helper.
        i = self._basic_search_play(ctx, opt)
        if i is not None:
            return i
        # Supporters (one per turn)
"""

HELPER_ANCHOR = "    def _deck_recover_play(self, ctx, opt):"


def main():
    s = open(ENG).read()
    if "_basic_search_play" in s:
        print("engine already patched; skipping")
    else:
        assert s.count(HELPER_ANCHOR) == 1, "helper anchor not unique"
        s = s.replace(HELPER_ANCHOR, HELPER.strip("\n") + "\n\n" + HELPER_ANCHOR)
        assert s.count(CALL_ANCHOR) == 1, "call anchor not unique (%d)" % s.count(CALL_ANCHOR)
        s = s.replace(CALL_ANCHOR, CALL_NEW)
        open(ENG, "w").write(s)
        print("patched", ENG)

    t = json.load(open(TUN))
    # ORDER = priority. Proton first (3 bodies), then the single-target searches.
    opts = {
        "rockets_honchkrow": [1220, 1152, 1121],   # Proton x4, Poke Pad x4, Ultra Ball x1
        "rockets_mewtwo": [1220, 1121],            # Proton x2, Ultra Ball x4 (no Poke Pad)
    }
    for deck, ids in opts.items():
        if deck not in t:
            print("MISSING deck in tuning.json:", deck)
            continue
        t[deck]["basic_search"] = ids
        print("set %-20s basic_search = %s" % (deck, ids))
    json.dump(t, open(TUN, "w"), indent=1, sort_keys=True, ensure_ascii=False)
    print("wrote", TUN)


if __name__ == "__main__":
    main()
