import json, os
ENG = os.path.join(os.getcwd(), "agents/engine_v2.py")
TUN = os.path.join(os.getcwd(), "agents/tuning.json")
s = open(ENG).read()
A = """        if self.profile.get("bench_target") is not None:
            self.bench_target = int(self.profile["bench_target"])
"""
N = A + """        #   deck_low: the anti-deckout floor. The class default 4 was calibrated globally
        #     ("8 hurt healthy decks"), but rockets_honchkrow decks itself out in 16-20% of
        #     games at 3.25 cards drawn per own turn -- 47 library cards / 15 turns, i.e. the
        #     library empties exactly when the game ends. Per-deck opt-in so raising it cannot
        #     touch the decks the global calibration was measured on.
        if self.profile.get("deck_low") is not None:
            self.deck_low = int(self.profile["deck_low"])
"""
if "profile.get(\"deck_low\")" in s:
    print("already patched")
else:
    assert s.count(A) == 1, "anchor not unique"
    open(ENG, "w").write(s.replace(A, N))
    print("patched deck_low profile support")
t = json.load(open(TUN))
# basic_search was measured and REJECTED (win rate +0.4pt at 2700 games, bench-out just
# converted into deck-out). Drop the config so the rule is inert and this copy is a clean
# behavioural match for the main repo plus the two new opt-in knobs.
for d in ("rockets_honchkrow", "rockets_mewtwo"):
    if t.get(d, {}).pop("basic_search", None) is not None:
        print("removed basic_search from", d)
json.dump(t, open(TUN, "w"), indent=1, sort_keys=True, ensure_ascii=False)
