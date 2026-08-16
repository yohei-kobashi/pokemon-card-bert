"""Fix the bug that ERRORed every LM submission: the deck-selection call.

The official `sample_submission/main.py` states the contract:

    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the
        # deck. The deck is a list of 60 card IDs.
        return read_deck_csv()

`agents/engine_v2.py:692` implements it (`if obs.select is None: return self.deck`), which is why
every engine_v2 submission scores. `lm/agent.py` does NOT, and the bundle's `agent()` delegates
straight to it -- so the LM crashed on the FIRST call of every episode. That matches the replays
exactly: step 1, `observation.select = null`, `action = null`, status ERROR, 0.05 s of the 600 s
bank consumed. Submissions 55044614, 55118823 and 55118843 all died this way.

Local verification never caught it because the harness calls `battle_start(deck_me, deck_op)`
first and only then steps the game -- it hands the engine the decks directly and therefore never
issues the one call Kaggle always makes first.
"""
import os

P = os.path.join(os.getcwd(), "tools/build_rerank_submission.py")
s = open(P).read()

if "deck-selection phase" in s:
    print("already patched")
    raise SystemExit(0)

OLD = '''def agent(obs_dict: dict) -> list:
    return _agent(obs_dict)'''
NEW = '''def agent(obs_dict: dict) -> list:
    # The FIRST call of every episode carries select=None and must return the 60-card DECK, not
    # an action -- see the official sample_submission/main.py. engine_v2 implements this at
    # agents/engine_v2.py:692 (`if obs.select is None: return self.deck`), which is why the
    # heuristic submissions score; lm/agent.py does not, so the LM raised on its very first call
    # and every LM submission came back ERROR. Handle it here, before anything model-related.
    if obs_dict.get("select") is None:
        return list(_deck)                    # deck-selection phase
    return _agent(obs_dict)'''
assert s.count(OLD) == 1, "agent() anchor not found"
s = s.replace(OLD, NEW)
open(P, "w").write(s)
print("patched", P)
