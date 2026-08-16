"""BRANCH B, step 3 -- what prompt does the model ACTUALLY receive when it plays?

The format was checked by calling serialize_stateless directly. That is not the path the gate
uses: the gate builds an agent through mirror_match.make_agent -> lm.agent.make_lm_agent, and a
single argument dropped anywhere along that chain renders a different prompt with no error --
which is exactly how the Kaggle bundle once shipped every prompt without its ID segment
(`bundle-drops-id-segment`). So intercept the scorer and read what it is handed mid-game.

Compares against the training pool it should match: DECK present or not, and the length
distribution. A model fed prompts 60 tokens longer than anything it trained on is not a
mysterious regression.
"""
import gzip, json, os, re, sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mirror_match as mm
from tools.mirror_env import DEFAULT_SO, MirrorEngine, play

MODEL = sys.argv[1] if len(sys.argv) > 1 else "hf:/root/out/dusk_s1"
FMT = sys.argv[2] if len(sys.argv) > 2 else "dusk"
DECK, OPP = "dragapult_dusknoir", "marnie_grimmsnarl"

tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
my_ids, opp_ids = mm.load_deck(DECK), mm.load_deck(OPP)

mm._FMT = FMT
agent, scorer = mm.make_agent(MODEL, DECK, my_ids, tuning.get(DECK, {}))
if scorer is None:
    raise SystemExit("%s has no scorer to intercept" % MODEL)

seen = []
_orig = scorer.score


def spy(state, cands, *a, **k):
    seen.append(state)
    return _orig(state, cands, *a, **k)


scorer.score = spy
mm._FMT = "prompt"
opp_agent, _ = mm.make_agent("engine", OPP, opp_ids, tuning.get(OPP, {}))

eng = MirrorEngine(DEFAULT_SO)
play(eng, agent, opp_agent, my_ids, opp_ids, 4242, mirror=1)
print("captured %d prompts from one live game (model=%s fmt=%s)" % (len(seen), MODEL, FMT))
if not seen:
    raise SystemExit("the scorer was never called -- the agent fell back to engine_v2 for the "
                     "whole game, which is itself the finding")

with_deck = sum(1 for s in seen if s.lstrip().startswith("DECK"))
lens = sorted(len(s) for s in seen)
print("prompts starting with DECK: %d / %d" % (with_deck, len(seen)))
print("play-time chars   p50 %d  p90 %d  max %d" % (lens[len(lens) // 2],
                                                    lens[int(0.9 * (len(lens) - 1))], lens[-1]))

pool = []
with gzip.open(os.path.join(ROOT, "data/rerank/v41_dusk11.jsonl.gz"), "rt") as f:
    for i, line in enumerate(f):
        pool.append(len(json.loads(line)["state"]))
        if i >= 4999:
            break
pool.sort()
print("training  chars   p50 %d  p90 %d  max %d" % (pool[len(pool) // 2],
                                                    pool[int(0.9 * (len(pool) - 1))], pool[-1]))
print("\nVERDICT: %s" % ("prompts match the training format"
                         if with_deck == 0 and abs(lens[len(lens) // 2] - pool[len(pool) // 2])
                         < 0.25 * pool[len(pool) // 2]
                         else "PLAY-TIME PROMPTS DO NOT MATCH THE TRAINING POOL"))
print("sample: %r" % seen[len(seen) // 2][:120])
