"""Same-deck / same-shuffle battles: common random numbers between the two seats.

Stock `cg.game.battle_start` gives each player an independently shuffled deck from an unseeded
RNG, so a win is part play and part "I drew better". This drives the patched engine built by
tools/build_engine_mirror.py, which seeds a separate stream per player from the SAME seed. Give
both seats the same 60 cards and their decks come out in an identical order (verified card by
card, all 60), so shuffle luck is held fixed and only the piloting differs.

    python3 tools/build_engine_mirror.py --fetch     # once: fetch + patch + build
    python3 tools/mirror_env.py --selftest           # deck order / determinism checks
    python3 tools/mirror_env.py --compare            # measure the variance reduction

The .so lives under data/ (gitignored) because the engine source is Competition-Use-Only and
this repo is public. This module carries no engine code -- only the C signatures.

One process plays one battle at a time: the engine keeps a single global battle pointer, same
as cg.game. Parallelise across processes.
"""

import argparse
import ctypes
import os
import random
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_SO = os.path.join(ROOT, "data", "kaggle_engine_ext", "libcg_mirror.so")


class _StartData(ctypes.Structure):
    _fields_ = [("battlePtr", ctypes.c_void_p), ("errorPlayer", ctypes.c_int),
                ("errorType", ctypes.c_int)]


class _SerialData(ctypes.Structure):
    _fields_ = [("json", ctypes.c_char_p), ("data", ctypes.POINTER(ctypes.c_ubyte)),
                ("count", ctypes.c_int), ("selectPlayer", ctypes.c_int)]


class MirrorEngine:
    """ctypes front end for the patched engine. `mirror=1` is the point of this file."""

    def __init__(self, so_path=DEFAULT_SO):
        if not os.path.exists(so_path):
            raise FileNotFoundError(
                f"{so_path} not found -- run: python3 tools/build_engine_mirror.py --fetch")
        self.lib = ctypes.cdll.LoadLibrary(so_path)
        self.lib.GameInitialize()
        self.lib.BattleStartEx.restype = _StartData
        self.lib.BattleStartEx.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint,
                                           ctypes.c_int, ctypes.c_int]
        self.lib.GetBattleData.restype = _SerialData
        self.lib.GetBattleData.argtypes = [ctypes.c_void_p]
        self.lib.Select.restype = ctypes.c_int
        self.lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.BattleFinish.argtypes = [ctypes.c_void_p]
        self.lib.DebugDeckIds.restype = ctypes.c_int
        self.lib.DebugDeckIds.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        # Only libcg_hidden.so exports this; the plain mirror build does not.
        self.has_hidden = hasattr(self.lib, "DebugHiddenState")
        if self.has_hidden:
            self.lib.DebugHiddenState.restype = ctypes.c_char_p
            self.lib.DebugHiddenState.argtypes = [ctypes.c_void_p]
            self.lib.DebugCardDeps.restype = ctypes.c_char_p
            self.lib.DebugCardDeps.argtypes = []
            self.lib.DebugCalcDamage.restype = ctypes.c_char_p
            self.lib.DebugCalcDamage.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4
            self.lib.DebugRetreatCost.restype = ctypes.c_int
            self.lib.DebugRetreatCost.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.lib.DebugInsufficientEnergy.restype = ctypes.c_int
            self.lib.DebugInsufficientEnergy.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                         ctypes.c_int]
        self.ptr = None

    def _obs(self):
        import json
        sd = self.lib.GetBattleData(self.ptr)
        obs = json.loads(sd.json.decode())
        obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
        return obs

    def start(self, deck0, deck1, seed, mirror=1, device_rand=0):
        if len(deck0) != 60 or len(deck1) != 60:
            raise ValueError("The deck must contain 60 cards.")
        cards = list(deck0) + list(deck1)
        arg = (ctypes.c_int * len(cards))(*cards)
        sd = self.lib.BattleStartEx(arg, ctypes.c_uint(seed & 0xFFFFFFFF), device_rand, mirror)
        self.ptr = sd.battlePtr
        if not self.ptr:
            return None
        return self._obs()

    def select(self, choices):
        arg = (ctypes.c_int * len(choices))(*choices)
        err = self.lib.Select(self.ptr, arg, len(choices))
        if err:
            raise IndexError(f"Select error {err}")
        return self._obs()

    def finish(self):
        if self.ptr:
            self.lib.BattleFinish(self.ptr)
            self.ptr = None

    def hidden_state(self):
        """Everything the observation drops: per-card / per-player / per-game effect state.

        Needs the instrumented build (`--out data/kaggle_engine_ext/libcg_hidden.so`). Returns
        `{"cards": [...], "players": [...], "game": {...}}` with only NON-ZERO fields, grouped
        H(istory) / T(his turn) / F(uture) / C(ontinual) -- see DebugHiddenState in
        mirror_export.cpp for what each class means. `{}` if the build lacks the symbol.
        """
        import json
        if not self.has_hidden or not self.ptr:
            return {}
        raw = self.lib.DebugHiddenState(self.ptr)
        return json.loads(raw.decode()) if raw else {}

    def calc_damage(self, attacker_serial, target_serial, base, attack_id):
        """The engine's own CalcDamage for one triple -- the oracle lm/hidden.py is diffed
        against. Instrumented build only."""
        import json
        raw = self.lib.DebugCalcDamage(self.ptr, attacker_serial, target_serial, base, attack_id)
        return json.loads(raw.decode()) if raw else None

    def retreat_cost(self, serial):
        """The engine's own State::retreatCost. Instrumented build only."""
        return self.lib.DebugRetreatCost(self.ptr, serial)

    def insufficient_energy(self, serial, attack_id):
        """The engine's own GameUtil.h:InsufficientEnergyCount. Instrumented build only."""
        return self.lib.DebugInsufficientEnergy(self.ptr, serial, attack_id)

    def card_deps(self):
        """`{"attacks": {attackId: ["H:koPreEnemyTurn", ...]}, "skills": {...}}` -- which cards
        actually READ the past. Static (card database), so no battle needs to be running."""
        import json
        if not self.has_hidden:
            return {"attacks": {}, "skills": {}}
        return json.loads(self.lib.DebugCardDeps().decode())

    def deck_ids(self, player):
        """Current deck order for `player`, top first. Debug/inspection only."""
        buf = (ctypes.c_int * 60)()
        n = self.lib.DebugDeckIds(self.ptr, player, buf, 60)
        return list(buf[:n]) if n >= 0 else []


def engine_fingerprint(eng, deck_ids, seeds=(1, 2, 3)):
    """Hash of the seed -> deck-order mapping. THE invariant that has to match across machines.

    The .so's own sha256 is too strict: instance2's binary hashes differently from the local one
    (embedded paths, toolchain patch level) yet deals byte-identical games. What actually has to
    agree is the permutation std::shuffle produces from a given mt19937 state -- which the C++
    standard does NOT specify, so it can genuinely differ across libstdc++ versions. This
    fingerprints that directly, so a real divergence is caught and a cosmetic one is not.

    Fingerprints (engine build + this decklist): editing the deck changes it too, which is
    correct -- both determine the game.
    """
    import hashlib
    h = hashlib.sha256()
    for s in seeds:
        eng.start(deck_ids, list(deck_ids), s, mirror=1)
        h.update(bytes(str(eng.deck_ids(0)), "ascii"))
        eng.finish()
    return h.hexdigest()[:16]


def play(eng, agent0, agent1, deck0, deck1, seed, mirror=1, max_steps=4000, device_rand=0):
    """One battle. Returns the winner index (0/1), or None for a draw/timeout.

    Same forfeit semantics as tools/arena.play: an illegal selection loses that game rather
    than crashing the run.
    """
    obs = eng.start(deck0, deck1, seed, mirror=mirror, device_rand=device_rand)
    if obs is None:
        return None
    agents = (agent0, agent1)
    try:
        for _ in range(max_steps):
            cur = obs.get("current")
            if cur is None:
                return None
            if cur.get("result", -1) != -1:
                return cur["result"]
            sel = obs.get("select")
            if sel is None:
                return None
            yi = cur["yourIndex"]
            try:
                obs = eng.select(agents[yi](obs))
            except Exception:
                return 1 - yi
        return None
    finally:
        eng.finish()


def paired_match(eng, agentA, agentB, deck, seeds, mirror=1, device_rand=0):
    """Play each seed twice with the seats swapped. Returns (winsA, winsB, draws, pairs).

    `pairs` holds one +1/0/-1 per seed: A's seat-averaged result on THAT shuffle. Because both
    games of a pair use the same deck order for both players, a pair scores 0 whenever the two
    pilots are interchangeable on that shuffle, and only differences in play survive. Feed
    `pairs` to a paired t-test rather than treating the 2N games as independent.
    """
    wa = wb = dr = 0
    pairs = []
    for s in seeds:
        got = []
        for a_first in (True, False):
            r = play(eng, agentA if a_first else agentB, agentB if a_first else agentA,
                     deck, deck, s, mirror=mirror, device_rand=device_rand)
            if r is None:
                got.append(0)
                dr += 1
                continue
            a_won = (r == 0) if a_first else (r == 1)
            got.append(1 if a_won else -1)
            wa += a_won
            wb += not a_won
        pairs.append(sum(got) / 2.0)
    return wa, wb, dr, pairs


# --------------------------------------------------------------------------- checks


def _load_deck(name):
    import library
    with open(library.deck_path(name)) as f:
        return [int(l) for l in f if l.strip()]


def _engine_agent(deck_ids):
    from lm.agent import make_lm_agent
    return make_lm_agent(deck_ids, None, model=None)


def _noisy(agent, q, salt=0):
    """`agent`, but on a q-fraction of states it plays a random legal move instead.

    DETERMINISTIC in the state: the coin is a hash of the observation, not a live RNG. A pilot
    that carries its own randomness would re-randomise between the two games of a pair and
    destroy the very common random numbers we are trying to measure -- and real policies
    (engine_v2, an LM at temperature 0) are deterministic anyway.
    """
    import hashlib
    import json

    def f(obs):
        key = json.dumps(obs["current"], sort_keys=True).encode()
        d = hashlib.blake2b(key, digest_size=8, salt=str(salt).encode()[:16]).digest()
        u = int.from_bytes(d, "big") / 2**64
        if u < q:
            sel = obs["select"]
            n = len(sel["option"])
            k = min(max(sel["minCount"], min(sel["maxCount"], 1)), n)
            if k <= 0:
                return []
            r = random.Random(int.from_bytes(d, "big"))
            return r.sample(range(n), k)
        return agent(obs)
    return f


def selftest(so, deck_name):
    eng = MirrorEngine(so)
    deck = _load_deck(deck_name)
    print(f"deck: {deck_name}")
    for mirror in (1, 0):
        for seed in (12345, 777, 4242):
            eng.start(deck, list(deck), seed, mirror=mirror)
            p0, p1 = eng.deck_ids(0), eng.deck_ids(1)
            eng.finish()
            same = p0 == p1
            print(f"  mirror={mirror} seed={seed:<6} deck order identical: {same}"
                  f"  ({len(p0)}/{len(p1)} cards)")
    first = [tuple(_replay_digest(eng, deck, 4242)) for _ in range(3)]
    print(f"  determinism (same seed x3): {len(set(first)) == 1}")


def _replay_digest(eng, deck, seed):
    import hashlib
    import json
    obs = eng.start(deck, list(deck), seed, mirror=1)
    h = hashlib.sha256()
    for _ in range(4000):
        sel = obs.get("select")
        if not sel or not sel.get("option"):
            break
        if obs["current"].get("result", -1) != -1:
            break
        h.update(json.dumps(obs["current"], sort_keys=True).encode())
        try:
            obs = eng.select(list(range(max(1, sel["minCount"]))))
        except Exception:
            break
    eng.finish()
    return (h.hexdigest()[:16],)


def compare(so, deck_name, pairs_n, q, seed0):
    """Does holding the shuffle fixed actually sharpen the A-vs-B estimate?

    The same contrast in three arms -- engine vs the same engine handicapped by `q`, so the
    true gap is identical and only how the games are drawn changes:

      stock    unseeded, as tools/mirror_match.py plays today; no common random numbers
      seeded   both games of a pair replay the same seed, but the two seats hold DIFFERENT
               shuffles of the deck (this is what a plain seed buys)
      mirror   as seeded, and both seats hold the SAME shuffle

    Reported per arm: the seat-averaged pair score (+1 = the strong pilot won from both seats
    on that shuffle, 0 = the seats split, so turn order decided it and the pair says nothing),
    its sd, and t = mean / SE. t is the number that matters -- an arm with a bigger gap but a
    proportionally bigger sd has bought nothing.
    """
    eng = MirrorEngine(so)
    deck = _load_deck(deck_name)
    base = _engine_agent(deck)
    weak = _noisy(base, q, salt=1)
    print(f"deck {deck_name} | {pairs_n} pairs ({2*pairs_n} games) per arm | handicap q={q}\n")
    print(f"  {'arm':<8} {'strong side':>12} {'pair mean':>10} {'sd':>7} {'t':>7} "
          f"{'split':>7} {'games for t=2':>14}")
    for label, mirror, device in (("stock", 0, 1), ("seeded", 0, 0), ("mirror", 1, 0)):
        seeds = [seed0 + i for i in range(pairs_n)]
        wa, wb, dr, pairs = paired_match(eng, base, weak, deck, seeds, mirror=mirror,
                                         device_rand=device)
        m = statistics.mean(pairs)
        sd = statistics.stdev(pairs) if len(pairs) > 1 else 0.0
        se = sd / len(pairs) ** 0.5 if sd else float("inf")
        t = m / se if se else 0.0
        split = sum(1 for p in pairs if p == 0) / len(pairs)
        need = (2 * (sd / m) ** 2 * 4) if m else float("inf")   # games for t=2
        print(f"  {label:<8} {wa/max(1,wa+wb)*100:11.1f}% {m:+10.3f} {sd:7.3f} {t:+7.2f} "
              f"{split*100:6.1f}% {need:14.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default=DEFAULT_SO)
    ap.add_argument("--deck", default="crustle_stall")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--q", type=float, default=0.15)
    ap.add_argument("--seed0", type=int, default=1000)
    a = ap.parse_args()
    if a.selftest:
        selftest(a.so, a.deck)
    if a.compare:
        compare(a.so, a.deck, a.pairs, a.q, a.seed0)
