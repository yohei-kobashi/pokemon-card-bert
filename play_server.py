"""Human vs AI interactive battle server for the cabt (Pokemon TCG) environment.

- Drives the battle with the low-level cg-lib API (battle_start / battle_select).
- Player 0 = human (chooses moves by clicking option cards in the browser).
- Player 1 = AI  (uses agent() from agents/<AI_AGENT_NAME>.py).
- The board is rendered by the REAL HEROZ visualizer: the server feeds the live
  game data to https://ptcgvis.heroz.jp/Visualizer/Replay and proxies the result
  (and its assets) so the look is identical to the official visualizer.

This one server also hosts the deck & agent management page at /manage, so a
single process covers both playing and choosing decks/agents (config.json).

Run:
    python play_server.py            # then open http://localhost:8000/
                                     #   /        -> play Human vs AI
                                     #   /manage  -> pick decks & agents
"""

import json
import os
import csv
import random
import re
import sys
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "cg-lib")
# The seeded engine build (a superset of the stock exports) makes battles
# deterministic per seed, which is what the undo feature replays against.
# Set BEFORE cg.game is imported, unless the caller already chose a lib.
_SEEDED_SO = os.path.join("data", "kaggle_engine_ext", "libcg_seeded.so")
if "CG_LIB" not in os.environ and os.path.exists(_SEEDED_SO):
    os.environ["CG_LIB"] = os.path.abspath(_SEEDED_SO)
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from cg.game import _get_battle_data  # noqa: E402  (obs builder, used by seeded start)
from cg.sim import Battle, StartData, lib as _cglib  # noqa: E402
import ctypes  # noqa: E402


def _battle_start_seeded(deck0, deck1, seed):
    """BattleStartSeeded via the seeded engine build; None if unavailable."""
    if not hasattr(_cglib, "BattleStartSeeded"):
        return None
    if not getattr(_battle_start_seeded, "_init", False):
        _cglib.BattleStartSeeded.restype = StartData
        _cglib.BattleStartSeeded.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_int]
        _battle_start_seeded._init = True
    cards = list(deck0) + list(deck1)
    arg = (ctypes.c_int * len(cards))(*cards)
    # deviceRand=0: EVERY in-game random event (shuffles, coin flips, mulligan
    # draws) comes from the seeded rng, so replaying the recorded selects against
    # the same seed reproduces the game exactly -- the property undo needs.
    sd = _cglib.BattleStartSeeded(arg, seed, 0)
    Battle.battle_ptr = sd.battlePtr
    if not Battle.battle_ptr:
        return None
    return _get_battle_data()
from cg.api import all_card_data, all_attack, OptionType, SelectContext  # noqa: E402
from battle_log import save_battle, deck_name, deck_path, load_agent  # noqa: E402
import library  # noqa: E402
from library import load_config  # noqa: E402

HOST, PORT = "0.0.0.0", int(os.environ.get("PLAY_PORT", "8000") or 8000)
HUMAN = 0  # human plays this player index; AI plays the other

# The human deck, AI agent and AI deck are chosen in config.json (edited from
# the management page) and re-read at the start of every game, so changing the
# selection and clicking Restart applies it without restarting the server.


def load_deck(name):
    with open(deck_path(name)) as f:
        return [int(l) for l in f if l.strip()]

HEROZ = "https://ptcgvis.heroz.jp"
# How many trailing game steps to send the board. Small = only the most recent
# action animates on each move (raise it to see more of the opponent's turn).
_BOARD_STEPS = 2

# The heroz board (Phaser) loads each card face from /img/<obfuscated-dir>/<cardId>.png.
# Match that shape so we can substitute our own Japanese card images.
_CARD_FACE_RE = re.compile(r"^/img/[A-Za-z0-9_-]+/(\d+)\.png(?:\?.*)?$")


def _card_face_id(path):
    m = _CARD_FACE_RE.match(path)
    return int(m.group(1)) if m else None


_jp_board_ok = None  # None=unprobed, then True/False (cached so we log at most once)


def _jp_board_ready():
    """Whether we can serve local Japanese card faces onto the board. Probes
    PyMuPDF + the image PDF once; if unavailable, disables the swap (falling back
    to heroz art) and prints a single hint instead of erroring per card."""
    global _jp_board_ok
    if _jp_board_ok is None:
        _jp_board_ok = False
        if library.has_card_images():
            try:
                import fitz  # noqa: F401  (PyMuPDF)
                _jp_board_ok = True
            except Exception:
                pass
        if not _jp_board_ok:
            print("[board-jp] Japanese card images disabled -- run with the venv "
                  "python (needs PyMuPDF: pip install PyMuPDF). Using heroz card art.")
    return _jp_board_ok


# heroz renders card faces at this pixel size and draws them at native size, so
# our (higher-res) local images must be scaled to match or they appear oversized.
_BOARD_CARD_W, _BOARD_CARD_H = 396, 552
_board_img_cache = {}


def _board_card_image(cid):
    """Local Japanese card face resized to heroz's card dimensions (cached)."""
    if cid in _board_img_cache:
        return _board_img_cache[cid]
    res = library.card_image(cid)
    out = res
    if res is not None:
        try:
            import io
            from PIL import Image
            im = Image.open(io.BytesIO(res[1])).convert("RGB")
            im = im.resize((_BOARD_CARD_W, _BOARD_CARD_H), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            out = ("image/png", buf.getvalue())
        except Exception as e:
            print("[board-jp] resize failed for", cid, ":", repr(e), "-- serving full size")
    # Cache SUCCESSES only: a transient miss (PDF lock contention, first-hit race)
    # cached as None would leave that card's face broken until the server restarts.
    if out is not None:
        _board_img_cache[cid] = out
    return out
_UA = "Mozilla/5.0 (X11; Linux x86_64) play_server proxy"
_asset_cache = {}          # path -> (content_type, bytes); heroz assets are static
_asset_lock = threading.Lock()


def heroz_get(path):
    """Proxy a GET to heroz (with a small in-memory cache for static assets)."""
    with _asset_lock:
        if path in _asset_cache:
            return _asset_cache[path]
    req = urllib.request.Request(HEROZ + path, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        ctype = r.headers.get("Content-Type", "application/octet-stream")
        body = r.read()
    if not path.startswith("/Visualizer"):
        with _asset_lock:
            _asset_cache[path] = (ctype, body)
    return ctype, body


def heroz_board_html(visualize_json):
    """POST live game data to heroz and return its rendered board HTML.

    We only send heroz the last few steps of the game (each step is a full,
    self-contained board snapshot), so it replays just the recent action instead
    of re-animating the whole game from step 1 on every move.
    """
    try:
        steps = json.loads(visualize_json)
        if isinstance(steps, list) and len(steps) > _BOARD_STEPS:
            visualize_json = json.dumps(steps[-_BOARD_STEPS:])
    except Exception:  # noqa: BLE001  (fall back to the full replay)
        pass
    body = urllib.parse.urlencode({"json": visualize_json}).encode()
    req = urllib.request.Request(
        f"{HEROZ}/Visualizer/Replay/{HUMAN}", data=body,
        headers={"User-Agent": _UA, "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", "replace")
    return html.replace("</body>", _JUMP_TO_END + "</body>")


_JUMP_TO_END = r"""
<script>
// Live play: jump the heroz replay to the LAST step immediately (dragging the
// #slide to its end calls playStop()+updateStep(), so this skips heroz's from-
// step-1 autoplay). Poll fast from t=0 to catch the moment the visualizer wires
// the slider, and hold it at the end against autoplay. Signal the parent when
// we're on the final step so it can reveal the (otherwise flashing) board.
(function(){
  var tries = 0, notified = false;
  function jump(){
    var d = window.replayData;
    var rng = document.getElementById('slide');
    if (!rng || !d || !d.length) return false;
    var n = d.length;
    if (Number(rng.value) !== n){
      rng.value = n;
      rng.dispatchEvent(new Event('input', {bubbles:true}));
    }
    return true;
  }
  function go(){
    tries++;
    var ok = false;
    try{ ok = jump(); }catch(e){}
    if (ok && !notified){
      notified = true;
      try{ parent.postMessage('board-ready', '*'); }catch(e){}
    }
    if (tries < 80) setTimeout(go, 40);   // ~3.2s: boot + hold-at-end vs autoplay
  }
  go();
})();
</script>
"""

# ---- static card / attack metadata -----------------------------------------
CARDS = {c.cardId: c for c in all_card_data()}
ATTACKS = {a.attackId: a for a in all_attack()}

ENERGY_SYMBOL = {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P",
                 6: "F", 7: "D", 8: "M", 9: "N", 10: "*", 11: "TR"}

CONTEXT_TEXT = {
    0: "行動を選んでください",
    1: "バトルポケモンを選んでください（準備）",
    2: "ベンチポケモンを選んでください（準備）",
    3: "バトル場に出すポケモンを選んでください",
    4: "バトル場に置くポケモンを選んでください",
    5: "ベンチに置くポケモンを選んでください",
    7: "手札に加えるカードを選んでください",
    8: "トラッシュするカードを選んでください",
    11: "サイドにするカードを選んでください",
    13: "ダメカンを乗せるポケモンを選んでください",
    15: "ダメージを与えるポケモンを選んでください",
    17: "回復するポケモンを選んでください",
    18: "進化元のポケモンを選んでください",
    19: "進化先を選んでください",
    24: "見るカードを選んでください",
    25: "効果の対象を選んでください",
    34: "効果を発動する順番を選んでください",
    35: "ワザを選んでください",
    38: "引く枚数を選んでください",
    41: "先攻にしますか？",
    42: "引き直し（マリガン）しますか？",
    43: "効果を発動しますか？",
}


def card_name(cid):
    c = CARDS.get(cid)
    return c.name if c else (f"#{cid}" if cid else "?")


_CARDS_JP = None


def card_name_jp(cid):
    """Japanese card name (data/JP_Card_Data.csv) for the action-choice UI.
    Falls back to the English name; the battle log keeps using card_name()."""
    global _CARDS_JP
    if _CARDS_JP is None:
        try:
            _CARDS_JP = {cid_: c["name"] for cid_, c in library.load_cards().items()}
        except Exception:
            _CARDS_JP = {}
    return _CARDS_JP.get(cid) or card_name(cid)


_ATK_JP = None      # {attackId: Japanese move name}
_ABIL_JP = None     # {cardId: [Japanese ability names]}


def _clen(s):
    return 0 if s in ("n/a", "") else len(s)


def _leading_int(s):
    m = re.match(r"\d+", s or "")
    return int(m.group()) if m else None


def _build_jp_moves():
    """Attack-id -> Japanese move name and card-id -> ability names, from
    data/JP_Card_Data.csv. cg.api lists a card's attacks in order but the CSV
    also carries ability ([特性]…) and Terastal rows, so we match each attack to
    a move row greedily by energy-cost length (then damage) rather than by index."""
    global _ATK_JP, _ABIL_JP
    _ATK_JP, _ABIL_JP = {}, {}
    moves = {}  # cid -> [(name, cost_len, damage), ...] (real moves, in order)
    try:
        with open(library.CARD_DATA, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row or not row[0].strip().isdigit():
                    continue
                cid = int(row[0])
                nm = row[13] if len(row) > 13 else ""
                if not nm or nm == "n/a":
                    continue
                if nm.startswith("[特性]"):
                    _ABIL_JP.setdefault(cid, []).append(nm[len("[特性]"):])
                else:
                    moves.setdefault(cid, []).append((nm, _clen(row[14]), _leading_int(row[15])))
    except Exception as e:  # noqa: BLE001
        print("[jp-moves] parse failed:", repr(e))
        return
    for c in CARDS.values():
        cand = list(moves.get(c.cardId, []))
        for aid in (c.attacks or []):
            a = ATTACKS.get(aid)
            if not a:
                continue
            clen, dmg = len(a.energies or []), (a.damage or None)
            pick = next((i for i, (_n, cl, dm) in enumerate(cand)
                         if cl == clen and dm is not None and dmg is not None and dm == dmg), None)
            if pick is None:
                pick = next((i for i, (_n, cl, _d) in enumerate(cand) if cl == clen), None)
            if pick is None and cand:
                pick = 0
            if pick is not None:
                _ATK_JP[aid] = cand.pop(pick)[0]


def attack_name_jp(aid):
    """Japanese move name for an attack id (falls back to the English name)."""
    if _ATK_JP is None:
        _build_jp_moves()
    a = ATTACKS.get(aid)
    return _ATK_JP.get(aid) or (a.name if a else "?")


def ability_name_jp(cid):
    """Japanese ability name of a card if it has exactly one, else None."""
    if _ABIL_JP is None:
        _build_jp_moves()
    names = _ABIL_JP.get(cid) or []
    return names[0] if len(names) == 1 else None


def _area_list(area, st, p):
    if area == 2:
        return p.get("hand")
    if area == 4:
        return p.get("active")
    if area == 5:
        return p.get("bench")
    if area == 3:
        return p.get("discard")
    if area == 6:
        return p.get("prize")
    if area == 7:
        return st.get("stadium")
    if area == 12:
        return st.get("looking")
    return None


def card_id_at(area, index, pidx, st, sel=None):
    """Resolve the card id referenced by (area, index, playerIndex)."""
    if index is None:
        return None
    if area == 1 and sel and sel.get("deck"):  # DECK
        lst = sel["deck"]
        if 0 <= index < len(lst) and lst[index]:
            return lst[index].get("id")
        return None
    if pidx is None:
        pidx = st["yourIndex"]
    try:
        p = st["players"][pidx]
    except (IndexError, KeyError):
        return None
    lst = _area_list(area, st, p)
    if lst and 0 <= index < len(lst) and lst[index]:
        return lst[index].get("id")
    return None


def option_label(o, sel, st):
    """Japanese label for a single action option (the battle log stays English,
    which is built separately in log_line())."""
    t = o["type"]
    pidx = o.get("playerIndex")
    if t == OptionType.PLAY:
        return f"出す：{card_name_jp(card_id_at(2, o.get('index'), st['yourIndex'], st))}"
    if t == OptionType.ATTACH:
        src = card_name_jp(card_id_at(o.get("area"), o.get("index"), pidx, st, sel))
        tgt = card_name_jp(card_id_at(o.get("inPlayArea"), o.get("inPlayIndex"), pidx, st))
        return f"エネつける：{src}  →  {tgt}"
    if t == OptionType.EVOLVE:
        src = card_name_jp(card_id_at(o.get("area"), o.get("index"), pidx, st, sel))
        tgt = card_name_jp(card_id_at(o.get("inPlayArea"), o.get("inPlayIndex"), pidx, st))
        return f"進化：{tgt}  →  {src}"
    if t == OptionType.ABILITY:
        cid = card_id_at(o.get("area"), o.get("index"), pidx, st, sel)
        ab = ability_name_jp(cid)
        return f"特性：{ab}（{card_name_jp(cid)}）" if ab else f"特性：{card_name_jp(cid)}"
    if t == OptionType.DISCARD:
        return f"トラッシュ：{card_name_jp(card_id_at(o.get('area'), o.get('index'), pidx, st, sel))}"
    if t == OptionType.RETREAT:
        return "にげる"
    if t == OptionType.ATTACK:
        aid = o.get("attackId")
        a = ATTACKS.get(aid)
        if a:
            cost = "".join(ENERGY_SYMBOL.get(e, "?") for e in a.energies)
            return f"ワザ：{attack_name_jp(aid)}  （{a.damage} ダメージ [{cost}]）"
        return "ワザ"
    if t == OptionType.END:
        return "ターンを終える"
    if t == OptionType.YES:
        return "はい"
    if t == OptionType.NO:
        return "いいえ"
    if t == OptionType.NUMBER:
        return f"{o.get('number')}"
    if t == OptionType.SKILL:
        return f"スキル：{card_name_jp(o.get('cardId'))}"
    if t == OptionType.SPECIAL_CONDITION:
        names = {0: "どく", 1: "やけど", 2: "ねむり", 3: "まひ", 4: "こんらん"}
        return names.get(o.get("specialConditionType"), "特殊状態")
    if t in (OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD, OptionType.ENERGY):
        cid = card_id_at(o.get("area"), o.get("index"), pidx, st, sel)
        return card_name_jp(cid)
    return OptionType(t).name


def option_card_id(o, sel, st):
    """The card id whose face best represents an option (for a thumbnail), or
    None for option types with no single card (Yes/No/Number/End)."""
    t = o["type"]
    pidx = o.get("playerIndex")
    yi = st["yourIndex"]
    if t == OptionType.PLAY:
        return card_id_at(2, o.get("index"), yi, st)
    if t in (OptionType.ATTACH, OptionType.EVOLVE, OptionType.ABILITY, OptionType.DISCARD,
             OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD, OptionType.ENERGY):
        return card_id_at(o.get("area"), o.get("index"), pidx, st, sel)
    if t in (OptionType.ATTACK, OptionType.RETREAT):
        act = st["players"][yi].get("active") or []
        return act[0]["id"] if act and act[0] else None
    if t == OptionType.SKILL:
        return o.get("cardId")
    return None


def pokemon_view(pk):
    if not pk:
        return None
    return {
        "name": card_name(pk["id"]),
        "hp": pk.get("hp"),
        "maxHp": pk.get("maxHp"),
        "energies": [ENERGY_SYMBOL.get(e, "?") for e in pk.get("energies", [])],
        "tools": [card_name(c["id"]) for c in pk.get("tools", [])],
    }


def player_view(p, reveal_hand):
    active = p.get("active") or []
    return {
        "active": pokemon_view(active[0]) if active and active[0] else None,
        "bench": [pokemon_view(b) for b in p.get("bench", [])],
        "hand": [card_name(c["id"]) for c in (p.get("hand") or [])] if reveal_hand else None,
        "handCount": p.get("handCount"),
        "deckCount": p.get("deckCount"),
        "prizeCount": len(p.get("prize", [])),
        "discardCount": len(p.get("discard", [])),
        "discard": [card_name(c["id"]) for c in p.get("discard", [])],
        "poisoned": p.get("poisoned"), "burned": p.get("burned"),
        "asleep": p.get("asleep"), "paralyzed": p.get("paralyzed"),
        "confused": p.get("confused"),
    }


def log_line(lg):
    LT = lg["type"]
    pi = lg.get("playerIndex")
    who = "You" if pi == HUMAN else "AI"
    if LT == 2:
        return f"--- {who} turn start ---"
    if LT == 3:
        return f"--- {who} turn end ---"
    if LT == 4:
        return f"{who} drew {card_name(lg.get('cardId'))}"
    if LT == 5:
        return "AI drew a card"
    if LT == 10:
        return f"{who} played {card_name(lg.get('cardId'))}"
    if LT == 11:
        return f"{who} attached {card_name(lg.get('cardId'))} to {card_name(lg.get('cardIdTarget'))}"
    if LT == 12:
        return f"{who} evolved {card_name(lg.get('cardIdTarget'))} into {card_name(lg.get('cardId'))}"
    if LT == 15:
        a = ATTACKS.get(lg.get("attackId"))
        return f"{who}'s {card_name(lg.get('cardId'))} used {a.name if a else 'attack'}"
    if LT == 16:
        v = lg.get("value")
        return f"{card_name(lg.get('cardId'))} HP {'+' if v and v > 0 else ''}{v}"
    if LT == 8:
        return f"{who} switched Pokemon"
    if LT == 22:
        return f"Coin: {'heads' if lg.get('head') else 'tails'}"
    if LT == 23:
        r = lg.get("result")
        return f"RESULT: {'draw' if r == 2 else ('You win!' if r == HUMAN else 'AI wins')}"
    if LT in (17, 18, 19, 20, 21):
        names = {17: "poisoned", 18: "burned", 19: "asleep", 20: "paralyzed", 21: "confused"}
        rec = " (recovered)" if lg.get("isRecover") else ""
        return f"{card_name(lg.get('cardId'))} {names[LT]}{rec}"
    return None


# ---- game state -------------------------------------------------------------
class Game:
    def __init__(self):
        self.lock = threading.Lock()
        self.obs = None
        self.human_deck = None
        self.ai_deck = None
        self.saved = False
        self.seed = None           # per-battle seed (None = stock lib, undo off)
        self.history = []          # [("h"|"a", indices), ...] every select this battle
        self._apply_config()

    def _apply_config(self):
        """Read the current deck/agent selection from config.json.

        A deck set to "random" (library.RANDOM_DECK) is resolved to a concrete
        deck here, i.e. re-drawn every battle since start()/reset() call this.
        """
        cfg = load_config()["play"]
        self.human_deck_name = library.resolve_deck(cfg["human_deck"])
        self.ai_deck_name = library.resolve_deck(cfg["ai_deck"])
        # When the AI deck is "random", auto-pick the agent tuned for whichever
        # deck was drawn this battle (agents/<deck>.py). Fall back to the
        # configured agent if that deck has no dedicated agent.
        if cfg["ai_deck"] == library.RANDOM_DECK:
            try:
                self.ai_agent = load_agent(self.ai_deck_name)
                self.ai_agent_name = self.ai_deck_name
                print(f"[agent] random deck {self.ai_deck_name} -> "
                      f"auto agent {self.ai_agent_name}")
            except Exception as e:  # noqa: BLE001
                self.ai_agent_name = cfg["ai_agent"]
                self.ai_agent = load_agent(self.ai_agent_name)
                print(f"[agent] no dedicated agent for {self.ai_deck_name} "
                      f"({e!r}); using {self.ai_agent_name}")
        else:
            self.ai_agent_name = cfg["ai_agent"]
            self.ai_agent = load_agent(self.ai_agent_name)

    def _load_decks(self):
        self.human_deck = load_deck(self.human_deck_name)
        try:
            self.ai_deck = load_deck(self.ai_deck_name)
        except FileNotFoundError:
            print(f"[deck] {deck_path(self.ai_deck_name)} not found; "
                  f"AI reuses {self.human_deck_name}")
            self.ai_deck = self.human_deck
            self.ai_deck_name = self.human_deck_name

    def _battle_decks(self):
        """Return (deck0, deck1) ordered by player index (human plays HUMAN)."""
        decks = [None, None]
        decks[HUMAN] = self.human_deck
        decks[1 - HUMAN] = self.ai_deck
        return decks[0], decks[1]

    def _begin(self):
        """Start a battle: seeded (deterministic, undo-capable) when the seeded
        engine build is present, stock otherwise."""
        self.saved = False
        self.history = []
        self.seed = random.randrange(1, 2**31)
        obs = _battle_start_seeded(*self._battle_decks(), self.seed)
        if obs is None:
            self.seed = None                     # undo unavailable on the stock lib
            obs, _ = battle_start(*self._battle_decks())
        self.obs = obs

    def start(self):
        self._apply_config()  # re-read selection so config changes apply
        self._load_decks()
        self._begin()
        self._advance_ai()

    def reset(self):
        try:
            battle_finish()
        except Exception:
            pass
        self._apply_config()  # re-read selection + deck files so changes apply on Restart
        self._load_decks()
        self._begin()
        self._advance_ai()

    def undo(self):
        """Rewind to just before the human's LAST confirmed action, by replaying
        the recorded selects of this battle against the same seed."""
        if self.seed is None:
            raise ValueError("取り消しはこの環境では使えません（シード付きエンジン未検出）")
        last_h = None
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i][0] == "h":
                last_h = i
                break
        if last_h is None:
            raise ValueError("取り消せる手がまだありません")
        prefix = self.history[:last_h]
        try:
            battle_finish()
        except Exception:
            pass
        obs = _battle_start_seeded(*self._battle_decks(), self.seed)
        if obs is None:
            raise ValueError("リプレイの再開始に失敗しました")
        for _, sel in prefix:
            obs = battle_select(list(sel))
        self.obs = obs
        self.history = prefix
        self.saved = False

    def _advance_ai(self):
        """Let the AI play until it is the human's turn or the game ends."""
        while True:
            st = self.obs["current"]
            if st["result"] >= 0:
                self._save_log()
                return
            if st["yourIndex"] == HUMAN:
                return
            action = self.ai_agent(self.obs)
            self.history.append(("a", list(action)))
            self.obs = battle_select(action)

    def _save_log(self):
        """Save the finished battle to logs/ once (Human vs AI, with deck names)."""
        if self.saved:
            return
        self.saved = True
        players = [None, None]
        players[HUMAN] = {"kind": "human", "deck": deck_name(self.human_deck_name)}
        players[1 - HUMAN] = {"kind": "ai", "agent": self.ai_agent_name,
                              "deck": deck_name(self.ai_deck_name)}
        try:
            from cg.game import visualize_data
            path = save_battle(visualize_data(), players)
            print("saved log:", path)
        except Exception as e:
            print("[log] failed to save:", e)

    def human_select(self, indices):
        sel = self.obs["select"]
        n = len(indices)
        if not (sel["minCount"] <= n <= sel["maxCount"]):
            raise ValueError(f"Choose between {sel['minCount']} and {sel['maxCount']} option(s).")
        if len(set(indices)) != n:
            raise ValueError("Duplicate selection.")
        if any(not (0 <= i < len(sel["option"])) for i in indices):
            raise ValueError("Option index out of range.")
        self.history.append(("h", list(indices)))
        self.obs = battle_select(indices)
        self._advance_ai()

    def state(self):
        st = self.obs["current"]
        over = st["result"] >= 0
        sel = self.obs.get("select")
        out = {
            "over": over,
            "result": st["result"],
            "humanIndex": HUMAN,
            "yourIndex": st["yourIndex"],
            "turn": st["turn"],
            "isHumanTurn": (not over) and st["yourIndex"] == HUMAN,
            "logs": [s for s in (log_line(l) for l in self.obs.get("logs", [])) if s],
            "you": player_view(st["players"][HUMAN], reveal_hand=True),
            "opponent": player_view(st["players"][1 - HUMAN], reveal_hand=False),
            "stadium": card_name(st["stadium"][0]["id"]) if st.get("stadium") else None,
        }
        if not over and sel is not None:
            out["decision"] = {
                "context": sel["context"],
                "text": CONTEXT_TEXT.get(sel["context"], SelectContext(sel["context"]).name),
                "minCount": sel["minCount"],
                "maxCount": sel["maxCount"],
                "options": [{"index": i, "label": option_label(o, sel, st), "type": o["type"],
                             "cardId": option_card_id(o, sel, st)}
                            for i, o in enumerate(sel["option"])],
            }
        return out

    def visualize(self):
        from cg.game import visualize_data
        return visualize_data()


GAME = Game()


# ---- HTTP handler -----------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            if "html" in ctype:
                self.send_header("Cache-Control", "no-store")  # always serve fresh UI
            self.end_headers()
            self.wfile.write(data)
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Client (browser) closed the connection mid-response -- e.g. it
            # cancelled an in-flight card image when the board reloaded. Harmless.
            self.close_connection = True
            return False

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path.startswith("/index"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/manage":
            self._send(200, MANAGE_PAGE, "text/html; charset=utf-8")
        elif path == "/decks":
            self._send(200, DECK_PAGE, "text/html; charset=utf-8")
        elif path == "/api/library":
            self._send(200, json.dumps(library.library()))
        elif path == "/api/cards":
            self._send(200, json.dumps(library.card_list()))
        elif path == "/api/deck":
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            name = (urllib.parse.parse_qs(qs).get("name") or [""])[0]
            self._send(200, json.dumps({"name": name, "ids": library.read_deck(name)}))
        elif path == "/api/card-image":
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            cid = (urllib.parse.parse_qs(qs).get("id") or [""])[0]
            if not cid.isdigit():
                self._send(400, "invalid id", "text/plain")
            else:
                try:
                    res = library.card_image(cid)
                    if res:
                        ctype, body = res
                        self.send_response(200)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Content-Length", str(len(body)))
                        self.send_header("Cache-Control", "max-age=86400")
                        self.end_headers()
                        self.wfile.write(body)
                    else:
                        self._send(404, f"no image for card {cid}", "text/plain")
                except ModuleNotFoundError as e:
                    print("[card-image] PyMuPDF missing:", e)
                    self._send(500, "PyMuPDF not installed in this Python "
                               "(run: pip install PyMuPDF)", "text/plain")
                except FileNotFoundError:
                    print("[card-image] PDF not found:", library.CARD_IMAGE_PDF)
                    self._send(500, "image PDF not found: " + library.CARD_IMAGE_PDF,
                               "text/plain")
                except Exception as e:  # noqa: BLE001
                    print("[card-image] error:", repr(e))
                    self._send(500, "image error: " + str(e), "text/plain")
        elif path == "/board":
            with GAME.lock:
                vis = GAME.visualize()
            try:
                html = heroz_board_html(vis)
                self._send(200, html, "text/html; charset=utf-8")
            except Exception as e:
                self._send(502, f"<p>heroz unavailable: {e}</p>", "text/html; charset=utf-8")
        elif path == "/api/state":
            with GAME.lock:
                self._send(200, json.dumps(GAME.state()))
        elif path == "/api/visualize":
            with GAME.lock:
                self._send(200, GAME.visualize(), "application/json")
        elif _card_face_id(path) is not None and _jp_board_ready():
            # The heroz board loads each card face as /img/<dir>/<cardId>.png.
            # Serve our local Japanese card image instead; fall back to heroz's
            # asset when we have no image for that id.
            cid = _card_face_id(path)
            res = None
            try:
                res = _board_card_image(cid)
            except Exception as e:  # noqa: BLE001  (a real image error, not a disconnect)
                print("[board-jp] card_image failed for", cid, ":", repr(e))
            if res is not None:
                self._send(200, res[1], res[0])  # _send swallows client disconnects
            else:
                try:
                    ctype, body = heroz_get(self.path)
                    self._send(200, body, ctype)
                except Exception:
                    self._send(404, b"")
        else:
            # Proxy everything else (heroz assets: /js, /img, /css, ...).
            try:
                ctype, body = heroz_get(self.path)
                self._send(200, body, ctype)
            except Exception:
                self._send(404, b"")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {}
        if self.path == "/api/select":
            with GAME.lock:
                try:
                    GAME.human_select([int(i) for i in body.get("indices", [])])
                    self._send(200, json.dumps(GAME.state()))
                except Exception as e:
                    self._send(400, json.dumps({"error": str(e)}))
        elif self.path == "/api/reset":
            with GAME.lock:
                GAME.reset()
                self._send(200, json.dumps(GAME.state()))
        elif self.path == "/api/undo":
            with GAME.lock:
                try:
                    GAME.undo()
                    self._send(200, json.dumps(GAME.state()))
                except Exception as e:
                    self._send(400, json.dumps({"error": str(e)}))
        elif self.path == "/api/config":
            try:
                stored = library.save_config(body)
                self._send(200, json.dumps({"config": stored}))
            except ValueError as e:
                self._send(400, json.dumps({"error": str(e)}))
        elif self.path == "/api/submission":
            try:
                result = library.build_submission(body.get("agent"), body.get("deck"))
                self._send(200, json.dumps(result))
            except ValueError as e:
                self._send(400, json.dumps({"error": str(e)}))
            except Exception as e:  # noqa: BLE001
                self._send(500, json.dumps({"error": str(e)}))
        elif self.path == "/api/submit-kaggle":
            try:
                result = library.submit_to_kaggle(
                    body.get("agent"), body.get("deck"), body.get("message"))
                self._send(200 if result.get("submitted") else 502, json.dumps(result))
            except ValueError as e:
                self._send(400, json.dumps({"error": str(e)}))
            except FileNotFoundError:
                self._send(500, json.dumps({"error": "kaggle CLI not found (pip install kaggle)"}))
            except Exception as e:  # noqa: BLE001
                self._send(500, json.dumps({"error": str(e)}))
        elif self.path == "/api/deck":
            try:
                result = library.save_deck(body.get("name"), body.get("ids") or [])
                self._send(200, json.dumps(result))
            except ValueError as e:
                self._send(400, json.dumps({"error": str(e)}))
        elif self.path == "/api/deck-delete":
            try:
                result = library.delete_deck(body.get("name"))
                self._send(200, json.dumps(result))
            except ValueError as e:
                self._send(400, json.dumps({"error": str(e)}))
        else:
            self._send(404, "{}")


PAGE = r"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Human vs AI - Card Battle</title>
<style>
  :root{ color-scheme: dark; }
  *{ box-sizing:border-box; }
  html,body{ margin:0; height:100%; }
  body{ font-family: system-ui,-apple-system,"Segoe UI","Noto Sans JP",sans-serif;
        background:#0d1117; color:#e6edf3; }
  /* Board stays FULL WIDTH and as TALL as possible so the heroz canvas (a fixed
     1400x950 board whose card detail preview sits at the top-right) scales down
     to fit height -> your own board AND the detail preview are visible at once.
     The controls sit in a THIN bottom bar; options are a single horizontally
     scrolling strip so they never steal vertical space from the board. */
  #app{ display:flex; flex-direction:column; width:100%; height:100vh; height:100dvh; }
  /* The heroz board renders at its native 1400x950; we keep the iframe at that
     size (so heroz sees a stable viewport and never scrolls or restarts its
     animation) and scale the whole iframe with CSS transform to fit the area. */
  #boardwrap{ flex:1 1 auto; position:relative; overflow:hidden; min-height:0; background:#8bc677; }
  #board{ position:absolute; top:0; left:0; width:1645px; height:950px; border:0;
          display:block; transform-origin:top left; opacity:0; transition:opacity .18s ease; }
  #side{ flex:0 0 auto; width:100%; background:#0d1117; border-top:2px solid #2b3947;
         display:flex; flex-direction:column; }
  #sidehead{ padding:5px 14px; border-bottom:1px solid #212b36;
             display:flex; align-items:center; gap:8px 14px; flex-wrap:wrap; }
  #decisionText{ font-size:14px; line-height:1.3; flex:1 1 220px; }
  #status,#hint{ font-size:12px; color:#9db2c8; white-space:nowrap; }
  #confirm{ margin-left:auto; }
  /* Option thumbnail size drives the strip height, which sets how much room the
     board gets: SMALLER .optimg -> taller board -> bigger detail preview. */
  #opts{ display:flex; flex-wrap:nowrap; overflow-x:auto; overflow-y:hidden;
         gap:8px; padding:6px 12px; align-items:flex-start; }
  #opts::-webkit-scrollbar{ height:8px; } #opts::-webkit-scrollbar-thumb{ background:#35506b; border-radius:5px; }
  #opts:empty::after{ content:"—"; color:#3a4552; }
  .opt{ flex:0 0 auto; background:#243447; border:1px solid #35506b; color:#e6edf3;
        cursor:pointer; border-radius:9px; padding:5px; width:72px;
        display:flex; flex-direction:column; align-items:center; gap:3px; }
  .opt .optimg{ width:60px; border-radius:4px; display:block; }
  .opt .optlabel{ font-size:10.5px; line-height:1.2; text-align:center; word-break:break-word;
                  display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
  .opt.noimg{ width:auto; min-width:72px; justify-content:center; align-self:stretch;
              padding:9px 13px; font-size:14px; }
  .opt:hover{ border-color:#2d6cdf; }
  .opt:focus-visible{ outline:3px solid #58a6ff; outline-offset:2px; }
  .opt.sel{ background:#234d2a; border-color:#3fb950; box-shadow:0 0 0 2px #3fb950 inset; }
  button{ font:inherit; cursor:pointer; border:none; border-radius:8px;
          padding:9px 15px; background:#2d6cdf; color:#fff; font-size:14px; }
  button.ghost{ background:#30363d; }
  button:disabled{ opacity:.4; cursor:default; }
  button:focus-visible, .mglink:focus-visible{ outline:3px solid #58a6ff; outline-offset:2px; }
  .mglink{ color:#9db2c8; font-size:13px; text-decoration:none; border:1px solid #30363d;
           padding:8px 11px; border-radius:8px; }
  .banner{ font-weight:bold; padding:4px 11px; border-radius:8px; display:inline-block; }
  .win{background:#1f4d2a}.lose{background:#5a2222}.draw{background:#4a4420}
  /* Short screens (tablets): shrink the option cards so the strip stays thin. */
  @media (max-height: 720px){
    .opt{ width:62px; } .opt .optimg{ width:52px; }
    .opt .optlabel{ font-size:10px; -webkit-line-clamp:2; }
  }
</style></head>
<body>
<div id="app">
  <div id="boardwrap"><iframe id="board" src="/board" title="Board"></iframe></div>
  <div id="side">
    <div id="sidehead">
      <b id="decisionText">読み込み中…</b>
      <span id="status"></span>
      <span id="hint"></span>
      <button id="confirm" onclick="confirm_()" disabled>決定</button>
      <button class="ghost" onclick="undo_()" title="直前の自分の決定の直前まで巻き戻します">1手戻す</button>
      <button class="ghost" onclick="reset()">リスタート</button>
      <a class="mglink" href="/manage">⚙ 管理</a>
    </div>
    <div id="opts"></div>
  </div>
</div>
<script>
let S=null, selected=[];
function esc(s){return (s==null?'':(''+s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
// Reload the board but keep it hidden until the injected jump-to-end script
// reports (via postMessage) that it's showing the final step -- so the user sees
// the up-to-date board, not heroz replaying from step 1. Fallback reveal in case
// the signal is missed.
function revealBoard(){ const b=document.getElementById('board'); if(b) b.style.opacity='1'; }
window.addEventListener('message', e=>{ if(e && e.data==='board-ready') revealBoard(); });
function reloadBoard(){
  const b=document.getElementById('board'); if(!b) return;
  b.style.opacity='0';
  b.src='/board?t='+Date.now();
  setTimeout(revealBoard, 1800);
}
// The heroz page is a left control column (0..243) + the game canvas
// (243..1643); a further right panel sits at 1645+. We size the iframe to
// 1645x950 so the WHOLE game canvas (incl. the detail preview at its right edge)
// is inside it, then scale the iframe to fit the wrapper. Board width no longer
// clips the detail.
const BOARD_W=1645, BOARD_H=950;
function fitBoard(){
  const wrap=document.getElementById('boardwrap'), b=document.getElementById('board');
  if(!wrap||!b || !wrap.clientWidth) return;
  const s=Math.min(wrap.clientWidth/BOARD_W, wrap.clientHeight/BOARD_H);
  const x=Math.max(0,(wrap.clientWidth-BOARD_W*s)/2), y=Math.max(0,(wrap.clientHeight-BOARD_H*s)/2);
  b.style.transform='translate('+x+'px,'+y+'px) scale('+s+')';
}
window.addEventListener('resize', fitBoard);
window.addEventListener('load', fitBoard);

function render(){
  let dt=document.getElementById('decisionText'), opt=document.getElementById('opts'),
      st=document.getElementById('status');
  selected=[]; updateHint();
  st.innerHTML='';
  if(S.over){
    let cls=S.result==2?'draw':(S.result==S.humanIndex?'win':'lose');
    let msg=S.result==2?'引き分け':(S.result==S.humanIndex?'あなたの勝ち！🎉':'AIの勝ち');
    st.innerHTML='<span class="banner '+cls+'">'+msg+'</span>';
    dt.textContent='ゲーム終了'; opt.innerHTML=''; return;
  }
  if(!S.decision){ dt.textContent='AIの番を待っています…'; opt.innerHTML=''; return; }
  let d=S.decision;
  st.textContent='ターン '+S.turn;
  let cnt=(d.minCount==d.maxCount?d.maxCount+'枚選択':d.minCount+'〜'+d.maxCount+'枚選択');
  dt.textContent=d.text+'（'+cnt+'）';
  opt.innerHTML='';
  d.options.forEach(o=>{
    let b=document.createElement('button');
    b.className='opt'; b.dataset.idx=o.index; b.title=o.label;
    if(o.cardId){
      let im=document.createElement('img');
      im.className='optimg'; im.loading='lazy'; im.alt='';
      im.src='/api/card-image?id='+o.cardId;
      im.onerror=()=>{ im.remove(); b.classList.add('noimg'); };
      b.appendChild(im);
    } else { b.classList.add('noimg'); }
    let lab=document.createElement('span');
    lab.className='optlabel'; lab.textContent=o.label;
    b.appendChild(lab);
    b.onclick=()=>toggle(o.index,b,d.maxCount);
    opt.appendChild(b);
  });
}
function toggle(idx,btn,maxCount){
  let pos=selected.indexOf(idx);
  if(pos>=0){ selected.splice(pos,1); btn.classList.remove('sel'); }
  else{
    if(maxCount==1){ selected=[]; document.querySelectorAll('.opt.sel').forEach(e=>e.classList.remove('sel')); }
    if(selected.length>=maxCount) return;
    selected.push(idx); btn.classList.add('sel');
  }
  updateHint();
}
function updateHint(){
  let d=S&&S.decision;
  let ok=d && selected.length>=d.minCount && selected.length<=d.maxCount;
  document.getElementById('confirm').disabled=!ok;
  document.getElementById('hint').textContent=d?(selected.length+'枚選択中'):'';
}
async function load(){ S=await (await fetch('/api/state')).json(); render(); }
async function confirm_(){
  let r=await fetch('/api/select',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({indices:selected})});
  let j=await r.json();
  if(j.error){
    alert(j.error);
    // Resync: a stale page (e.g. after a server restart) holds a menu the server
    // no longer has -- refetch the live state so the next click is valid.
    await load(); reloadBoard();
    return;
  }
  S=j; render(); reloadBoard();
}
async function reset(){
  S=await (await fetch('/api/reset',{method:'POST'})).json(); render(); reloadBoard();
}
async function undo_(){
  let r=await fetch('/api/undo',{method:'POST'});
  let j=await r.json();
  if(j.error){ alert(j.error); await load(); reloadBoard(); return; }
  S=j; render(); reloadBoard();
}
load(); fitBoard(); setTimeout(fitBoard, 200); setTimeout(revealBoard, 2000);
</script>
</body></html>"""


# ---- management page (decks & agents) ---------------------------------------
MANAGE_PAGE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deck &amp; Agent Manager</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         margin: 0; background:#f4f5f7; color:#1c1e21; }
  header { background:#2b3a55; color:#fff; padding:14px 22px;
           display:flex; align-items:center; justify-content:space-between; }
  header h1 { margin:0; font-size:18px; }
  header p { margin:4px 0 0; font-size:12px; opacity:.8; }
  header a { color:#cfe0ff; font-size:13px; text-decoration:none;
             border:1px solid #4a5a7a; padding:7px 12px; border-radius:7px; }
  header a:hover { background:#33456a; }
  main { max-width:980px; margin:0 auto; padding:20px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:760px){ .grid{ grid-template-columns:1fr; } }
  .card { background:#fff; border:1px solid #dfe1e6; border-radius:10px;
          padding:16px; margin-bottom:16px; }
  .card h2 { margin:0 0 10px; font-size:15px; }
  .list { max-height:240px; overflow:auto; border:1px solid #eee; border-radius:8px; }
  .item { display:flex; justify-content:space-between; gap:8px;
          padding:7px 10px; border-bottom:1px solid #f0f0f0; font-size:13px; }
  .item:last-child { border-bottom:0; }
  .item .name { font-weight:600; }
  .item .meta { color:#6b7280; font-size:12px; }
  .bad { color:#c0392b; }
  .row { display:flex; align-items:center; gap:8px; margin:8px 0; }
  .row label { width:120px; font-size:13px; color:#374151; }
  select { flex:1; padding:6px 8px; border:1px solid #cbd2d9; border-radius:6px;
           background:#fff; color:#1c1e21; font-size:13px; }
  select option { background:#fff; color:#1c1e21; }
  .vs { text-align:center; color:#9ca3af; font-size:12px; margin:6px 0; }
  button { background:#2b6cb0; color:#fff; border:0; border-radius:7px;
           padding:9px 16px; font-size:14px; cursor:pointer; }
  button:hover { background:#2c5282; }
  button.secondary { background:#e2e8f0; color:#1a202c; }
  button.secondary:hover { background:#cbd5e0; }
  .actions { display:flex; gap:10px; align-items:center; margin-top:8px; }
  #status { font-size:13px; min-height:18px; }
  .ok { color:#2f855a; } .err { color:#c0392b; }
  code { background:#edf2f7; padding:1px 5px; border-radius:4px; font-size:12px; }
  .note { font-size:12px; color:#6b7280; margin-top:6px; }
</style>
</head>
<body>
<header>
  <div>
    <h1>Deck &amp; Agent Manager</h1>
    <p>Pick the decks and agents used by <code>run.py</code> and <code>play_server.py</code>. Saved to <code>config.json</code>.</p>
  </div>
  <div style="display:flex;gap:8px">
    <a href="/decks">🃏 Build deck</a>
    <a href="/">▶ Play (Human vs AI)</a>
  </div>
</header>
<main>
  <div class="grid">
    <div class="card">
      <h2>Decks <span class="meta" id="deckcount"></span></h2>
      <div class="list" id="decklist"></div>
    </div>
    <div class="card">
      <h2>Agents <span class="meta" id="agentcount"></span></h2>
      <div class="list" id="agentlist"></div>
    </div>
  </div>

  <div class="card">
    <h2>run.py &mdash; AI vs AI</h2>
    <div class="row"><label>Player 0 agent</label><select id="r0a"></select></div>
    <div class="row"><label>Player 0 deck</label><select id="r0d"></select></div>
    <div class="vs">vs</div>
    <div class="row"><label>Player 1 agent</label><select id="r1a"></select></div>
    <div class="row"><label>Player 1 deck</label><select id="r1d"></select></div>
  </div>

  <div class="card">
    <h2>play_server.py &mdash; Human vs AI</h2>
    <div class="row"><label>Human deck</label><select id="phd"></select></div>
    <div class="vs">vs</div>
    <div class="row"><label>AI agent</label><select id="paa"></select>
      <span id="paahint" style="font-size:12px;color:#6b7280;margin-left:8px"></span></div>
    <div class="row"><label>AI deck</label><select id="pad"></select></div>
    <div class="note">After saving, open the Play page and click <b>Restart</b> to apply the new selection (no server restart needed).</div>
  </div>

  <div class="card">
    <h2>Submission &mdash; build Kaggle package</h2>
    <div class="row"><label>Agent</label><select id="sba"></select></div>
    <div class="row"><label>Deck</label><select id="sbd"></select></div>
    <div class="row"><label>Message</label><input id="sbm" type="text" placeholder="(optional) submission note"
         style="flex:1;padding:6px 8px;border:1px solid #cbd2d9;border-radius:6px;font-size:13px"></div>
    <div class="actions">
      <button onclick="makeSubmission()">Create file only</button>
      <button onclick="submitKaggle()" style="background:#b7791f">Submit to Kaggle &#9650;</button>
      <span id="substatus"></span>
    </div>
    <div class="note">Bundles <code>main.py</code> (the agent), <code>deck.csv</code> and the <code>cg/</code> library into <code>submissions/&lt;agent&gt;-&lt;deck&gt;.tar.gz</code>.
      <b>Create file only</b> just writes the tarball; <b>Submit to Kaggle</b> also uploads it via the kaggle CLI (uses a daily submission slot &mdash; you'll be asked to confirm).</div>
    <pre id="subresult" style="display:none;background:#0d1117;color:#e6edf3;padding:10px;border-radius:8px;font-size:12px;overflow:auto;margin-top:10px"></pre>
  </div>

  <div class="card">
    <div class="actions">
      <button onclick="save()">Save selection</button>
      <button class="secondary" onclick="load()">Reload</button>
      <span id="status"></span>
    </div>
  </div>
</main>

<script>
let LIB = {decks:[], agents:[], config:null};
const RANDOM_DECK = "__random__";

function opts(sel, names, value, withRandom){
  sel.innerHTML = "";
  if (withRandom){
    const o = document.createElement("option");
    o.value = RANDOM_DECK; o.textContent = "🎲 ランダム (random)";
    if (value === RANDOM_DECK) o.selected = true;
    sel.appendChild(o);
  }
  for (const n of names){
    const o = document.createElement("option");
    o.value = n; o.textContent = n;
    if (n === value) o.selected = true;
    sel.appendChild(o);
  }
}
// When the AI deck is "random", the server auto-picks the agent matching each
// randomly drawn deck (agents/<deck>.py), so the AI-agent choice is ignored.
function syncAiAuto(){
  const auto = pad.value === RANDOM_DECK;
  paa.disabled = auto;
  paa.style.opacity = auto ? 0.5 : 1;
  document.getElementById("paahint").textContent =
    auto ? "自動: 引いたデッキ専用エージェント" : "";
}
function renderList(el, items, fmt){
  el.innerHTML = "";
  if (!items.length){ el.innerHTML = '<div class="item meta">(none)</div>'; return; }
  for (const it of items){
    const d = document.createElement("div");
    d.className = "item";
    d.innerHTML = fmt(it);
    el.appendChild(d);
  }
}
async function load(){
  const r = await fetch("/api/library");
  LIB = await r.json();
  const deckNames = LIB.decks.map(d => d.name);
  const agentNames = LIB.agents.map(a => a.name);

  document.getElementById("deckcount").textContent = "(" + LIB.decks.length + ")";
  document.getElementById("agentcount").textContent = "(" + LIB.agents.length + ")";
  renderList(document.getElementById("decklist"), LIB.decks, d =>
    '<span class="name">'+d.name+'</span>' +
    '<span class="meta'+(d.ok?'':' bad')+'">'+(d.ok? d.count+' cards' : 'unreadable')+'</span>');
  renderList(document.getElementById("agentlist"), LIB.agents, a =>
    '<span class="name">'+a.name+'</span>' +
    '<span class="meta">'+(a.doc||'')+'</span>');

  const c = LIB.config;
  opts(r0a, agentNames, c.run.player0.agent);
  opts(r0d, deckNames,  c.run.player0.deck, true);
  opts(r1a, agentNames, c.run.player1.agent);
  opts(r1d, deckNames,  c.run.player1.deck, true);
  opts(phd, deckNames,  c.play.human_deck, true);
  opts(paa, agentNames, c.play.ai_agent);
  opts(pad, deckNames,  c.play.ai_deck, true);
  pad.onchange = syncAiAuto;
  syncAiAuto();
  opts(sba, agentNames, c.submit.agent);
  opts(sbd, deckNames,  c.submit.deck);
  setStatus("Loaded.", "ok");
}
function setStatus(msg, cls){
  const s = document.getElementById("status");
  s.textContent = msg; s.className = cls || "";
}
async function save(){
  const cfg = {
    run: {
      player0: {agent: r0a.value, deck: r0d.value},
      player1: {agent: r1a.value, deck: r1d.value},
    },
    play: {
      human_deck: phd.value, ai_agent: paa.value, ai_deck: pad.value,
    },
    submit: { agent: sba.value, deck: sbd.value },
  };
  setStatus("Saving...");
  const r = await fetch("/api/config", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify(cfg)});
  const j = await r.json();
  if (r.ok){ LIB.config = j.config; setStatus("Saved to config.json.", "ok"); }
  else setStatus("Error: " + (j.error||"save failed"), "err");
}
async function makeSubmission(){
  const sub = document.getElementById("substatus");
  const out = document.getElementById("subresult");
  sub.textContent = "Building..."; sub.className = "";
  const r = await fetch("/api/submission", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({agent: sba.value, deck: sbd.value})});
  const j = await r.json();
  if (!r.ok){
    sub.textContent = "Error: " + (j.error||"build failed"); sub.className = "err";
    out.style.display = "none";
    return;
  }
  const kb = (j.bytes/1024).toFixed(1);
  sub.textContent = "Built " + j.tar + " (" + kb + " KB)."; sub.className = "ok";
  out.style.display = "block";
  out.textContent =
    "tar:   " + j.tar_abspath + "\n" +
    "size:  " + kb + " KB\n" +
    "files:\n  " + j.files.join("\n  ");
}
async function submitKaggle(){
  const sub = document.getElementById("substatus");
  const out = document.getElementById("subresult");
  if (!confirm("Submit agent '"+sba.value+"' with deck '"+sbd.value+
               "' to Kaggle?\nThis uploads to the competition and uses a daily submission slot.")) return;
  sub.textContent = "Submitting to Kaggle... (uploading)"; sub.className = "";
  const r = await fetch("/api/submit-kaggle", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({agent: sba.value, deck: sbd.value, message: document.getElementById("sbm").value})});
  const j = await r.json();
  out.style.display = "block";
  if (j.submitted){
    sub.textContent = "Submitted to " + j.competition + "."; sub.className = "ok";
  } else {
    sub.textContent = "Submission failed (see output)."; sub.className = "err";
  }
  out.textContent =
    "competition: " + (j.competition||"") + "\n" +
    "message:     " + (j.message||"") + "\n" +
    "tar:         " + (j.tar_abspath||"") + "\n" +
    "returncode:  " + (j.returncode!=null? j.returncode : "") + "\n\n" +
    (j.output || j.error || "");
}
load();
</script>
</body>
</html>"""


# ---- deck builder page ------------------------------------------------------
DECK_PAGE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deck Builder</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         margin:0; background:#f4f5f7; color:#1c1e21; }
  header { background:#2b3a55; color:#fff; padding:12px 20px;
           display:flex; align-items:center; justify-content:space-between; }
  header h1 { margin:0; font-size:17px; }
  header a { color:#cfe0ff; font-size:13px; text-decoration:none;
             border:1px solid #4a5a7a; padding:6px 11px; border-radius:7px; margin-left:8px; }
  header a:hover { background:#33456a; }
  main { display:grid; grid-template-columns:1fr 380px; gap:14px; padding:14px;
         max-width:1200px; margin:0 auto; align-items:start; }
  @media (max-width:820px){ main{ grid-template-columns:1fr; } }
  .card { background:#fff; border:1px solid #dfe1e6; border-radius:10px; padding:12px; }
  .card h2 { margin:0 0 8px; font-size:14px; }
  .controls { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
  input, select { padding:6px 8px; border:1px solid #cbd2d9; border-radius:6px; font-size:13px; }
  #search { flex:1; min-width:140px; }
  .numf { font-size:12px; color:#374151; display:flex; align-items:center; gap:4px; }
  .numf input { width:62px; }
  .crow .nm { cursor:pointer; }
  .crow .nm:hover { color:#2b6cb0; text-decoration:underline; }
  #imgmodal { display:none; position:fixed; inset:0; background:rgba(0,0,0,.72);
              align-items:center; justify-content:center; z-index:50; cursor:zoom-out; }
  #imgmodal img { max-width:92vw; max-height:84vh; border-radius:10px;
                  box-shadow:0 10px 40px rgba(0,0,0,.5); background:#fff; }
  #imgmodal .imgbox { display:flex; flex-direction:column; align-items:center; gap:10px; }
  #imgcap { color:#fff; font-size:13px; max-width:92vw; text-align:center; }
  .list { max-height:62vh; overflow:auto; border:1px solid #eee; border-radius:8px; }
  .crow { display:flex; align-items:center; gap:8px; padding:6px 9px;
          border-bottom:1px solid #f1f1f1; font-size:13px; }
  .crow:last-child { border-bottom:0; }
  .crow .nm { flex:1; font-weight:600; }
  .crow .meta { color:#6b7280; font-size:11px; white-space:nowrap; }
  .crow .id { color:#9aa3af; font-size:11px; width:46px; }
  button { border:0; border-radius:6px; cursor:pointer; font-size:13px; }
  .add { background:#2b6cb0; color:#fff; padding:4px 10px; }
  .add:hover { background:#2c5282; }
  .add:disabled { background:#cbd5e0; cursor:default; }
  .deckhead { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
  .total { font-weight:700; font-size:15px; }
  .total.ok { color:#2f855a; } .total.bad { color:#c0392b; }
  .drow { display:flex; align-items:center; gap:6px; padding:5px 8px;
          border-bottom:1px solid #f1f1f1; font-size:13px; }
  .drow .nm { flex:1; }
  .cnt { min-width:20px; text-align:center; font-weight:700; }
  .pm { background:#e2e8f0; color:#1a202c; width:24px; height:24px; line-height:1; }
  .pm:hover { background:#cbd5e0; }
  .save { background:#2f855a; color:#fff; padding:8px 14px; }
  .save:hover { background:#276749; }
  .del { background:#e2e8f0; color:#c0392b; padding:6px 10px; }
  .del:hover { background:#f6d3d0; }
  .note { font-size:12px; color:#6b7280; margin-top:6px; }
  #status { font-size:13px; min-height:18px; margin-top:6px; }
  .ok { color:#2f855a; } .err { color:#c0392b; }
</style>
</head>
<body>
<header>
  <h1>🃏 Deck Builder</h1>
  <div>
    <a href="/manage">⚙ Manage</a>
    <a href="/">▶ Play</a>
  </div>
</header>
<main>
  <div class="card">
    <h2>Cards <span class="meta" id="shown" style="color:#6b7280;font-weight:400"></span></h2>
    <div class="controls">
      <input id="search" type="text" placeholder="名前・進化前で検索 (name / pre-evolution)" oninput="renderCatalog()">
      <select id="kind" onchange="renderCatalog()"><option value="">すべての種類</option></select>
      <select id="ptype" onchange="renderCatalog()"><option value="">すべてのタイプ</option></select>
    </div>
    <div class="controls">
      <label class="numf">HP ≥ <input id="hpmin" type="number" min="0" step="10" oninput="renderCatalog()"></label>
      <label class="numf">最大ダメージ ≥ <input id="dmgmin" type="number" min="0" step="10" oninput="renderCatalog()"></label>
      <label class="numf">最小エネルギー ≤ <input id="costmax" type="number" min="0" step="1" oninput="renderCatalog()"></label>
    </div>
    <div class="list" id="catalog"></div>
  </div>

  <div class="card">
    <h2>Deck</h2>
    <div class="controls">
      <input id="dname" type="text" placeholder="deck name (A-Z 0-9 _ -)" style="flex:1">
      <select id="loadsel" onchange="loadDeck(this.value)"><option value="">load existing…</option></select>
      <button class="del" onclick="deleteDeck()" title="選択中のデッキを削除">🗑 削除</button>
    </div>
    <div class="deckhead">
      <span class="total" id="total">0 / 60</span>
      <span style="flex:1"></span>
      <button class="save" onclick="saveDeck()">Save deck</button>
    </div>
    <div class="list" id="deck"></div>
    <div class="note">60 cards, max 4 copies per card (basic energy unlimited).</div>
    <div id="status"></div>
  </div>
</main>
<div id="imgmodal" onclick="this.style.display='none'">
  <div class="imgbox"><img id="imgmodalimg" alt=""><div id="imgcap"></div></div>
</div>

<script>
let CARDS = [], BYID = {}, deck = new Map();  // id -> count

function isBasicEnergy(c){ return c && c.kind === "基本エネルギー"; }
function total(){ let t=0; for (const n of deck.values()) t+=n; return t; }

async function init(){
  CARDS = await (await fetch("/api/cards")).json();
  BYID = {}; const kinds = new Set();
  for (const c of CARDS){ BYID[c.id]=c; if (c.kind) kinds.add(c.kind); }
  const ksel = document.getElementById("kind");
  for (const k of [...kinds].sort()){
    const o=document.createElement("option"); o.value=k; o.textContent=k; ksel.appendChild(o);
  }
  // Pokémon types in the canonical TCG order (skip energy/trainer "n/a" etc.)
  const TYPE_ORDER = ["草","炎","水","雷","超","闘","悪","鋼","竜","無"];
  const types = new Set();
  for (const c of CARDS){
    if (c.kind && c.kind.startsWith("ポケモン") && c.type && c.type !== "n/a") types.add(c.type);
  }
  const ordered = TYPE_ORDER.filter(t => types.has(t))
                            .concat([...types].filter(t => !TYPE_ORDER.includes(t)).sort());
  const tsel = document.getElementById("ptype");
  for (const t of ordered){
    const o=document.createElement("option"); o.value=t; o.textContent=t; tsel.appendChild(o);
  }
  await refreshDeckList();
  renderCatalog(); renderDeck();
}
async function refreshDeckList(){
  const lib = await (await fetch("/api/library")).json();
  const ls = document.getElementById("loadsel");
  ls.innerHTML = '<option value="">load existing…</option>';
  for (const d of lib.decks){
    const o=document.createElement("option"); o.value=d.name; o.textContent=d.name+" ("+d.count+")"; ls.appendChild(o);
  }
}
function renderCatalog(){
  const q = document.getElementById("search").value.trim().toLowerCase();
  const k = document.getElementById("kind").value;
  const ty = document.getElementById("ptype").value;
  const hpMin   = parseInt(document.getElementById("hpmin").value, 10);
  const dmgMin  = parseInt(document.getElementById("dmgmin").value, 10);
  const costMax = parseInt(document.getElementById("costmax").value, 10);
  const box = document.getElementById("catalog");
  box.innerHTML = "";
  let shown=0, matched=0; const CAP=300;
  for (const c of CARDS){
    if (k && c.kind !== k) continue;
    if (ty && c.type !== ty) continue;
    if (!isNaN(hpMin)   && !(c.hp != null && c.hp >= hpMin)) continue;
    if (!isNaN(dmgMin)  && !(c.maxDamage != null && c.maxDamage >= dmgMin)) continue;
    if (!isNaN(costMax) && !(c.minCost != null && c.minCost <= costMax)) continue;
    if (q && !(c.name.toLowerCase().includes(q) || String(c.id)===q
               || (c.prevNames||[]).some(n => n.toLowerCase().includes(q)))) continue;
    matched++;
    if (shown >= CAP) continue;
    shown++;
    const have = deck.get(c.id)||0;
    const cap = isBasicEnergy(c) ? Infinity : 4;
    const meta = [c.kind,
                  c.type && c.type!=="n/a" ? c.type : "",
                  c.hp != null ? "HP"+c.hp : "",
                  c.maxDamage != null ? "最大"+c.maxDamage : "",
                  c.minCost != null ? "⚡"+c.minCost : ""].filter(Boolean).join(" · ");
    const d = document.createElement("div"); d.className="crow";
    d.innerHTML = '<span class="id">#'+c.id+'</span>'+
      '<span class="nm"></span><span class="meta">'+meta+'</span>'+
      '<button class="add"'+(have>=cap?' disabled':'')+'>+'+(have?' ('+have+')':'')+'</button>';
    const nm = d.querySelector(".nm");
    nm.textContent = c.name;
    nm.title = "クリックで画像表示";
    nm.onclick = ()=>showImage(c.id, c.name);
    if (c.prevNames && c.prevNames.length) d.title = "進化前: " + c.prevNames.join(" ← ");
    d.querySelector(".add").onclick = ()=>addCard(c.id);
    box.appendChild(d);
  }
  document.getElementById("shown").textContent =
    matched ? ("showing "+Math.min(matched,CAP)+" / "+matched) : "no match";
}
function addCard(id){
  const c = BYID[id]; const have = deck.get(id)||0;
  const cap = isBasicEnergy(c) ? Infinity : 4;
  if (have >= cap){ return; }
  if (total() >= 60){ setStatus("Deck already has 60 cards.", "err"); return; }
  deck.set(id, have+1); renderDeck(); renderCatalog();
}
function removeCard(id){
  const have = deck.get(id)||0;
  if (have<=1) deck.delete(id); else deck.set(id, have-1);
  renderDeck(); renderCatalog();
}
function renderDeck(){
  const box = document.getElementById("deck"); box.innerHTML="";
  const ids = [...deck.keys()].sort((a,b)=>a-b);
  for (const id of ids){
    const c = BYID[id]; const n = deck.get(id);
    const d = document.createElement("div"); d.className="drow";
    d.innerHTML = '<span class="id" style="color:#9aa3af;font-size:11px;width:46px">#'+id+'</span>'+
      '<span class="nm"></span>'+
      '<button class="pm">−</button><span class="cnt">'+n+'</span><button class="pm add2">＋</button>';
    d.querySelector(".nm").textContent = c ? c.name : ("id "+id);
    const btns = d.querySelectorAll(".pm");
    btns[0].onclick = ()=>removeCard(id);
    btns[1].onclick = ()=>addCard(id);
    box.appendChild(d);
  }
  const t = total(); const el = document.getElementById("total");
  el.textContent = t+" / 60"; el.className = "total "+(t===60?"ok":"bad");
}
function setStatus(msg, cls){ const s=document.getElementById("status"); s.textContent=msg; s.className=cls||""; }
async function showImage(id, name){
  const m=document.getElementById("imgmodal"), im=document.getElementById("imgmodalimg"),
        cap=document.getElementById("imgcap");
  im.removeAttribute("src"); im.alt = name || ("id "+id);
  cap.textContent = (name||("id "+id)) + " を読み込み中…";
  m.style.display = "flex";
  try{
    const r = await fetch("/api/card-image?id=" + id);
    if(!r.ok){ cap.textContent = (name||("id "+id)) + ": " + (await r.text()); return; }
    const url = URL.createObjectURL(await r.blob());
    im.onload = ()=>URL.revokeObjectURL(url);
    im.src = url;
    cap.textContent = (name||("id "+id)) + "  (#" + id + ")";
  }catch(e){ cap.textContent = "読み込みエラー: " + e; }
}
async function loadDeck(name){
  if (!name) return;
  const j = await (await fetch("/api/deck?name="+encodeURIComponent(name))).json();
  deck = new Map();
  for (const id of j.ids) deck.set(id, (deck.get(id)||0)+1);
  document.getElementById("dname").value = name;
  renderDeck(); renderCatalog(); setStatus("Loaded deck '"+name+"'.", "ok");
}
async function deleteDeck(){
  const name = document.getElementById("loadsel").value || document.getElementById("dname").value.trim();
  if (!name){ setStatus("削除するデッキを選んでください（load existing… から選択）。", "err"); return; }
  if (!confirm("デッキ '"+name+"' を削除しますか？ この操作は元に戻せません。")) return;
  const r = await fetch("/api/deck-delete", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({name})});
  const j = await r.json();
  if (!r.ok){ setStatus("Error: "+(j.error||"delete failed"), "err"); return; }
  await refreshDeckList();
  document.getElementById("loadsel").value = "";
  let msg = "デッキ '"+name+"' を削除しました。";
  if (j.repointed && j.repointed.length)
    msg += " 選択設定 ("+j.repointed.join(", ")+") を '"+j.fallback+"' に変更しました。";
  setStatus(msg, "ok");
}
async function saveDeck(){
  const name = document.getElementById("dname").value.trim();
  const ids = [];
  for (const [id,n] of deck) for (let i=0;i<n;i++) ids.push(id);
  setStatus("Saving…");
  const r = await fetch("/api/deck", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({name, ids})});
  const j = await r.json();
  if (r.ok) setStatus("Saved "+j.path+" ("+j.count+" cards).", "ok");
  else setStatus("Error: "+(j.error||"save failed"), "err");
}
init();
</script>
</body>
</html>"""


def main():
    GAME.start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Human vs AI server running:  http://localhost:{PORT}/")
    print(f"Deck & Agent manager:        http://localhost:{PORT}/manage")
    print(f"Deck builder:                http://localhost:{PORT}/decks")
    try:
        import fitz  # noqa: F401
        imgs = "ON" if library.has_card_images() else f"OFF (missing {library.CARD_IMAGE_PDF})"
    except ImportError:
        imgs = "OFF (PyMuPDF not installed in this Python: pip install PyMuPDF)"
    print(f"Card images:                 {imgs}")
    print("(You are Player 0. Click options to make your move. Ctrl+C to stop.)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            battle_finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
