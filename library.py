"""Shared "library" of decks and agents, plus the selection config.

- decks live in decks/<name>.csv (see battle_log.deck_path / deck_name)
- agents live in agents/<name>.py, each defining agent(obs_dict)
- config.json records which deck/agent run.py and play_server.py should use;
  it is edited from the management web page (manage_server.py) and read back
  by run.py and play_server.py.
"""

import ast
import copy
import csv
import json
import os
import random
import re
import shutil
import subprocess
import tarfile
import threading
import time
from collections import Counter

from battle_log import DECKS_DIR, AGENTS_DIR, deck_path

CONFIG_PATH = os.environ.get("PLAY_CONFIG", "config.json")
SUBMISSIONS_DIR = "submissions"
COMPETITION = "pokemon-tcg-ai-battle"  # Kaggle competition slug for submission
CARD_DATA = os.path.join("data", "JP_Card_Data.csv")  # card catalog (Japanese)
CARD_IMAGE_PDF = os.path.join("data", "Card_ID List_JP.pdf")  # card face images
DECK_SIZE = 60
MAX_COPIES = 4  # per card, except basic energy (基本エネルギー)
_DECK_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Sentinel deck selection: pick a random existing deck fresh every battle.
# Valid for the run.py and play (human/AI) deck slots, but not for submissions.
RANDOM_DECK = "__random__"

# Where to find the cg library to bundle into a submission. The downloaded
# sample submission ships both libcg.so (Linux) and cg.dll (Windows); cg-lib is
# the fallback the local servers already use.
_CG_SOURCES = [
    os.path.join("data", "sample_submission", "cg"),
    os.path.join("cg-lib", "cg"),
]

# Selections used by run.py (AI vs AI), play_server.py (Human vs AI), and the
# submission builder.
DEFAULT_CONFIG = {
    "run": {
        "player0": {"agent": "agent", "deck": "mega_abomasnow_sample"},
        "player1": {"agent": "agent", "deck": "mega_abomasnow_sample"},
    },
    "play": {
        "human_deck": "mega_abomasnow_sample",
        "ai_agent": "agent",
        "ai_deck": "deck_ai",
    },
    "submit": {
        "agent": "agent",
        "deck": "mega_abomasnow_sample",
    },
}


# ---- listing ---------------------------------------------------------------
def list_decks():
    """Sorted names of decks under decks/ (filename without .csv)."""
    if not os.path.isdir(DECKS_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(DECKS_DIR) if f.endswith(".csv"))


def list_agents():
    """Sorted names of agents under agents/ (module name, no .py).

    Skips dunder/private files such as __init__.py.
    """
    if not os.path.isdir(AGENTS_DIR):
        return []
    return sorted(f[:-3] for f in os.listdir(AGENTS_DIR)
                  if f.endswith(".py") and not f.startswith("_"))


def deck_info(name):
    """{name, count, ok} for a deck without trusting the count blindly."""
    try:
        with open(deck_path(name)) as f:
            ids = [ln for ln in f if ln.strip()]
        return {"name": name, "count": len(ids), "ok": True}
    except OSError:
        return {"name": name, "count": 0, "ok": False}


def agent_info(name):
    """{name, doc} for an agent. doc is the module docstring's first line.

    Parsed with ast (no import), so listing never runs agent code.
    """
    path = os.path.join(AGENTS_DIR, name + ".py")
    doc = ""
    try:
        with open(path) as f:
            tree = ast.parse(f.read())
        doc = (ast.get_docstring(tree) or "").strip()
    except (OSError, SyntaxError):
        pass
    return {"name": name, "doc": doc.split("\n")[0] if doc else ""}


def library():
    """Everything the management page needs in one call."""
    return {
        "decks": [deck_info(n) for n in list_decks()],
        "agents": [agent_info(n) for n in list_agents()],
        "config": load_config(),
    }


# ---- config read / write ---------------------------------------------------
def _merge(base, override):
    """Recursively overlay override onto a copy of base (dicts only)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config():
    """Return the saved config, filled in with defaults for missing keys."""
    user = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                user = json.load(f)
        except (OSError, ValueError):
            user = {}
    return _merge(DEFAULT_CONFIG, user)


def save_config(cfg):
    """Validate selections against the library, then write config.json.

    Unknown decks/agents are rejected so the page can't point run.py /
    play_server.py at a file that does not exist. Returns the stored config.
    """
    merged = _merge(load_config(), cfg)
    decks = set(list_decks())
    agents = set(list_agents())

    def check_deck(d, allow_random=True):
        if allow_random and d == RANDOM_DECK:
            return
        if d not in decks:
            raise ValueError(f"unknown deck: {d!r}")

    def check_agent(a):
        if a not in agents:
            raise ValueError(f"unknown agent: {a!r}")

    for side in ("player0", "player1"):
        check_agent(merged["run"][side]["agent"])
        check_deck(merged["run"][side]["deck"])
    check_deck(merged["play"]["human_deck"])
    check_agent(merged["play"]["ai_agent"])
    check_deck(merged["play"]["ai_deck"])
    check_agent(merged["submit"]["agent"])
    check_deck(merged["submit"]["deck"], allow_random=False)  # submission needs a real deck

    with open(CONFIG_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    return merged


# ---- submission packaging --------------------------------------------------
def _cg_source():
    for p in _CG_SOURCES:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("cg library not found in: " + ", ".join(_CG_SOURCES))


# Match `from agents.<mod> import ...` in BOTH single-line and parenthesized
# multi-line forms (policies.py uses `from agents._engine import (\n  ...\n)`).
# `\([^)]*\)` spans newlines to swallow the whole parenthesized block; otherwise
# `[^\n]*` takes a single-line import. A single-line strip left the continuation
# lines orphaned -> IndentationError in the submitted main.py.
_ENGINE_IMPORT_RE = re.compile(
    r"^[ \t]*from\s+agents\._engine\s+import\s+(?:\([^)]*\)|[^\n]*)", re.M)



def _agent_main_source(agent):
    """Build a self-contained main.py for a Kaggle submission.

    A legacy agent imports the shared engine (``from agents._engine import ...``),
    which does not exist in the submission (only main.py + deck.csv + cg/ are bundled),
    so inline that source above the agent's own code and drop the import. Self-contained
    agents (agents/agent.py, the random sample) are used verbatim. engine_v2 agents do
    NOT come through here at all -- build_submission refuses them and points at
    tools/build_engine_v2_submission.py, which assembles its own bundle.
    """
    src = open(os.path.join(AGENTS_DIR, agent + ".py")).read()
    if not _ENGINE_IMPORT_RE.search(src):
        return src  # self-contained agent (e.g. the random sample)
    engine = open(os.path.join(AGENTS_DIR, "_engine.py")).read()
    parts = [
        "# --- auto-generated: shared engine inlined for a self-contained submission ---\n",
        engine,
    ]
    body = _ENGINE_IMPORT_RE.sub("", src)
    parts += ["\n\n# --- deck agent ---\n", body]
    return "".join(parts)


SUBMISSION_ARCHIVE = os.path.join("submissions", "_archive")


def _preserve_prior_bundle(name):
    """Never clobber a bundle that may be the only copy of a SCORING build.

    A submission tarball is the ONLY archive of a build: it is self-contained by design
    (main.py inlines the whole engine) and this repo has no version control, so once the
    working tree moves on, the tarball is the sole way back to a build that scored.

    `name` is just f"{agent}-{deck}", so every untagged rebuild of the same pair
    OVERWRITES the last one. That is not hypothetical: of 43 Kaggle submissions, the 10
    with no local copy are ALL `crustle_stall-crustle_stall.tar.gz` -- 11 submissions
    sharing one filename, including our #2 deck's **664.7** build, each silently
    destroying its predecessor. (mega_lucario's **684.5** survived only because it was
    tagged `v29`.)

    So move any existing bundle into submissions/_archive/ stamped with its own mtime
    instead of deleting it. Cheap insurance; the alternative is unrecoverable.
    """
    for suffix in ("", ".tar.gz"):
        src = os.path.join(SUBMISSIONS_DIR, name + suffix)
        if not os.path.exists(src):
            continue
        os.makedirs(SUBMISSION_ARCHIVE, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(src)))
        dst = os.path.join(SUBMISSION_ARCHIVE, f"{name}-{ts}{suffix}")
        if os.path.exists(dst):
            shutil.rmtree(dst, ignore_errors=True) if os.path.isdir(dst) else os.remove(dst)
        shutil.move(src, dst)


def _tuning_entry(name):
    """One tuning.json entry, or {} -- the single source of truth for a deck's engine."""
    try:
        with open(os.path.join(AGENTS_DIR, "tuning.json"), encoding="utf-8") as f:
            return json.load(f).get(name) or {}
    except (OSError, ValueError):
        return {}


def build_submission(agent, deck, _v2_stage=False):
    """Package a Kaggle submission from one agent + one deck.

    Lays out submissions/<agent>-<deck>/ with the files the competition
    expects at archive root, then tars it to submissions/<agent>-<deck>.tar.gz:
        main.py    <- agents/<agent>.py  (defines agent(obs_dict))
        deck.csv   <- decks/<deck>.csv
        cg/        <- the cg library (libcg.so / cg.dll, api.py, ...)

    Returns a dict describing the result (tar path, staged dir, files, bytes).
    Raises ValueError for unknown agent/deck.
    """
    if agent not in set(list_agents()):
        raise ValueError(f"unknown agent: {agent!r}")
    if deck not in set(list_decks()):
        raise ValueError(f"unknown deck: {deck!r}")

    info = deck_info(deck)
    if info["count"] != 60:
        raise ValueError(f"deck {deck!r} has {info['count']} cards (must be 60)")

    # An engine_v2 agent cannot ship through this path: main.py is a straight copy of
    # agents/<agent>.py, and that file now imports agents.engine_v2, which the archive
    # (main.py + deck.csv + cg/) does not contain. tools/build_engine_v2_submission.py
    # exists exactly for this -- it concatenates _engine + engine_v2 + the wrapper into a
    # self-contained main.py. Refuse loudly rather than tar up an ImportError.
    # (_v2_stage=True is build_engine_v2_submission.py reusing this only to STAGE
    # deck.csv + cg/; it overwrites main.py with its own self-contained bundle after.)
    if not _v2_stage and _tuning_entry(agent).get("engine") == "v2":
        raise ValueError(
            f"agent {agent!r} is an engine_v2 agent (tuning.json engine='v2'); "
            f"build it with: PYTHONPATH=cg-lib python tools/build_engine_v2_submission.py "
            f"<l2-key> {deck} --tag <tag>")

    name = f"{agent}-{deck}"
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    _preserve_prior_bundle(name)
    stage = os.path.join(SUBMISSIONS_DIR, name)
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)

    with open(os.path.join(stage, "main.py"), "w") as f:
        f.write(_agent_main_source(agent))
    shutil.copy(deck_path(deck), os.path.join(stage, "deck.csv"))
    shutil.copytree(_cg_source(), os.path.join(stage, "cg"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    tar_path = os.path.join(SUBMISSIONS_DIR, name + ".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tf:
        for root in ("main.py", "deck.csv", "cg"):
            tf.add(os.path.join(stage, root), arcname=root)

    files = []
    for base, _dirs, fnames in os.walk(stage):
        for fn in sorted(fnames):
            rel = os.path.relpath(os.path.join(base, fn), stage)
            files.append(rel.replace(os.sep, "/"))
    return {
        "agent": agent,
        "deck": deck,
        "tar": tar_path,
        "tar_abspath": os.path.abspath(tar_path),
        "dir": stage,
        "bytes": os.path.getsize(tar_path),
        "files": sorted(files),
    }


def submit_to_kaggle(agent, deck, message=None):
    """Build a submission and upload it to Kaggle via the kaggle CLI.

    Returns build info plus {submitted, returncode, output, message}. Raises
    ValueError for an unknown/invalid agent/deck (before anything is uploaded)
    and FileNotFoundError if the kaggle CLI is not installed.
    """
    info = build_submission(agent, deck)  # validates first; nothing uploaded if it raises
    msg = (message or f"{agent} + {deck}").strip() or f"{agent} + {deck}"
    cmd = ["kaggle", "competitions", "submit", "-c", COMPETITION,
           "-f", info["tar"], "-m", msg]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = (proc.stdout + proc.stderr).strip()
    return {
        **info,
        "competition": COMPETITION,
        "message": msg,
        "returncode": proc.returncode,
        "submitted": proc.returncode == 0,
        "output": output,
    }


# ---- card catalog & deck building ------------------------------------------
_cards_cache = None


def _leading_int(s):
    """Leading integer of a string ('120+', '30x', '90') -> int, else None."""
    m = re.match(r"\s*(\d+)", s or "")
    return int(m.group(1)) if m else None


_ace_ids_cache = None


def _ace_spec_ids():
    """Set of card IDs flagged 'ACE SPEC' in the card catalog (rule column)."""
    global _ace_ids_cache
    if _ace_ids_cache is None:
        ids = set()
        with open(CARD_DATA, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) > 5 and row[5].strip() == "ACE SPEC" and row[0].isdigit():
                    ids.add(int(row[0]))
        _ace_ids_cache = ids
    return _ace_ids_cache


def load_cards():
    """Card catalog keyed by id, aggregating every move row of a card:
        {id, name, kind, type, hp, maxDamage, minCost}
    - hp:        numeric HP, or None (energy/trainers).
    - maxDamage: highest move damage, or None if the card has no damaging move.
    - minCost:   fewest energy a move needs (len of the cost symbols), or None
                 if the card has no energy-cost attack (abilities have cost n/a).

    Parsed from CARD_DATA (data/JP_Card_Data.csv) and cached.
    """
    global _cards_cache
    if _cards_cache is None:
        cards = {}
        with open(CARD_DATA, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if not row or not row[0].strip():
                    continue
                try:
                    cid = int(row[0])
                except ValueError:
                    continue
                c = cards.get(cid)
                if c is None:
                    c = cards[cid] = {
                        "id": cid,
                        "name": row[1],
                        "kind": row[4],   # 段階/種類 (ポケモン/たね, グッズ, 基本エネルギー, ...)
                        "hp": _leading_int(row[8]),
                        "type": row[9],
                        "prev": row[7] if row[7] not in ("", "n/a") else None,  # 進化前
                        "prevNames": [],  # full pre-evolution chain (filled below)
                        "maxDamage": None,
                        "minCost": None,
                    }
                # aggregate this row's move (if any) into the card
                move, cost, dmg = row[13], row[14], row[15]
                if move and move != "n/a":
                    d = _leading_int(dmg)
                    if d is not None and (c["maxDamage"] is None or d > c["maxDamage"]):
                        c["maxDamage"] = d
                    if cost and cost != "n/a":
                        n = len(cost)  # each energy symbol is one character
                        if c["minCost"] is None or n < c["minCost"]:
                            c["minCost"] = n

        # Resolve each card's full pre-evolution chain by walking 進化前 links
        # (e.g. オーダイル -> アリゲイツ -> ワニノコ) so a search for a basic
        # Pokemon's name also matches everything that evolves from it.
        name_to_prev = {}
        for c in cards.values():
            if c["prev"]:
                name_to_prev.setdefault(c["name"], c["prev"])
        for c in cards.values():
            chain, seen, p = [], set(), c["prev"]
            while p and p not in seen:
                seen.add(p)
                chain.append(p)
                p = name_to_prev.get(p)
            c["prevNames"] = chain

        _cards_cache = cards
    return _cards_cache


def card_list():
    """Catalog as a list, sorted by card id (for the deck-builder page)."""
    cards = load_cards()
    return [cards[k] for k in sorted(cards)]


def _is_basic_energy(cid):
    c = load_cards().get(cid)
    return bool(c and c["kind"] == "基本エネルギー")


def resolve_deck(name):
    """Resolve a deck *selection* to a concrete deck name.

    RANDOM_DECK picks a random existing deck, chosen fresh on every call (so a
    new deck is drawn each battle). Any other value is returned unchanged.
    """
    if name == RANDOM_DECK:
        decks = list_decks()
        if not decks:
            raise ValueError("no decks available to pick a random one from")
        return random.choice(decks)
    return name


def read_deck(name):
    """Card IDs in decks/<name>.csv, or [] if the deck does not exist."""
    try:
        with open(deck_path(name)) as f:
            return [int(ln) for ln in f if ln.strip()]
    except OSError:
        return []


def save_deck(name, ids):
    """Validate (60 cards, known IDs, <=4 copies except basic energy) and write
    decks/<name>.csv. Returns {name, count, path}. Raises ValueError otherwise.
    """
    if not _DECK_NAME_RE.match(name or ""):
        raise ValueError("deck name must use only letters, digits, '_' or '-'")
    try:
        ids = [int(i) for i in ids]
    except (TypeError, ValueError):
        raise ValueError("card IDs must be integers")
    if len(ids) != DECK_SIZE:
        raise ValueError(f"deck must have exactly {DECK_SIZE} cards (got {len(ids)})")

    cards = load_cards()
    unknown = sorted({i for i in ids if i not in cards})
    if unknown:
        raise ValueError("unknown card IDs: " + ", ".join(map(str, unknown)))

    counts = Counter(ids)
    over = [i for i, n in counts.items() if n > MAX_COPIES and not _is_basic_energy(i)]
    if over:
        detail = ", ".join(f"{cards[i]['name']} x{counts[i]}" for i in sorted(over))
        raise ValueError(f"at most {MAX_COPIES} copies per card (except basic energy): {detail}")

    # At most one ACE SPEC card in the whole deck (else the engine/Kaggle reject it).
    ace = _ace_spec_ids()
    ace_in = [i for i in counts if i in ace]
    ace_total = sum(counts[i] for i in ace_in)
    if ace_total > 1:
        detail = ", ".join(cards[i]["name"] for i in sorted(ace_in))
        raise ValueError(f"at most 1 ACE SPEC card allowed; found {ace_total}: {detail}")

    os.makedirs(DECKS_DIR, exist_ok=True)
    with open(deck_path(name), "w") as f:
        f.write("\n".join(str(i) for i in ids) + "\n")
    return {"name": name, "count": len(ids), "path": deck_path(name)}


def delete_deck(name):
    """Delete decks/<name>.csv. Any config selection pointing at it is repointed
    to another remaining deck so run.py / play_server.py keep working.

    Returns {name, repointed: [slot,...], fallback}. Raises ValueError if the
    name is invalid or the deck does not exist.
    """
    if not _DECK_NAME_RE.match(name or ""):
        raise ValueError("deck name must use only letters, digits, '_' or '-'")
    path = deck_path(name)
    if not os.path.isfile(path):
        raise ValueError(f"deck not found: {name!r}")

    os.remove(path)

    # Repoint any config slot that referenced the deleted deck.
    remaining = list_decks()
    fallback = remaining[0] if remaining else DEFAULT_CONFIG["run"]["player0"]["deck"]
    cfg = load_config()
    repointed = []
    slots = [
        (cfg["run"]["player0"], "deck", "run.player0.deck"),
        (cfg["run"]["player1"], "deck", "run.player1.deck"),
        (cfg["play"], "human_deck", "play.human_deck"),
        (cfg["play"], "ai_deck", "play.ai_deck"),
        (cfg["submit"], "deck", "submit.deck"),
    ]
    for obj, key, label in slots:
        if obj.get(key) == name:
            obj[key] = fallback
            repointed.append(label)
    if repointed:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    return {"name": name, "repointed": repointed, "fallback": fallback}


# ---- card images (from data/Card_ID List_JP.pdf) ---------------------------
# The PDF's first pages are an index whose "券面画像" links jump to a per-card
# page that embeds one face image. We map card id -> image page once, then
# extract the embedded image on demand. fitz (PyMuPDF) is imported lazily so the
# rest of the module works even when it (or the PDF) is absent.
_pdf_doc = None
_pdf_pages = None          # {card_id: page_index}
_pdf_lock = threading.Lock()


def _open_pdf():
    global _pdf_doc, _pdf_pages
    if _pdf_doc is None:
        import fitz  # PyMuPDF
        doc = fitz.open(CARD_IMAGE_PDF)
        pages = {}
        for pno in range(doc.page_count):
            links = doc[pno].get_links()
            if not links:
                continue  # per-card image pages have no links
            words = doc[pno].get_text("words")  # (x0,y0,x1,y1, text, ...)
            for ln in links:
                if ln.get("kind") != 1 or "page" not in ln:
                    continue
                r = ln["from"]
                ymid = (r.y0 + r.y1) / 2
                # the card id is the left-most word on the same row
                row = [w for w in words if w[0] < 100 and w[1] - 2 <= ymid <= w[3] + 2]
                if not row:
                    continue
                token = sorted(row, key=lambda w: w[0])[0][4]
                if token.isdigit():
                    pages[int(token)] = ln["page"]
        _pdf_doc, _pdf_pages = doc, pages
    return _pdf_doc, _pdf_pages


def has_card_images():
    return os.path.exists(CARD_IMAGE_PDF)


def card_image(card_id):
    """Return (content_type, bytes) for a card's face image, or None if there is
    no image for that id. Raises FileNotFoundError if the image PDF is missing.
    """
    if not has_card_images():
        raise FileNotFoundError(CARD_IMAGE_PDF)
    card_id = int(card_id)
    with _pdf_lock:  # fitz Document is not safe for concurrent access
        doc, pages = _open_pdf()
        pno = pages.get(card_id)
        if pno is None:
            return None
        imgs = doc[pno].get_images(full=True)
        if not imgs:
            return None
        info = doc.extract_image(imgs[0][0])
    ext = "jpeg" if info["ext"] in ("jpg", "jpeg") else info["ext"]
    return ("image/" + ext, info["image"])
