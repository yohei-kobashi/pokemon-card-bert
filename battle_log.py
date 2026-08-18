"""Shared battle-log saver for run.py (AI vs AI) and play_server.py (Human vs AI).

Both entry points save a replay into the same ``logs/`` folder in the same
format: the heroz "visualize" array (== env.steps[0][0]["visualize"] ==
cg.game.visualize_data()), which can be replayed with visualizer.html or the
heroz visualizer.

Filename encodes who played and with what:
    {timestamp}_{mode}_{p0agent}-{p0deck}_vs_{p1agent}-{p1deck}.json
e.g.
    20260621-153012_AIvAI_agent-deck_vs_agent-deck.json
    20260621-153500_HumanvAI_human-deck_vs_agent-deck_ai.json
- mode (AIvAI / HumanvAI / AIvHuman) tells AI-vs-AI from AI-vs-Human at a glance.
- each side shows the agent name and the deck name (per player index).
"""

import gzip
import importlib
import json
import os
import re
from datetime import datetime

LOG_DIR = "logs"
DECKS_DIR = "decks"  # deck csv files live here (decks/<name>.csv)
AGENTS_DIR = "agents"  # agent modules live here (agents/<name>.py, each defining agent())


def load_agent(name):
    """Load an agent by name and return its agent(obs_dict) callable.

    'agent' -> agents/agent.py's agent function. Accepts a bare module name
    (the file under agents/, without .py).
    """
    name = os.path.splitext(os.path.basename(name))[0]
    module = importlib.import_module(f"{AGENTS_DIR}.{name}")
    return module.agent


def _slug(s):
    # keep letters/digits/underscore (so "deck_ai" stays "deck_ai"); collapse the rest to "-"
    return re.sub(r"[^A-Za-z0-9_]+", "-", str(s)).strip("-_") or "x"


def deck_path(name):
    """Resolve a deck name to its csv path, e.g. 'deck' -> 'decks/deck.csv'.

    Accepts a bare name, a name with .csv, or a full path (returned as-is).
    """
    if os.path.sep in name or name.endswith(".csv"):
        return name
    return os.path.join(DECKS_DIR, name + ".csv")


def deck_name(deck_file):
    """Deck label from a name or csv path, e.g. 'decks/deck_ai.csv' -> 'deck_ai'."""
    return _slug(os.path.splitext(os.path.basename(deck_file))[0])


def save_battle(visualize, players, when=None):
    """Write a replay to ``logs/`` and return the file path.

    visualize: the visualize/replayData array (list) or its JSON string.
    players:   length-2 list, indexed by player index, each a dict:
                 {"kind": "ai"|"human", "agent": <str>, "deck": <str>}
               ("agent" is ignored for human and shown as "human".)
    when:      optional datetime (defaults to now).
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    data = visualize if isinstance(visualize, str) else json.dumps(visualize)
    ts = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")

    def tag(p):
        return "Human" if p["kind"] == "human" else "AI"

    def label(p):
        agent = "human" if p["kind"] == "human" else p.get("agent", "ai")
        return f"{_slug(agent)}-{_slug(p['deck'])}"

    mode = f"{tag(players[0])}v{tag(players[1])}"
    name = f"{ts}_{mode}_{label(players[0])}_vs_{label(players[1])}.json"
    path = os.path.join(LOG_DIR, name)
    with open(path, "w") as f:
        f.write(data)
    _mirror(path, data, name)
    return path


def _mirror_dir():
    """Second home for finished games: $PTCG_LOG_MIRROR, else config.json play.log_mirror.

    Point it at a Google Drive folder and every battle is uploaded by Drive for desktop
    while the next one is being played, so a play session needs no export step. config.json
    is read RAW here rather than through library.load_config: library imports this module,
    and the reverse import would be a cycle."""
    d = os.environ.get("PTCG_LOG_MIRROR")
    if d:
        return d
    try:
        with open(os.environ.get("PLAY_CONFIG", "config.json")) as f:
            return (json.load(f).get("play") or {}).get("log_mirror") or ""
    except (OSError, ValueError):
        return ""


def _mirror(path, data, name):
    """Copy the finished game to the mirror, gzipped (a battle is ~1-3 MB of JSON).

    Never fatal: a game that has just been played must not be lost because a cloud folder
    was unmounted, so a failure only prints."""
    d = _mirror_dir()
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        with gzip.open(os.path.join(d, name + ".gz"), "wt", encoding="utf-8") as f:
            f.write(data)
    except OSError as e:
        print(f"[log] mirror to {d} failed: {e}")
