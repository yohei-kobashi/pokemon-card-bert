"""Map an internet decklist (count + card name) to pool card IDs and build a CSV.

Usage (as a library): from tools._deckbuild import name_to_id, map_list, write_deck
Names are matched case/space/apostrophe/accent-insensitively against
data/EN_Card_Data.csv. Basic energies collapse to their canonical id. Reports
any unmatched names so they can be substituted by hand.
"""
import os, csv, re, unicodedata, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD_DATA = os.path.join(ROOT, "data", "EN_Card_Data.csv")


def _norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("’", "'").replace("`", "'").replace("’", "'")
    s = s.lower()
    s = re.sub(r"\{([a-z])\}", r"\1", s)  # keep energy letter: {r}->r (basic energies)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


_NAME2ID = None
_ID2NAME = None


def _build():
    global _NAME2ID, _ID2NAME
    _NAME2ID, _ID2NAME = {}, {}
    with open(CARD_DATA, encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            if len(row) < 2 or not row[0].strip().isdigit():
                continue
            cid = int(row[0]); nm = row[1].strip()
            _ID2NAME[cid] = nm
            _NAME2ID.setdefault(_norm(nm), cid)


def name_to_id(name):
    if _NAME2ID is None:
        _build()
    n = _norm(name)
    if n in _NAME2ID:
        return _NAME2ID[n]
    # try common energy phrasings: "fire energy" -> "basic r energy"
    ENERGY = {"grass": "g", "fire": "r", "water": "w", "lightning": "l",
              "psychic": "p", "fighting": "f", "darkness": "d", "dark": "d",
              "metal": "m"}
    m = re.match(r"(?:basic\s+)?([a-z]+)\s+energy$", n)
    if m and m.group(1) in ENERGY:
        cand = _norm(f"basic {{{ENERGY[m.group(1)]}}} energy")
        if cand in _NAME2ID:
            return _NAME2ID[cand]
    # loose contains match (unique)
    hits = [cid for k, cid in _NAME2ID.items() if n and (n in k or k in n)]
    if len(set(hits)) == 1:
        return hits[0]
    return None


def map_list(entries):
    """entries: list of (count, name). Returns (id_counts dict, unmatched list)."""
    cnt = collections.Counter()
    unmatched = []
    for c, nm in entries:
        cid = name_to_id(nm)
        if cid is None:
            unmatched.append((c, nm))
        else:
            cnt[cid] += c
    return dict(cnt), unmatched


def parse_text(txt):
    """Parse 'Nx Card Name' or 'N Card Name' lines -> [(count, name)]."""
    out = []
    for line in txt.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        m = re.match(r"(\d+)\s*[xX]?\s+(.+)", line)
        if not m:
            continue
        name = m.group(2)
        name = re.sub(r"\s*\([^)]*\)\s*$", "", name)   # strip trailing (SET 12)
        name = re.sub(r"\s+[A-Z]{2,4}\s+\d+.*$", "", name)  # strip 'SV9a 20'
        out.append((int(m.group(1)), name.strip()))
    return out


def write_deck(name, id_counts):
    from agents._engine import _CARDS  # noqa
    total = sum(id_counts.values())
    lines = []
    for cid, n in sorted(id_counts.items(), key=lambda x: -x[1]):
        lines += [str(cid)] * n
    path = os.path.join(ROOT, "decks", f"{name}.csv")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return total, path
