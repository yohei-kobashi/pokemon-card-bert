"""Which deck plays with which adapter.

WHY THIS EXISTS
    Up to round 8 one checkpoint piloted all eleven decks, so "the model" was a single path
    typed into every command line. Per-deck LoRAs break that: five decks, five adapters, and
    every tool that builds a pilot -- gate_protagonist, mirror_match, lm_mirror_log,
    build_rerank_submission -- has to agree on which one a deck gets. A registry is the single
    place that answers it, so a deck cannot be evaluated with one adapter and shipped with
    another.

TWO RULES THAT ARE NOT NEGOTIABLE
    * A deck with an entry whose adapter is MISSING raises. It must never fall back to the
      default: the failure mode is silent and total -- all five decks would load the same
      weights and every per-deck comparison would measure one model against itself.
    * `spec_for` returns a CONCRETE spec (an absolute path). mirror_match caches scorers by
      spec string, so resolving lazily inside the cache -- keying on "reg" -- would hand deck
      two the adapter built for deck one.

PATHS ARE RELATIVE ON PURPOSE
    Adapters live at /root/out on the rented machines and under models/ locally. Entries store
    a bare name ("lora_dusknoir_r3"); the root comes from $PTCG_MODELS, so the same registry
    file is correct on the laptop and on both instances. An entry may still hold an absolute
    path or a hub id (anything with a '/') and it is used as written.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILE = os.path.join(ROOT, "models", "adapters.json")
FMTS = ("prompt", "dusk")
KINDS = ("qwen", "hf", "rerank", "engine", "remote")


class RegistryError(RuntimeError):
    pass


def registry_path(path=None):
    return path or os.environ.get("PTCG_ADAPTERS") or DEFAULT_FILE


def models_root(reg=None):
    """Where bare adapter names resolve. $PTCG_MODELS wins; then the registry's own "root";
    then /root/out on the vast boxes; then <repo>/models."""
    env = os.environ.get("PTCG_MODELS")
    if env:
        return env
    if reg and reg.get("root"):
        return reg["root"]
    for c in ("/root/out", os.path.join(ROOT, "models")):
        if os.path.isdir(c):
            return c
    return os.path.join(ROOT, "models")


def load(path=None):
    p = registry_path(path)
    if not os.path.exists(p):
        return {"version": 1, "default": None, "decks": {}}
    with open(p) as f:
        reg = json.load(f)
    reg.setdefault("decks", {})
    reg.setdefault("default", None)
    return reg


def save(reg, path=None):
    p = registry_path(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)          # never leave a half-written registry behind
    return p


def _abs_target(target, reg):
    """"qwen:lora_x" -> "qwen:/root/out/lora_x". A target with a '/' is taken as written, so
    absolute paths and hub ids ("unsloth/Qwen3-4B") pass through."""
    kind, _, rest = target.partition(":")
    if not rest:
        return target                     # bare "engine"
    if "/" in rest or os.path.isabs(rest):
        return target
    return "%s:%s" % (kind, os.path.join(models_root(reg), rest))


def _entry_target(entry):
    t = entry.get("target") or entry.get("spec")
    if not t:
        raise RegistryError("entry has no 'target': %r" % (entry,))
    return t


def entry_for(deck, reg=None):
    """The registry entry a deck plays with, or the default. Returns (entry, source) where
    source is 'deck' or 'default' -- callers report it so an unnoticed fallback is visible."""
    reg = load() if reg is None else reg
    e = (reg.get("decks") or {}).get(deck)
    if e is not None:
        return e, "deck"
    d = reg.get("default")
    if d is None:
        raise RegistryError(
            "no entry for deck %r and no default in %s" % (deck, registry_path()))
    return d, "default"


def resolve(deck, reg=None, require_exists=True):
    """-> {"spec", "fmt", "target", "source", "entry"}.

    `spec` is what mirror_match.make_agent takes, wrappers already applied. `require_exists`
    checks the adapter directory is present on THIS machine; leave it on anywhere a pilot is
    actually built."""
    reg = load() if reg is None else reg
    entry, source = entry_for(deck, reg)
    target = _abs_target(_entry_target(entry), reg)
    fmt = entry.get("fmt", "prompt")
    if fmt not in FMTS:
        raise RegistryError("deck %r: fmt %r not in %s" % (deck, fmt, list(FMTS)))
    kind = target.partition(":")[0]
    if kind not in KINDS:
        raise RegistryError("deck %r: unknown target kind %r in %r" % (deck, kind, target))
    path = target.partition(":")[2]
    # "the directory is there" is not "the model is there": a checkpoint copied between machines
    # exists as an empty tree from the first second of the transfer, and every consumer of this
    # registry would happily point a run at it. Require actual weights.
    WEIGHTS = ("model.safetensors", "adapter_model.safetensors", "pytorch_model.bin",
               "adapter_model.bin", "model.onnx")
    if kind == "remote":
        # The weights are on the OTHER machine by definition, so there is nothing here to
        # check and a filesystem test would reject every valid entry. The equivalent check --
        # does that server actually hold this adapter -- costs a round trip, so it lives in
        # `tools/adapters.py check` rather than in a function every pilot build calls.
        # Falls through to the defer/wrap wrapping below, which applies the same way.
        missing = False
    else:
        missing = bool(path) and "/" in path and not os.path.exists(path)
    if not missing and os.path.isdir(path):
        missing = not any(os.path.exists(os.path.join(path, w)) for w in WEIGHTS)
    if missing and require_exists:
        why = "does not exist here" if not os.path.isdir(path) else "holds no weights file yet"
        if source == "deck":
            raise RegistryError(
                "deck %r points at %s, which %s. Refusing to fall back to the default -- that "
                "would silently pilot every deck with one adapter." % (deck, path, why))
        raise RegistryError("default adapter %s %s" % (path, why))

    spec = target
    for kinds in (entry.get("defer") or [],):
        if kinds:
            spec = "defer:%s:%s" % (",".join(kinds), spec)
    if entry.get("wrap"):
        # free-form prefix, e.g. "planengine:recon" -> "planengine:recon:<spec>"
        spec = "%s:%s" % (entry["wrap"].rstrip(":"), spec)
    return {"spec": spec, "fmt": fmt, "target": target, "source": source,
            "entry": entry, "exists": not missing}


def spec_for(deck, reg=None, with_fmt=False, require_exists=True):
    """The spec string. with_fmt appends '@dusk' for the tools that parse it (gate_protagonist,
    lm_mirror_log arms); mirror_match's own --a/--b take the format from --fmt instead."""
    r = resolve(deck, reg, require_exists=require_exists)
    return r["spec"] + ("@" + r["fmt"] if with_fmt else "")


def set_deck(deck, target=None, fmt=None, defer=None, wrap=None, note=None,
             live=None, gate=None, reg=None, path=None):
    """Create or update one deck's entry, then persist. Only the fields passed are touched."""
    reg = load(path) if reg is None else reg
    e = dict((reg.get("decks") or {}).get(deck) or {})
    if target is not None:
        e["target"] = target
    if fmt is not None:
        if fmt not in FMTS:
            raise RegistryError("fmt %r not in %s" % (fmt, list(FMTS)))
        e["fmt"] = fmt
    if defer is not None:
        e["defer"] = list(defer)
    if wrap is not None:
        e["wrap"] = wrap or None
    if note is not None:
        e["note"] = note
    if live is not None:
        e["live"] = live
    if gate is not None:
        e["gate"] = gate
    if "target" not in e:
        raise RegistryError("deck %r has no target; pass --target" % deck)
    e.setdefault("fmt", "prompt")
    reg.setdefault("decks", {})[deck] = e
    save(reg, path)
    return e


def remove_deck(deck, reg=None, path=None):
    reg = load(path) if reg is None else reg
    gone = (reg.get("decks") or {}).pop(deck, None)
    if gone is not None:
        save(reg, path)
    return gone


def check(reg=None):
    """-> list of (deck, ok, detail) over every entry plus the default. Machine-local: an
    adapter absent here is not a broken registry, it is a machine that has not synced."""
    reg = load() if reg is None else reg
    rows = []
    names = sorted((reg.get("decks") or {}))
    for deck in names:
        try:
            r = resolve(deck, reg, require_exists=False)
            rows.append((deck, r["exists"], r["spec"]))
        except RegistryError as ex:
            rows.append((deck, False, str(ex)))
    return rows
