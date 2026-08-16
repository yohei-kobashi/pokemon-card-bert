"""Map a rendered menu option to the single vocabulary token that names the ACT it performs.

Why not the menu index. An index names a POSITION, so the same act carries a different label
depending on where the engine happened to list it, and the model has to learn a pointer lookup
instead of the act itself. A content token is the same symbol in every game, and it shares its
embedding with the occurrences of the same card on the board, in hand and in DECK[].

Why not the card id alone. Measured on 400,000 training decisions, a bare card id leaves the act
undetermined on 20.2% of them -- overwhelmingly "attach this energy to the active or to a bench
Pokemon", which is a real decision and, per `attach-decisions-at-chance`, already the model's
weakest. So the target slot has to be part of the token.

What is kept and what is dropped. Board slots (ACTIVE0, BENCH0..BENCH4) are DIFFERENT Pokemon
carrying different damage and different energy, so their index is kept and two bench slots get
two tokens. Positions inside a pile are not: DECK3 and DECK17 are two copies of one card in a
shuffled pile, HAND0 and HAND4 likewise, and a face-down prize is indistinguishable from another
face-down prize. Those indices are dropped, which is what keeps the vocabulary at a few thousand
instead of tens of thousands, and collapsing them loses nothing because the acts are the same.

Per-kind rules, because the options do not share a shape (measured on the real data):

    play:c1227              card, no target
    skill:c1260             card, no target
    card:c1227@HAND0        card + pile position      -> pile index dropped
    card:c131@BENCH1        card + board slot         -> slot kept
    attach:c6@ACTIVE0       card + board slot         -> slot kept
    evolve:c345@BENCH1      card + board slot         -> slot kept
    ability:c675@BENCH2     card + board slot         -> slot kept
    energy:c1@ACTIVE0#0     card + slot + sub-index   -> sub-index dropped (same card, same
                                                         Pokemon: the copies are interchangeable)
    attack:368              a BARE NUMBER that is the attack id -- no `a` prefix, so a regex
                            looking for `a\\d+` silently matches nothing and every attack in the
                            game collapses onto one token. Normalised to a368 here.
    facedown:PRIZE2         no card at all; face-down, so every position is the same act
    num:0                   the NUMBER is the choice (how many to draw/discard), so it is kept
    end retreat stop yes no  bare
"""
import re

_PILE = re.compile(r"^(DECK|HAND|DISCARD|LOST|PRIZE|STADIUM)\d*$")
_SLOT = re.compile(r"^(ACTIVE|BENCH)(\d+)$")
_CARD = re.compile(r"^c\d+$")
_NUM = re.compile(r"^\d+$")

# Kinds whose whole identity is the kind itself.
_BARE = ("end", "retreat", "stop", "yes", "no", "pass")


def action_token(opt):
    """A rendered menu option -> the token naming its act. Never returns None.

    Anything unrecognised falls through to the option string itself, so a format that changes
    under us produces an unknown token (visible, and refused at build time) rather than a wrong
    collision with an existing one.
    """
    kind, _, rest = opt.partition(":")
    if not rest:
        return "A|" + kind                       # end / retreat / stop / yes / no
    if kind == "facedown":
        return "A|facedown"                      # face-down: all positions are the same act
    if kind == "num":
        return "A|num|" + rest                   # the quantity IS the decision
    if kind == "attack":
        return "A|attack|a" + rest if _NUM.match(rest) else "A|attack|" + rest
    body, _, sub = rest.partition("#")           # energy:c1@ACTIVE0#0 -> drop the sub-index
    card, _, tgt = body.partition("@")
    if not tgt:
        return "A|%s|%s" % (kind, card)          # play / skill
    m = _PILE.match(tgt)
    if m:
        tgt = m.group(1)                         # DECK17 -> DECK: interchangeable copies
    return "A|%s|%s@%s" % (kind, card, tgt)


def equivalent(a, b):
    """Do two options perform the same act? (independent of the token, so it can CHECK it)

    True only for differences this module deliberately erases: which copy inside a pile, which
    face-down prize, which of several identical energies. Defined on the raw strings rather than
    on `action_token`, so that comparing the two is a real test and not a tautology.
    """
    if a == b:
        return True
    ka, _, ra = a.partition(":")
    kb, _, rb = b.partition(":")
    if ka != kb:
        return False
    if ka == "facedown":
        return True
    ba, _, _sa = ra.partition("#")
    bb, _, _sb = rb.partition("#")
    ca, _, ta = ba.partition("@")
    cb, _, tb = bb.partition("@")
    if ca != cb:
        return False
    if ta == tb:
        return True                              # differed only in the energy sub-index
    ma, mb = _PILE.match(ta or ""), _PILE.match(tb or "")
    return bool(ma and mb and ma.group(1) == mb.group(1))


def resolve(options, token, engine_index=None):
    """Which menu option did the model mean? -> index, or None if the token names none of them.

    A token can cover several options. Almost always they are the same act (two copies of a card
    in the deck), and any of them is correct. When they are NOT -- which the build refuses to
    ship above a threshold, but which can still arise on unseen data -- the tie goes to
    engine_v2's own choice if that choice is inside the tied set. Deferring exactly the decisions
    the model cannot express is the pattern that paid +11.4pt in `attach-decisions-at-chance`;
    here it costs nothing, because engine_v2 is consulted anyway as the fallback policy.
    """
    hits = [i for i, o in enumerate(options) if action_token(o) == token]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    if engine_index is not None and engine_index in hits:
        return engine_index
    return hits[0]


def ambiguous(options, token):
    """True when `token` covers options that are NOT all the same act -- the cases resolve()
    has to break a tie on."""
    hits = [o for o in options if action_token(o) == token]
    return len(hits) > 1 and not all(equivalent(o, hits[0]) for o in hits[1:])


# ---------------------------------------------------------------------------
# BOARD-SLOT dedup: two slots the PROMPT renders identically
#
# `equivalent` above works on the rendered strings alone, so it collapses positions inside a
# pile but never two board slots -- ACTIVE0/BENCH0/BENCH1 are, in general, different Pokemon.
# In general. Three copies of the same Basic sitting on the bench at full HP with nothing
# attached render as three identical descriptors, and then `attach:c7@BENCH0` and
# `attach:c7@BENCH2` differ only by a slot number that carries no information. Measured on the
# DAgger pool (6,617 attach decisions with >=2 attach candidates): 5.50 candidates of which
# 4.83 are distinguishable, 38.6% of decisions hold an indistinguishable pair, and the top1
# ceiling for a PERFECT model is 86.3%. engine_v2's pick inside such a group is arbitrary, so
# the remaining 13.7pt is label noise that no amount of training removes -- and per
# `attach-decisions-at-chance` attach is already the worst decision kind by a wide margin.
#
# The criterion is what the MODEL can see, not what the observation holds: the descriptor is
# the very string `lm.serialize._pk` puts in the prompt. Two slots that differ only in
# something the renderer drops are indistinguishable to the model whatever the engine knows,
# so ranking them apart is unlearnable either way; collapsing at least stops punishing it.
#
# This must be applied in build_rerank, collect_dagger AND lm/agent together. A model trained
# with the twins collapsed has never had to rank them apart, so leaving them apart at
# inference asks for a comparison its training never contained.
# ---------------------------------------------------------------------------

_AREA_LIST = {"ACTIVE": "active", "BENCH": "bench"}
# One rendered Pokemon: `c123*:180/210|G2|c456 need:1 rt:2`. Bench entries are comma-joined but
# so is the TOOL list inside an entry, so splitting on "," is wrong; a new entry is the only
# place a `c<id>[*]:` can follow a comma.
_ENTRY = re.compile(r",(?=c\d+\*?:)")
_ME = re.compile(r"(?:^|\s)ME (.*?)(?: \| OP |$)", re.S)
_ACTIVE_SEG = re.compile(r"(?:^|\s)A\[(.*?)\](?=\s|$)")
_BENCH_SEG = re.compile(r"(?:^|\s)B\[(.*?)\](?=\s|$)")


def slot_map_from_obs(obs, board_facts=True):
    """{"ACTIVE0": descriptor, "BENCH2": descriptor} for the side whose prompt this is.

    The descriptor is the very string `lm.serialize._pk` writes into the prompt, so "equal
    descriptor" means "the model was shown the same thing" by construction rather than by a
    re-implementation that could drift from the renderer.
    """
    out = {}
    try:
        cur = obs["current"]
        pl = cur["players"][cur["yourIndex"]]
        from lm.serialize import _pk
        for nm, key in (("ACTIVE", "active"), ("BENCH", "bench")):
            for i, p in enumerate(pl.get(key) or []):
                if p:
                    out["%s%d" % (nm, i)] = _pk(p, board_facts)
    except Exception:
        return {}
    return out


def slot_map_from_state(state):
    """The same map recovered from an already-rendered prompt, for pools with no observation.

    NOTE the bench index here is the RENDERED position. `lm.serialize._side` drops empty slots
    before joining, so a board with a hole renders BENCH1 where the option says BENCH2 -- which
    is a real prompt-level defect, not one this function should paper over. It only ever makes
    two entries look DIFFERENT that are the same, so the dedup stays conservative.
    """
    m = _ME.search(state or "")
    if not m:
        return {}
    seg = m.group(1)
    out = {}
    a = _ACTIVE_SEG.search(seg)
    if a and a.group(1) != "-":
        for i, e in enumerate(_ENTRY.split(a.group(1))):
            out["ACTIVE%d" % i] = e.strip()
    b = _BENCH_SEG.search(seg)
    if b and b.group(1) != "-":
        for i, e in enumerate(_ENTRY.split(b.group(1))):
            out["BENCH%d" % i] = e.strip()
    return out


# WHICH SIDE is the slot on? The rendered option does not say -- `lm/actions._target` emits
# `@BENCH2` from inPlayArea/inPlayIndex and drops `playerIndex` -- while the prompt only renders
# OUR board in full. Measured over 21 games on 7 decks:
#
#     kind      own-side   OPP-side
#     attach       7199          0
#     ability       654          0
#     evolve        599          0
#     card          885        305      <- 25.6% of them are the opponent's board
#     energy        113         31      <- 21.5%
#
# So a descriptor lookup is only sound for the three kinds that are structurally own-side. For
# `card` and `energy` the string is genuinely ambiguous and the MODEL cannot resolve it either;
# that is a prompt defect to fix in `encode_option` (render the side), not something to paper
# over here by silently keying an opponent's bench against our own.
_OWN_SIDE_KINDS = ("attach", "evolve", "ability")


def canon_key(text, slots=None):
    """A key that is equal for two options exactly when they are the same act to the model.

    With `slots` empty this returns exactly the grouping `equivalent` gives, so the string-only
    behaviour is unchanged wherever no board is available.

    Deliberately NOT a function of the option dict. The observation carries `playerIndex` and a
    pool of rendered prompts does not, so keying on it would make the live path and the
    post-processing path collapse different sets -- and a model trained on one grouping and run
    on the other is the exact failure this dedup exists to remove.
    """
    kind, _, rest = text.partition(":")
    if not rest:
        return kind                                   # end / retreat / stop / yes / no
    if kind == "facedown":
        return "facedown"                             # every face-down position is one act
    body, _, _sub = rest.partition("#")               # energy:c1@ACTIVE0#0 -> drop the sub-index
    card, _, tgt = body.partition("@")
    if not tgt:
        return "%s|%s" % (kind, card)
    m = _PILE.match(tgt)
    if m:
        return "%s|%s@%s" % (kind, card, m.group(1))  # DECK17 -> DECK
    d = (slots or {}).get(tgt) if kind in _OWN_SIDE_KINDS else None
    return "%s|%s@%s" % (kind, card, tgt if d is None else "=" + d)


def dedup_options(texts, obs=None, state=None, board_facts=True):
    """-> (kept texts, index in `texts` of each kept one, key per original option).

    Give it `obs` on a live path and `state` when post-processing a built pool; the two are
    checked to agree (0 disagreements over 2,371 live decisions). With neither it degrades to
    the string-only dedup. The FIRST occurrence wins, so every surviving text is one the engine
    actually produced.
    """
    slots = (slot_map_from_obs(obs, board_facts) if obs
             else (slot_map_from_state(state) if state else {}))
    keys = [canon_key(t, slots) for t in texts]
    seen, keep, pos = {}, [], []
    for i, k in enumerate(keys):
        if k not in seen:
            seen[k] = len(keep)
            keep.append(texts[i])
            pos.append(i)
    return keep, pos, keys


# ---------------------------------------------------------------------------
# CARD-FIRST answer scheme
#
# The answer is the CARD token -- one the model already knows, because every card appears in the
# prompt (board, hand, DECK[]) millions of times and the embedding is tied, so its output row is
# its input row. That removes the long tail the atomic `A|kind|card@slot` scheme created: 6,282
# freshly initialised rows of which 28% were seen fewer than 20 times.
#
# A second token is emitted ONLY where the card does not settle it -- measured at 14.86% of
# decisions, so inference costs 1.149 forward passes on average. It names the option by a
# deterministic SORT rather than by menu position: in play before anything else, active before
# bench, more energy attached, then lower remaining HP. That order is meant to put the
# most-committed target first, and it does -- <s0> is the answer 57.1% of the time against 30%
# for an uninformative order.
#
# The sort is computed FROM THE PROMPT, not from the observation. The prompt is what the model
# reads, so training and inference cannot end up ordering the same menu differently -- and the
# labels in the training file are prompts, with no observation attached to reconstruct.
# ---------------------------------------------------------------------------

_IDTOK = re.compile(r"(c\d+|a\d+)")
MAX_SUB = 64
SUB_TOKENS = ["<s%d>" % k for k in range(MAX_SUB)]


def first_token(opt):
    """The card (or attack) an option names -- the first half of the answer."""
    kind, _, rest = opt.partition(":")
    if not rest:
        return "A|" + kind                     # end / retreat / stop / yes / no
    if kind == "facedown":
        return "A|facedown"
    if kind == "num":
        return "A|num|" + rest                 # the quantity IS the decision
    if kind == "attack":
        return ("a" + rest) if rest.isdigit() else rest
    m = _IDTOK.search(rest)
    return m.group(1) if m else "A|" + kind


def parse_board(prompt):
    """My side of the board, as {slot: (energy attached, remaining HP)}.

    Read out of the rendered prompt, e.g. `ME A[c741*:50/50|G3 need:1 rt:1] B[c66:60/60|G2]`.
    A slot that cannot be parsed falls back to (0, 9999), which sorts it last among in-play
    options rather than crashing -- a mis-parse should cost ordering, not the decision.
    """
    m = re.search(r" ME (A\[[^\]]*\])(?: (B\[[^\]]*\]))?", prompt)
    out = {}
    if not m:
        return out

    def one(txt):
        hp = re.search(r":(\d+)/(\d+)", txt)
        en = re.search(r"\|([A-Z](?:\d+)?(?:[A-Z]\d*)*)", txt)
        e = 0
        if en:
            for sym, cnt in re.findall(r"([A-Z])(\d*)", en.group(1)):
                e += int(cnt) if cnt else 1
        return (e, int(hp.group(1)) if hp else 9999)

    out["ACTIVE0"] = one(m.group(1)[2:-1])
    if m.group(2):
        for i, s in enumerate(m.group(2)[2:-1].split(",")):
            out["BENCH%d" % i] = one(s)
    return out


def sort_key(opt, board):
    tgt = (opt.split("@", 1)[1] if "@" in opt else "").split("#")[0]
    s = _SLOT.match(tgt)
    if not s:
        return (2, 0, 0, opt)                      # not in play, after everything that is
    e, hp = board.get(tgt, (0, 9999))
    return (0 if s.group(1) == "ACTIVE" else 1, -e, hp, opt)


def tie_group(options, tok):
    """Indices of the options that share this first token."""
    return [i for i, o in enumerate(options) if first_token(o) == tok]


def sub_index(prompt, options, idx, board=None):
    """Which <sN> names options[idx], or None when the card alone settles it.

    None is returned whenever every option sharing the card performs the SAME act, because then
    any of them is a correct answer and a second token would be teaching a distinction that does
    not exist.

    KEPT ON THE STRING-ONLY TEST ON PURPOSE. Board slots the prompt renders identically are the
    same act too, and scheme B collapses them -- but <sN> is a RANK in `order`, so collapsing
    here shortens that list and every rank after the collapse shifts. The scheme-A checkpoint
    was trained against the un-collapsed ranks, so changing this would silently redirect its
    answers. Scheme A therefore keeps the arbitrary label; the fix ships with scheme B, which
    names the act instead of counting positions. Same reason for `resolve_card_first`.
    """
    grp = tie_group(options, first_token(options[idx]))
    if len(grp) < 2 or all(equivalent(options[i], options[idx]) for i in grp):
        return None
    b = parse_board(prompt) if board is None else board
    order = sorted(grp, key=lambda i: sort_key(options[i], b))
    r = order.index(idx)
    return r if r < MAX_SUB else None              # beyond the alphabet: fall back to one token


def resolve_card_first(prompt, options, tok, sub=None, engine_index=None):
    """-> the menu index the model meant, or None if the token names no legal option."""
    grp = tie_group(options, tok)
    if not grp:
        return None
    if len(grp) == 1:
        return grp[0]
    if all(equivalent(options[i], options[grp[0]]) for i in grp):
        return grp[0]                              # same act; any of them is right
    order = sorted(grp, key=lambda i: sort_key(options[i], parse_board(prompt)))
    if sub is not None and 0 <= sub < len(order):
        return order[sub]
    if engine_index is not None and engine_index in grp:
        return engine_index                        # no second token: defer the tie
    return order[0]


# ---------------------------------------------------------------------------
# SCHEME B -- the answer's second half names the ACT, not a rank
#
# Scheme A settles a tie with <sN>, where N is the option's place in a sort computed by parsing
# the board out of the prompt. That works, but the order is something the model must re-derive,
# and the parser is a regex over rendered text: if it ever mis-reads, training and inference
# mis-read identically and nothing reports it.
#
# Here the second token names the act directly -- `K|attach@BENCH1` -- so there is no sort and no
# parse. The menu is rendered with THE SAME TOKENS the answer uses, so choosing a move is copying
# a symbol out of the menu rather than counting positions in it.
#
# The vocabulary stays small because the target is normalised exactly as elsewhere: board slots
# keep their index (different Pokemon), pile positions do not (interchangeable copies).
# ---------------------------------------------------------------------------


def second_token(opt):
    """The token naming WHICH option, within the group that shares a first token."""
    kind, _, rest = opt.partition(":")
    if not rest or kind in ("facedown", "num", "attack"):
        return "K|" + kind
    body, _, _sub = rest.partition("#")
    _card, _, tgt = body.partition("@")
    if not tgt:
        return "K|" + kind
    m = _PILE.match(tgt)
    return "K|%s@%s" % (kind, m.group(1) if m else tgt)


def groups(options, prompt=None):
    """-> [(first token, [options])], in menu order, equivalent options collapsed.

    Give it the PROMPT and board slots the prompt renders identically collapse too. Without it
    `K|attach@BENCH0` and `K|attach@BENCH2` are two menu entries and two possible answers for
    three copies of one Basic sitting at full HP with nothing attached -- the same arbitrary
    label the reranker was carrying, and here it also spends menu tokens on the duplicates.
    """
    slots = slot_map_from_state(prompt) if prompt else {}
    out = []
    idx = {}
    for o in options:
        t = first_token(o)
        if t not in idx:
            idx[t] = len(out)
            out.append((t, []))
        keep = out[idx[t]][1]
        k = canon_key(o, slots)
        if not any(canon_key(x, slots) == k for x in keep):
            keep.append(o)
    return out


def render_menu_b(options, prompt=None):
    """The scheme-B menu: one entry per act, spelled with the answer's own tokens.

    A group with a single act needs no second token, so it renders as the first token alone --
    which is also exactly what the answer will be.
    """
    parts = []
    for tok, os_ in groups(options, prompt):
        if len(os_) == 1:
            parts.append(tok)
        else:
            parts.append(tok + ">" + " ".join(second_token(o) for o in os_))
    return " ".join(parts)


def to_scheme_b(prompt):
    """Rewrite a scheme-A prompt's menu in place. Everything before ` :: ` is untouched.

    Converting rather than re-serialising is deliberate: the training pool holds rendered
    prompts, not observations, so re-serialising would mean replaying every game. The menu is
    fully recoverable from the menu, which is the only part that changes.
    """
    head, sep, menu = prompt.rpartition(" :: ")
    if not sep:
        return prompt
    opts = [t for _, t in _RE_MENU.findall(menu)]
    if not opts:
        return prompt
    # the board the slots are read from lives in `head`, so pass the whole prompt
    return head + " :: " + render_menu_b(opts, prompt)


_RE_MENU = re.compile(r"(?:^| )(\d+)=(\S+)")


def label_b(prompt_or_options, idx, options=None):
    """The scheme-B answer for options[idx]: (first token, second token or None).

    Pass the PROMPT as the first argument when `options` is given, so the menu this label is
    read against and the label itself collapse the same board slots. The answer names the
    SURVIVING member of the collapsed group -- emitting `K|attach@BENCH2` when the menu only
    shows `K|attach@BENCH0` would train the model to produce a token that is not on offer.
    """
    opts = options if options is not None else prompt_or_options
    prompt = prompt_or_options if (options is not None
                                   and isinstance(prompt_or_options, str)) else None
    tok = first_token(opts[idx])
    grp = [g for t, g in groups(opts, prompt) if t == tok]
    if not grp or len(grp[0]) < 2:
        return tok, None
    slots = slot_map_from_state(prompt) if prompt else {}
    k = canon_key(opts[idx], slots)
    rep = next((o for o in grp[0] if canon_key(o, slots) == k), opts[idx])
    return tok, second_token(rep)


def parse_menu_b(menu):
    """Read back a scheme-B menu. -> [(first token, [second tokens])] in menu order.

    Needed because the B menu has no `N=` markers for a regex to key on -- which is the point,
    but it means the menu can only be read by the same grammar that wrote it. Group heads are
    bare tokens (optionally followed by `>`); every `K|...` that follows belongs to the group
    above it.
    """
    out = []
    for w in menu.split():
        if w.startswith("K|"):
            if out:
                out[-1][1].append(w)
            continue
        head, sep, first_sec = w.partition(">")
        out.append((head, [first_sec] if sep and first_sec else []))
    return out
