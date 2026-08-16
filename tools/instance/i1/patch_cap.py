"""Tighten the body-need gate so it fires only near the bench-out cliff.

With deck_low=20 already applied, the boost pays where bench-out is still HIGH after the mill fix
(archaludon 50.0% -> 24.3%, +6.2pt; honchkrow 49.7% -> 38.9%, +3.1pt) and does nothing or slightly
hurts where bench-out was already low (crustle_geco 4.4%, mega_venusaur 12.8%, ns_zoroark 14.3%,
ceruledge 16.8%). Firing through `bench < bench_target` covers most of the early game, so on those
decks it displaces fetches that were fine. ENGINE_BODY_NEED_CAP caps the trigger.
"""
import os

p = os.path.join(os.getcwd(), "agents/engine_v2.py")
s = open(p).read()

OLD_CONST = '_BODY_NEED = 2000 if os.environ.get("ENGINE_BODY_NEED", "1") != "0" else 0\n'
NEW_CONST = OLD_CONST + (
    '# How thin the bench must be before the body-need boost fires. bench_target (the default)\n'
    '# fires through most of the early game, which displaces useful fetches on decks that were\n'
    '# never bench-out-prone; a tighter cap fires only near the cliff.\n'
    '_BODY_NEED_CAP = int(os.environ.get("ENGINE_BODY_NEED_CAP", "0"))   # 0 = use bench_target\n'
)

OLD_GATE = (
    "            if len(ctx.me.bench) < self.bench_target:\n"
    "                s += _BODY_NEED\n"
)
NEW_GATE = (
    "            _cap = _BODY_NEED_CAP or self.bench_target\n"
    "            if len(ctx.me.bench) < min(self.bench_target, _cap):\n"
    "                s += _BODY_NEED\n"
)

if "_BODY_NEED_CAP" in s:
    print("already patched")
else:
    assert s.count(OLD_CONST) == 1, "const anchor"
    assert s.count(OLD_GATE) == 1, "gate anchor"
    open(p, "w").write(s.replace(OLD_CONST, NEW_CONST).replace(OLD_GATE, NEW_GATE))
    print("patched cap into", p)
