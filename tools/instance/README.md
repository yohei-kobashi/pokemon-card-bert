# Instance-side scripts (archived 2026-08-17, competition close)

Working scripts pulled verbatim from the two rented Vast.ai boxes on the final day.
They are an ARCHIVE of how the training/eval infrastructure actually ran, not a curated
library -- most were written for one experiment, and their comments carry the measurements
that justified each step.

- `i1/` -- instance1 (RTX 4090, 61 effective cores): the DeBERTa mirror-RL/field loop
  (`field_chain.sh` + `field_keep.sh`/`keepd.sh` supervisors), DPO pair branching,
  gate harnesses (`gate_rules.sh`, `ogre_wrap_ab.sh`, `wrapdiff_gate.sh`), submission
  builders (`submit_dusk_v*.sh`), the human-game pipeline (`human_dpo.py`,
  `human_diverge_dump.py`), and fleet baselines (`baseline_*.py`).
- `i2/` -- instance2 (13.4 effective cores): the Qwen3.5-4B side -- per-deck LoRA DPO rounds
  (`dpo_round*.sh`, `deck_loras.sh`), trace generation for instance1 (`gend2.sh`), the
  cross-instance score server plumbing, and the ssh keep-alive (`keyheal.sh`) that survives
  vast.ai's periodic authorized_keys rewrites.

Paths inside the scripts are absolute (`/root/...`) and assume the repo at /root/ptcg/repo
with cg-lib on PYTHONPATH. Companion models are on HuggingFace (see the top-level README).
