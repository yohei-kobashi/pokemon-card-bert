"""Deploy scorer (component C): CPU llama.cpp candidate scorer for the Kaggle build.

``LlamaScorer.score(prompt, candidates, obs=None) -> list[float]`` returns a
length-normalised log-prob per candidate (higher = more likely). It is the ``model``
consumed by ``lm.agent.make_lm_agent`` -- the agent turns EVERY real decision into a
candidate-scoring call and takes the argmax, so the output is always a legal move.

Speed comes from KV reuse: the prompt is prefilled ONCE per decision; the post-prompt
recurrent+KV state is snapshotted with the low-level ``llama_state_seq_get_data`` API
(~20MB, ~10ms) and RESTORED before each candidate, so candidates are scored from a warm
state instead of re-prefilling the prompt N times. (Qwen3.5 is a recurrent/linear-attn
hybrid -- its state can't be position-rolled-back, so we save/restore the whole seq.)

Last-token logits are read straight from ``llama_get_logits_ith(ctx, -1)`` via ctypes;
the Python-side ``eval_logits``/``scores`` buffers read back all-zero (uniform) with
``logits_all=False`` and must NOT be used.

TIME BANK: Kaggle gives a 600s cumulative thinking-time bank per game; running out is a
forfeit LOSS. ``score`` tracks cumulative inference time and RAISES once ``time_budget``
is spent, so ``make_lm_agent`` falls back to the engine for the rest of the game and
never forfeits. This is pure safety (never routes on difficulty).
"""
import ctypes
import math
import time

import numpy as np


class LlamaScorer:
    def __init__(self, model_path, n_threads=4, n_ctx=2048,
                 time_budget=480.0, max_prompt=1900):
        import llama_cpp
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         n_threads_batch=n_threads,   # cap PREFILL threads too (else it
                         logits_all=False, verbose=False)  # oversubscribes on many-core hosts
        self.ctx = self.llm._ctx.ctx
        self.NV = self.llm.n_vocab()
        self._GZ = llama_cpp.llama_state_seq_get_size
        self._G = llama_cpp.llama_state_seq_get_data
        self._S = llama_cpp.llama_state_seq_set_data
        self._GLI = llama_cpp.llama_get_logits_ith
        self._GLI.restype = ctypes.POINTER(ctypes.c_float)
        self.time_budget = time_budget
        self.max_prompt = max_prompt
        self.spent = 0.0            # cumulative inference seconds THIS GAME (the time bank)
        self.n_decisions = 0

    def reset_bank(self):
        """New game -> refill the 600s thinking-time bank. Called at deck selection so a
        persistent agent process doesn't carry one game's spend into the next."""
        self.spent = 0.0
        self.n_decisions = 0

    def _tok(self, s):
        return self.llm.tokenize(s.encode(), add_bos=False, special=False)

    def _last_logits(self):
        p = self._GLI(self.ctx, -1)
        return np.ctypeslib.as_array(p, shape=(self.NV,)).astype(np.float64)

    @staticmethod
    def _lsm(logits, i):
        a = logits - logits.max()
        e = np.exp(a)
        return math.log(e[i] / e.sum())

    def score(self, prompt, candidates, obs=None):
        if self.spent >= self.time_budget:
            raise RuntimeError("time budget spent -> engine fallback")
        t0 = time.time()
        pt = self._tok(prompt)
        if len(pt) > self.max_prompt:
            pt = pt[-self.max_prompt:]
        self.llm.reset()
        self.llm.eval(pt)
        n_prompt = self.llm.n_tokens
        prompt_logits = self._last_logits()             # predicts candidate token 0
        sz = self._GZ(self.ctx, 0)
        buf = (ctypes.c_uint8 * sz)()
        self._G(self.ctx, buf, sz, 0)                   # snapshot post-prompt state
        out = []
        for cs in candidates:
            ct = self._tok(cs) or [self.llm.token_eos()]
            lp = self._lsm(prompt_logits, ct[0])
            for t in range(1, len(ct)):
                self.llm.eval([ct[t - 1]])
                lp += self._lsm(self._last_logits(), ct[t])
            self._S(self.ctx, buf, sz, 0)               # restore warm state
            self.llm.n_tokens = n_prompt
            out.append(lp / len(ct))
        self.spent += time.time() - t0
        self.n_decisions += 1
        return out
