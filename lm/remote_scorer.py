"""Score a decision on ANOTHER machine's GPU.

WHY. instance1 trains the dusknoir reranker on its own 4090 and gates it against eight
opponent decks. The plan is for those opponents to stop being engine_v2 and start being the
per-deck Qwen3-4B LoRAs -- but those adapters live on instance2, the 4090 is busy training,
and 4B x 8 adapters does not fit next to a training run in 24 GiB anyway. So instance1 renders
the prompt exactly as it does today and sends only `(prompt, candidates)` to instance2, which
holds one shared 4B base with every adapter loaded on top (tools/score_server.py).

Nothing about the PROMPT moves. The registry still decides which adapter a deck plays with and
which format it renders in, mirror_match still recurses through `reg`, and the only thing that
changes is where the forward pass happens. That matters: an opponent whose prompt were rendered
by the remote side would silently drift from the one its adapter was trained on.

THE FAILURE MODE THIS FILE EXISTS TO PREVENT
    ``lm/agent.py`` ends every scoring attempt with ``except Exception: return policy.act(obs)``
    -- correct for a Kaggle submission, which must never forfeit, and wrong for a measurement.
    If the tunnel drops mid-gate, every remote decision quietly becomes engine_v2 while the log
    still says the opponent was `reg`. That is the same class of silent substitution that cost
    a round on 08-12 (six rules shipped inert, nothing errored).

    So: a single blip degrades one decision to engine_v2 and is COUNTED and printed. Once
    `max_consecutive_failures` in a row fail, the scorer raises ``SystemExit`` -- a
    BaseException, which ``except Exception`` does not catch -- and takes the worker down. A
    dead server stops the run instead of changing the pilot.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("PTCG_SCORE_URL", "http://127.0.0.1:8077")


class RemoteScorerDown(SystemExit):
    """SystemExit on purpose: lm/agent.py catches Exception, and a dead server must not be
    absorbed into an engine_v2 fallback that no log would ever mention."""


class RemoteScorer:
    """The lm/agent scorer contract -- ``score(prompt, cands, obs=None) -> [float]`` -- served
    over HTTP. Stateless per call, so any number of game workers may share one server."""

    def __init__(self, url=None, adapter=None, timeout=180.0, retries=3,
                 max_consecutive_failures=8, token=None, label=""):
        self.url = (url or DEFAULT_URL).rstrip("/")
        self.adapter = adapter or ""
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.max_fail = int(max_consecutive_failures)
        self.token = token or os.environ.get("PTCG_SCORE_TOKEN", "")
        self.label = label or self.adapter or self.url
        self.n = 0                  # decisions scored (the name mirror_match reports on)
        self.t = 0.0                # seconds spent inside score()
        self.net = 0.0              # of which round-trip
        self.fallbacks = 0          # decisions that degraded to engine_v2
        self._streak = 0
        self._lock = threading.Lock()

    # -- transport ---------------------------------------------------------------------
    def _post(self, path, payload):
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.url + path, data=body,
                                     headers={"Content-Type": "application/json"})
        if self.token:
            req.add_header("X-Score-Token", self.token)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def health(self):
        return self._post("/health", {})

    def reset_bank(self):
        """Called by lm/agent at deck selection. There is no time bank here (the bank exists to
        keep a Kaggle submission inside its 600 s), but a new game is the right moment to prove
        the link is alive: failing HERE names the problem, while failing mid-game only shows up
        as a win rate."""
        try:
            self.health()
            self._streak = 0
        except Exception as e:                                          # noqa: BLE001
            raise RemoteScorerDown("score server %s unreachable at game start: %s"
                                   % (self.url, e))

    # -- the contract ------------------------------------------------------------------
    def score(self, prompt, cands, obs=None):
        if not prompt.startswith("[ACT]"):
            # Same guard as QwenScorer: the tag is part of the trained format, and prepending
            # it silently would hide a serializer change behind a merely-lower win rate.
            raise ValueError("prompt does not start with [ACT]")
        t0 = time.time()
        payload = {"adapter": self.adapter, "prompt": prompt, "cands": list(cands)}
        last = None
        for attempt in range(self.retries + 1):
            try:
                out = self._post("/score", payload)
                sc = out.get("scores")
                if not isinstance(sc, list) or len(sc) != len(cands):
                    raise ValueError("server returned %s scores for %d candidates"
                                     % (len(sc) if isinstance(sc, list) else type(sc).__name__,
                                        len(cands)))
                with self._lock:
                    self.n += 1
                    self.t += time.time() - t0
                    self.net += float(out.get("wire", 0.0)) or 0.0
                    self._streak = 0
                return [float(x) for x in sc]
            except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as e:
                last = e
                if attempt < self.retries:
                    time.sleep(0.4 * (attempt + 1))     # a restarting server, not a dead one
        with self._lock:
            self.fallbacks += 1
            self._streak += 1
            streak = self._streak
        print("[remote] %s: scoring FAILED (%s); this decision falls back to engine_v2 "
              "[%d consecutive, %d total]" % (self.label, last, streak, self.fallbacks),
              file=sys.stderr, flush=True)
        if streak >= self.max_fail:
            raise RemoteScorerDown(
                "score server %s failed %d times in a row (%s). Stopping: continuing would "
                "silently play engine_v2 while the log says %s."
                % (self.url, streak, last, self.adapter or "reg"))
        raise RuntimeError("remote scoring failed: %s" % last)   # caught -> one engine_v2 move

    def stats(self):
        return {"decisions": self.n, "seconds": round(self.t, 1),
                "ms_per_decision": round(1000.0 * self.t / max(1, self.n), 1),
                "fallbacks": self.fallbacks}


def parse_spec(path):
    """``remote:<url>[|<adapter>]`` -> RemoteScorer.

    The adapter is named after the '|' rather than in the URL because mirror_match splits a
    spec on its FIRST colon, so the rest must survive containing "http://host:port" intact.
    """
    url, _, adapter = path.partition("|")
    return RemoteScorer(url=url or DEFAULT_URL, adapter=adapter or None)
