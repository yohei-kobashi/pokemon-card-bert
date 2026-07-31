"""Evaluate a trained cross-encoder reranker as the game pilot: plug it into lm/agent as a
drop-in scorer and play vs the heuristic engine_v2, reporting WIN RATE on the SAME protocol
used for the decoder LMs (so numbers compare directly to LFM2 / Qwen3.5).

RerankerScoringModel.score(prompt, candidates, obs) matches the lm/agent contract: it strips
the "[ACT]\n" prefix (training states carry none), scores each (state, candidate) pair with
the reranker head (scalar logit), returns the per-candidate logits -> agent takes argmax.

Usage:
  python tools/eval_rerank.py --adapter /root/out/rerank_gte --games 30 \
    --decks mega_lucario,alakazam_nz_fez,crustle_stall --opp alakazam,crustle,dragapult
  python tools/eval_rerank.py --adapter /root/out/rerank_gte --games 60 \
    --decks crustle,alakazam,dragapult            # mirror (opp empty)
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

_ACT = "[ACT]\n"


class RerankerScoringModel:
    def __init__(self, model, tok, max_len=1024, device="cuda", batch=48):
        import torch
        self.torch = torch
        self.model = model.eval()
        self.tok = tok
        # Same direction as the deploy scorer: an overflowing state loses its HEAD, never the
        # board and option menu at the tail. Currently inert (max pair = 584 tokens under the
        # glossary='none' format) but the two paths must not disagree on the rule.
        self.tok.truncation_side = "left"
        self.max_len = max_len
        self.device = device
        self.batch = batch

    def score(self, prompt, candidates, obs=None):
        torch = self.torch
        state = prompt[len(_ACT):] if prompt.startswith(_ACT) else prompt
        out = []
        for i in range(0, len(candidates), self.batch):
            chunk = candidates[i:i + self.batch]
            enc = self.tok([[state, c] for c in chunk], padding=True, truncation="only_first",
                           max_length=self.max_len, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.squeeze(-1)
            out.extend(float(x) for x in logits.reshape(-1))
        return out


# The ONNX deploy path lives in lm/rerank_scorer.py (a deploy component, shipped in the
# bundle) so eval and the submission run the SAME code. Measuring anything else risks
# validating a scorer we do not ship. Note that argmax-agreement vs PyTorch is a poor proxy
# for play strength -- the fp32 model's top1-vs-top2 logit gap is only ~0.8 median -- so
# win rate through this exact class is the metric that counts.


def main():
    import arena, library
    from battle_log import load_agent
    from lm.agent import make_lm_agent

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="trained reranker dir (tokenizer always "
                    "read from here; PyTorch weights read from here unless --onnx is given)")
    ap.add_argument("--onnx", default="", help="score through this ONNX model on CPU instead "
                    "of PyTorch/CUDA -- measures the real deploy path")
    ap.add_argument("--threads", type=int, default=4, help="--onnx: ORT intra-op threads "
                    "(4 = the competition runtime's vCPU count)")
    ap.add_argument("--remap", default="", help="vocab_remap.npy for a vocab-pruned --onnx")
    ap.add_argument("--time-budget", type=float, default=480.0, help="--onnx: per-game "
                    "thinking-bank seconds before the scorer raises and the agent falls "
                    "back to engine_v2 (set low to TEST the fallback)")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--decks", default="mega_lucario,alakazam_nz_fez,crustle_stall")
    ap.add_argument("--opp", default="", help="comma opponents; empty = mirror")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--glossary", default="full", choices=("full", "structured", "none"),
                    help="must match how the training data was rendered")
    ap.add_argument("--no-deck-glossary", dest="deck_glossary", action="store_false",
                    help="visible-only glossary -- what the rerank data was ACTUALLY built "
                         "with (build_rerank always passed an empty deck list)")
    ap.add_argument("--deck-mode", default="static", choices=("static", "remaining"),
                    help="must match build_rerank --deck-mode")
    ap.add_argument("--deck-shuffle", action="store_true",
                    help="must match build_rerank --deck-shuffle")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.onnx:
        from lm.rerank_scorer import OnnxRerankerScorer
        sm = OnnxRerankerScorer(args.onnx, args.adapter, max_len=args.max_len,
                                threads=args.threads, remap=args.remap or None,
                                time_budget=args.time_budget)
        tag = (f"onnx:{os.path.basename(args.onnx)} threads={args.threads} "
               f"bank={args.time_budget:.0f}s")
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.adapter)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.adapter, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
        sm = RerankerScoringModel(model, tok, max_len=args.max_len)
        tag = args.adapter
    tun = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))

    decks = args.decks.split(",")
    opps = args.opp.split(",") if args.opp else None
    results = {}
    print(f"=== reranker ({tag}) vs heuristic engine_v2 ===", flush=True)
    for d in decks:
        dl = library.read_deck(d)
        # deck_name is NOT optional: build_rerank renders ``ID ME d_x a_y`` from it, so
        # omitting it here deletes a segment the model was trained on -- exactly the kind of
        # silent train/deploy prompt divergence the make_lm_agent docstring warns about.
        lm_agent = make_lm_agent(dl, profile=tun.get(d, {}), model=sm, deck_name=d,
                                 glossary=args.glossary, deck_glossary=args.deck_glossary,
                                 deck_mode=args.deck_mode, deck_shuffle=args.deck_shuffle)
        for opp in (opps or [d]):
            ol = library.read_deck(opp); oa = load_agent(opp)
            w = n = 0
            banks = []          # seconds the scorer burned per game (reset at deck selection,
            for g in range(args.games):     # so reading it AFTER the game gives that game)
                mine = g % 2
                r = (arena.play(lm_agent, oa, dl, ol) if mine == 0
                     else arena.play(oa, lm_agent, ol, dl))
                n += 1; w += (r == mine)
                if getattr(sm, "spent", None) is not None:
                    banks.append((sm.spent, sm.n_decisions))
            wr = 100.0 * w / max(1, n)
            cell = {"win": w, "games": n, "win_rate": wr}
            line = f"  {d:20} (RR) vs {opp:20} (heuristic): {w}/{n} = {wr:.1f}%"
            if banks:
                sp = sorted(x[0] for x in banks)
                hit = sum(1 for x in sp if x >= args.time_budget)
                cell.update(bank_mean_s=sum(sp) / len(sp), bank_p90_s=sp[int(0.9 * (len(sp) - 1))],
                            bank_max_s=sp[-1], games_hitting_budget=hit,
                            mean_scored_decisions=sum(x[1] for x in banks) / len(banks))
                line += (f"  | bank mean {cell['bank_mean_s']:.0f}s p90 {cell['bank_p90_s']:.0f}s "
                         f"max {cell['bank_max_s']:.0f}s, {hit}/{n} games fell back")
            results[f"{d} (RR) vs {opp} (heuristic)"] = cell
            print(line, flush=True)
    ov = list(results.values())
    overall = 100.0 * sum(v["win"] for v in ov) / max(1, sum(v["games"] for v in ov))
    print(f"OVERALL reranker win rate vs heuristic: {overall:.1f}%", flush=True)
    if args.out:
        json.dump({"results": results, "overall_win_rate": overall}, open(args.out, "w"))


if __name__ == "__main__":
    main()
