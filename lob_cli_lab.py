#!/usr/bin/env python3
"""LOB Strategy Lab: hypothesis -> LLM proposes params -> simulator -> metrics -> LLM revises -> repeat.

Rules of this file:
- Nothing is faked silently. Missing simulator/LLM fails loudly.
- Mock backends exist only behind --mock-llm / --mock-sim and every line they print is tagged [MOCK].
- Strategies are chosen on TRAIN seeds and reported on HELD-OUT seeds (no cherry-picking).
- Baseline (default Avellaneda-Stoikov) and random search with the same budget are always reported.
- Every round is logged to runs/<timestamp>.jsonl.

Simulator backends (--sim):
  python : sim_py.py reference Avellaneda-Stoikov simulator (default; NOT the C++ engine)
  binary : your C++ engine via -b PATH. Contract: params JSON (+ "seed") on stdin -> metrics JSON on stdout
  mock   : random toy numbers, plumbing test only
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

# name: (min, max, default)
SCHEMA = {
    "spread_k": (0.5, 5.0, 1.0),
    "gamma": (0.001, 1.0, 0.1),
    "inventory_limit": (5, 200, 50),
    "quote_size": (1, 50, 5),
    "vol_widen_factor": (1.0, 4.0, 1.5),
}
METRIC_KEYS = ["pnl", "sharpe", "max_inventory", "fill_rate", "avg_markout"]
INT_KEYS = ("inventory_limit", "quote_size")


def defaults() -> dict:
    return {k: v[2] for k, v in SCHEMA.items()}


def validate(params: dict) -> dict:
    """Keep only known keys, fill defaults, clamp to bounds."""
    clean = {}
    for k, (lo, hi, default) in SCHEMA.items():
        v = float(params.get(k, default))
        clean[k] = min(max(v, lo), hi)
    for k in INT_KEYS:
        clean[k] = int(round(clean[k]))
    return clean


# ---------- LLM backends ----------
def build_prompt(hypothesis: str, history: list) -> str:
    """Prompt for the model. Ready for the Meta Model API; no API call happens here."""
    schema = {k: {"min": v[0], "max": v[1], "default": v[2]} for k, v in SCHEMA.items()}
    past = [{"round": h["round"], "params": h["params"], "train_metrics": h["metrics"]} for h in history]
    return (
        "You tune a market-making strategy (Avellaneda-Stoikov style).\n"
        f"Trader hypothesis: {hypothesis}\n"
        f"Parameter schema (respect bounds): {json.dumps(schema)}\n"
        f"Previous rounds (higher sharpe and avg_markout is better; max_inventory should stay bounded): {json.dumps(past)}\n"
        'Reply with ONLY a JSON object: {"params": {...}, "reasoning": "<one or two sentences>"}'
    )


def llm_meta(hypothesis: str, history: list) -> tuple:
    """TODO: call Meta Model API (Muse Spark) after approval on Oct 26 with build_prompt(...),
    parse JSON, return (params_dict, reasoning_str)."""
    _ = build_prompt(hypothesis, history)
    raise NotImplementedError("Meta Model API not wired yet. Use --mock-llm for local testing.")


def llm_mock(hypothesis: str, history: list) -> tuple:
    print("[MOCK] llm: heuristic proposal, no model called")
    if not history:
        p = defaults()
        p.update(vol_widen_factor=2.0, gamma=0.2)
        return p, "[MOCK] initial guess from hypothesis keywords"
    last = history[-1]["params"]
    p = dict(last)
    p["spread_k"] = last["spread_k"] * 0.9
    return p, "[MOCK] tighten spread 10% to raise fill rate"


# ---------- Simulator backends ----------
def sim_python(params: dict, binary: str, seed: int) -> dict:
    from sim_py import simulate
    return simulate(params, seed=seed)


def sim_binary(params: dict, binary: str, seed: int) -> dict:
    payload = json.dumps({**params, "seed": seed})
    r = subprocess.run([binary], input=payload, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"simulator failed ({r.returncode}): {r.stderr.strip()}")
    return json.loads(r.stdout)


def sim_mock(params: dict, binary: str, seed: int) -> dict:
    rng = random.Random(seed * 31 + int(params["spread_k"] * 100))
    return {
        "pnl": rng.uniform(-100, 500),
        "sharpe": rng.uniform(-0.5, 2.5),
        "max_inventory": rng.randint(5, params["inventory_limit"]),
        "fill_rate": rng.uniform(0.2, 0.9),
        "avg_markout": rng.uniform(-0.3, 0.1),
    }


def evaluate(simulate, params: dict, binary: str, seeds: list) -> dict:
    """Average metrics over several seeds (reduces luck)."""
    runs = [simulate(params, binary, s) for s in seeds]
    return {k: sum(r[k] for r in runs) / len(runs) for k in METRIC_KEYS}


def fmt(m: dict) -> str:
    return (f"pnl={m['pnl']:.1f} sharpe={m['sharpe']:.2f} max_inv={m['max_inventory']:.0f} "
            f"fill_rate={m['fill_rate']:.3f} markout={m['avg_markout']:.3f}")


def main():
    ap = argparse.ArgumentParser(description="LOB Strategy Lab")
    ap.add_argument("-H", "--hypothesis", required=True)
    ap.add_argument("-b", "--binary", default="", help="path to C++ engine (selects --sim binary)")
    ap.add_argument("--sim", choices=["python", "binary", "mock"], default="python")
    ap.add_argument("-n", "--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-seeds", type=int, default=3)
    ap.add_argument("--test-seeds", type=int, default=5)
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--mock-sim", action="store_true", help="alias for --sim mock")
    args = ap.parse_args()

    sim_name = "mock" if args.mock_sim else ("binary" if args.binary else args.sim)
    if sim_name == "binary" and not args.binary:
        sys.exit("--sim binary needs -b PATH")
    simulate = {"python": sim_python, "binary": sim_binary, "mock": sim_mock}[sim_name]
    propose = llm_mock if args.mock_llm else llm_meta
    tag = "[MOCK] " if sim_name == "mock" else ""
    print(f"{tag}simulator: {sim_name}" + ("  (reference Python A-S sim, not the C++ engine)" if sim_name == "python" else ""))

    train = [args.seed + i for i in range(args.train_seeds)]
    test = [args.seed + 1000 + i for i in range(args.test_seeds)]  # held-out

    Path("runs").mkdir(exist_ok=True)
    log_path = Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    history = []

    for rnd in range(1, args.rounds + 1):
        raw, reasoning = propose(args.hypothesis, history)
        params = validate(raw)
        metrics = evaluate(simulate, params, args.binary, train)
        entry = {"round": rnd, "hypothesis": args.hypothesis, "params": params,
                 "reasoning": reasoning, "metrics": metrics, "sim": sim_name}
        history.append(entry)
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"\n== Round {rnd} ==\nparams   : {json.dumps(params)}\nreasoning: {reasoning}\ntrain    : {fmt(metrics)}")

    # Fair comparison: same eval budget for random search, pick on train seeds, report on held-out seeds.
    best_llm = max(history, key=lambda e: e["metrics"]["sharpe"])["params"]
    rng = random.Random(args.seed)
    rand_best, rand_score = None, float("-inf")
    for _ in range(args.rounds):
        cand = validate({k: rng.uniform(v[0], v[1]) for k, v in SCHEMA.items()})
        sc = evaluate(simulate, cand, args.binary, train)["sharpe"]
        if sc > rand_score:
            rand_best, rand_score = cand, sc

    print(f"\n{tag}== Held-out comparison ({len(test)} unseen seeds) ==")
    for name, p in [("LLM-best", best_llm), ("random-search", rand_best), ("baseline A-S", defaults())]:
        print(f"{name:14s}: {fmt(evaluate(simulate, p, args.binary, test))}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    sys.exit(main())