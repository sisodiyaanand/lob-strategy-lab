#!/usr/bin/env python3
"""LOB Strategy Lab: hypothesis -> LLM proposes params -> simulator -> metrics -> LLM revises -> repeat.

Rules of this file:
- Nothing is faked silently. Missing simulator/LLM fails loudly.
- Mock backends exist only behind --mock-llm / --mock-sim and every line they print is tagged [MOCK].
- Round 0 is the default Avellaneda-Stoikov baseline; the model sees it and starts from there.
- Strategies are chosen on TRAIN seeds and reported on HELD-OUT seeds (no cherry-picking).
- Comparators with the SAME evaluation budget: local hill-climb from baseline, and global random search.
- Every round is logged to runs/<timestamp>.jsonl.

Simulator backends (--sim):
  python : sim_py.py reference Avellaneda-Stoikov simulator (default; NOT the C++ engine)
  binary : C++ simulator via -b PATH. Contract: params JSON (+ "seed") on stdin -> metrics JSON on stdout
  mock   : random toy numbers, plumbing test only
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
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
PROMPT_NOTE = ""  # extra line appended to the prompt (used when the model repeats an earlier point)


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
PARAM_HELP = (
    "spread_k: multiplier on the quoted half-spread (higher = wider quotes, fewer fills). "
    "gamma: risk aversion (higher = quotes skew harder against current inventory). "
    "inventory_limit: max absolute inventory before quoting on one side stops. "
    "quote_size: units per quote (larger = more risk per fill). "
    "vol_widen_factor: extra spread multiplier applied when realized volatility spikes."
)


def build_prompt(hypothesis: str, history: list) -> str:
    """Prompt v3: parameter meanings, baseline as round 0, best-so-far anchor, small-step rule."""
    schema = {k: {"min": v[0], "max": v[1], "default": v[2]} for k, v in SCHEMA.items()}
    past = [{"round": h["round"], "params": h["params"], "train_metrics": h["metrics"]} for h in history]
    best = max(history, key=lambda h: h["metrics"]["sharpe"]) if history else None
    best_txt = (json.dumps({"round": best["round"], "params": best["params"], "train_metrics": best["metrics"]})
                if best else "none yet")
    return (
        "You tune a market-making strategy (Avellaneda-Stoikov style) by trial and error.\n"
        f"Trader hypothesis: {hypothesis}\n"
        f"Parameter meanings: {PARAM_HELP}\n"
        f"Parameter schema (respect bounds): {json.dumps(schema)}\n"
        "Round 0 is the default baseline. Rounds so far (higher sharpe and avg_markout is better; "
        f"max_inventory should stay bounded): {json.dumps(past)}\n"
        f"Best round so far (highest sharpe): {best_txt}\n"
        "Rules: change at most two parameters per round, and change them modestly. If the last round made sharpe "
        "worse than the best round, move back toward the best round's parameters instead of continuing in the same "
        "direction. Propose a point you have not tried before. Do not claim an improvement that the numbers do "
        "not support.\n"
        + (PROMPT_NOTE + "\n" if PROMPT_NOTE else "")
        + 'Reply with ONLY a JSON object: {"params": {...}, "reasoning": "<one or two sentences>", '
        '"predict_sharpe": "up" or "down" (will sharpe be higher or lower than in the previous round listed)}'
    )


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))


def _http_json(url: str, payload: dict, headers: dict = None, timeout: int = 180) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _propose_via_chat(call, hypothesis: str, history: list, retries: int = 3) -> tuple:
    """Ask a chat model for {"params": {...}, "reasoning": "..."}; retry on invalid replies.
    Params are clamped by validate() afterwards, so a bad model reply can never reach the simulator."""
    prompt = build_prompt(hypothesis, history)
    last_err = None
    for attempt in range(retries):
        text = call(prompt if attempt == 0 else
                    prompt + f"\nYour previous reply was invalid ({last_err}). Reply with ONLY the JSON object.")
        try:
            obj = _extract_json(text)
            params, reasoning = obj["params"], str(obj.get("reasoning", ""))
            if not isinstance(params, dict):
                raise ValueError("params is not an object")
            pred = str(obj.get("predict_sharpe") or "").strip().lower()
            return params, reasoning, (pred if pred in ("up", "down") else None)
        except Exception as e:  # noqa: BLE001 - any parse problem triggers a retry
            last_err = str(e)
            print(f"[llm] invalid reply (attempt {attempt + 1}/{retries}): {last_err}")
    raise RuntimeError(f"model returned no valid JSON after {retries} attempts")


def llm_ollama(hypothesis: str, history: list) -> tuple:
    """Local open-weights model through Ollama (e.g. Meta Llama). Env: OLLAMA_MODEL, OLLAMA_URL."""
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    base = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    def call(prompt: str) -> str:
        out = _http_json(f"{base}/api/chat", {
            "model": model, "stream": False, "format": "json", "options": {"temperature": 0.3},
            "messages": [{"role": "user", "content": prompt}]})
        return out["message"]["content"]

    print(f"[llm] ollama model: {model}")
    return _propose_via_chat(call, hypothesis, history)


def llm_meta(hypothesis: str, history: list) -> tuple:
    """Meta Model API, assuming an OpenAI-style /chat/completions endpoint.
    CHECK THIS against the real docs on Oct 26; adjust the URL/payload in call() if it differs.
    Env: META_API_BASE, META_API_KEY, META_MODEL."""
    base, key, model = (os.environ.get(k) for k in ("META_API_BASE", "META_API_KEY", "META_MODEL"))
    if not (base and key and model):
        raise NotImplementedError("Set META_API_BASE, META_API_KEY and META_MODEL after approval "
                                  "(Oct 26), or use --llm ollama / --llm mock.")

    def call(prompt: str) -> str:
        out = _http_json(f"{base.rstrip('/')}/chat/completions",
                         {"model": model, "temperature": 0.3,
                          "messages": [{"role": "user", "content": prompt}]},
                         {"Authorization": f"Bearer {key}"})
        return out["choices"][0]["message"]["content"]

    print(f"[llm] meta api model: {model}")
    return _propose_via_chat(call, hypothesis, history)


def llm_mock(hypothesis: str, history: list) -> tuple:
    print("[MOCK] llm: heuristic proposal, no model called")
    if len(history) <= 1:  # only the baseline so far
        p = defaults()
        p.update(vol_widen_factor=2.0, gamma=0.2)
        return p, "[MOCK] initial guess from hypothesis keywords", "up"
    last = history[-1]["params"]
    p = dict(last)
    p["spread_k"] = last["spread_k"] * 0.9
    return p, "[MOCK] tighten spread 10% to raise fill rate", "up"


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


def hill_climb(simulate, binary: str, train: list, seed: int, steps: int, base_sharpe: float) -> dict:
    """Comparator: local random search from the baseline (same budget as the LLM rounds)."""
    rng = random.Random(seed + 7)
    cur, cur_sc = defaults(), base_sharpe
    for _ in range(steps):
        cand = dict(cur)
        for k in rng.sample(list(SCHEMA), 2):
            cand[k] = cand[k] * rng.uniform(0.8, 1.25)
        cand = validate(cand)
        sc = evaluate(simulate, cand, binary, train)["sharpe"]
        if sc > cur_sc:
            cur, cur_sc = cand, sc
    return cur


def main():
    global PROMPT_NOTE
    ap = argparse.ArgumentParser(description="LOB Strategy Lab")
    ap.add_argument("-H", "--hypothesis", required=True)
    ap.add_argument("-b", "--binary", default="", help="path to C++ simulator (selects --sim binary)")
    ap.add_argument("--sim", choices=["python", "binary", "mock"], default="python")
    ap.add_argument("-n", "--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-seeds", type=int, default=3)
    ap.add_argument("--test-seeds", type=int, default=5)
    ap.add_argument("--llm", choices=["meta", "ollama", "mock"], default="meta")
    ap.add_argument("--mock-llm", action="store_true", help="alias for --llm mock")
    ap.add_argument("--mock-sim", action="store_true", help="alias for --sim mock")
    args = ap.parse_args()
    if args.mock_llm:
        args.llm = "mock"

    sim_name = "mock" if args.mock_sim else ("binary" if args.binary else args.sim)
    if sim_name == "binary" and not args.binary:
        sys.exit("--sim binary needs -b PATH")
    simulate = {"python": sim_python, "binary": sim_binary, "mock": sim_mock}[sim_name]
    propose = {"mock": llm_mock, "ollama": llm_ollama, "meta": llm_meta}[args.llm]
    tag = "[MOCK] " if sim_name == "mock" else ""
    print(f"{tag}simulator: {sim_name}" + ("  (reference Python A-S sim, not the C++ engine)" if sim_name == "python" else ""))

    train = [args.seed + i for i in range(args.train_seeds)]
    test = [args.seed + 1000 + i for i in range(args.test_seeds)]  # held-out

    Path("runs").mkdir(exist_ok=True)
    log_path = Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    def log(entry: dict):
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    # Round 0: default baseline, shown to the model as the starting point.
    base_metrics = evaluate(simulate, defaults(), args.binary, train)
    base_entry = {"round": 0, "hypothesis": args.hypothesis, "params": defaults(),
                  "reasoning": "baseline (default A-S parameters)", "predicted_sharpe": None,
                  "metrics": base_metrics, "sim": sim_name}
    log(base_entry)
    print(f"\n== Round 0 (baseline) ==\ntrain    : {fmt(base_metrics)}")

    history = [base_entry]  # what the model sees; includes the baseline
    for rnd in range(1, args.rounds + 1):
        PROMPT_NOTE = ""
        for attempt in range(3):
            raw, reasoning, predicted = propose(args.hypothesis, history)
            params = validate(raw)
            repeat_of = next((h["round"] for h in history if h["params"] == params), None)
            if repeat_of is None or attempt == 2:
                break
            print(f"[llm] repeated the point from round {repeat_of}; asking for a different one")
            PROMPT_NOTE = (f"Your last proposal was identical to round {repeat_of}. "
                           "You must propose different parameter values.")
        PROMPT_NOTE = ""
        metrics = evaluate(simulate, params, args.binary, train)
        entry = {"round": rnd, "hypothesis": args.hypothesis, "params": params, "reasoning": reasoning,
                 "predicted_sharpe": predicted, "repeated": repeat_of is not None,
                 "metrics": metrics, "sim": sim_name}
        history.append(entry)
        log(entry)
        print(f"\n== Round {rnd} ==\nparams   : {json.dumps(params)}\nreasoning: {reasoning}\ntrain    : {fmt(metrics)}")

    llm_rounds = history[1:]

    # Calibration: did the model's "sharpe goes up/down" claim match the simulator?
    scored = []
    for prev, cur in zip(history, history[1:]):
        if cur["predicted_sharpe"] is not None and cur["metrics"]["sharpe"] != prev["metrics"]["sharpe"]:
            actual = "up" if cur["metrics"]["sharpe"] > prev["metrics"]["sharpe"] else "down"
            scored.append(cur["predicted_sharpe"] == actual)
    n_rep = sum(1 for e in llm_rounds if e.get("repeated"))
    if scored:
        print(f"\n{tag}Calibration: model's sharpe prediction correct {sum(scored)}/{len(scored)} rounds"
              f" (repeated points: {n_rep})")
    else:
        print(f"\nCalibration: no usable predictions from the model (repeated points: {n_rep})")

    # Fair comparison: same evaluation budget for every method; choose on train seeds, report on held-out seeds.
    best_llm = max(llm_rounds, key=lambda e: e["metrics"]["sharpe"])["params"]
    hc_best = hill_climb(simulate, args.binary, train, args.seed, args.rounds, base_metrics["sharpe"])
    rng = random.Random(args.seed)
    rand_best, rand_score = None, float("-inf")
    for _ in range(args.rounds):
        cand = validate({k: rng.uniform(v[0], v[1]) for k, v in SCHEMA.items()})
        sc = evaluate(simulate, cand, args.binary, train)["sharpe"]
        if sc > rand_score:
            rand_best, rand_score = cand, sc

    print(f"\n{tag}== Held-out comparison ({len(test)} unseen seeds) ==")
    rows = [("LLM-best", best_llm), ("hill-climb", hc_best), ("random-search", rand_best),
            ("baseline A-S", defaults())]
    for name, p in rows:
        print(f"{name:14s}: {fmt(evaluate(simulate, p, args.binary, test))}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    sys.exit(main())