"""Reference Avellaneda-Stoikov market-making simulator (pure Python, seeded).

This is NOT the user's C++ engine. It exists so the full loop (LLM -> params -> sim -> metrics)
runs end to end today. The C++ engine plugs in through the same params/metrics contract.

Model: mid follows a Brownian motion; our bid/ask fills arrive as Poisson with
intensity A*exp(-k*delta) (delta = distance from mid). A fraction of fills are "informed":
the mid then jumps against us. Equity is marked to mid.
"""
import math
import random
from collections import deque

S0, SIGMA, T = 100.0, 2.0, 1.0
A, K = 1500.0, 1.5
P_INFORMED, JUMP = 0.25, 0.35
MARKOUT_STEPS = 20
VOL_WINDOW = 50


def simulate(params: dict, seed: int = 42, steps: int = 20000) -> dict:
    rng = random.Random(seed)
    dt = T / steps
    sq = math.sqrt(dt)

    spread_k = params["spread_k"]
    gamma = params["gamma"]
    inv_limit = params["inventory_limit"]
    size = params["quote_size"]
    vol_widen = params["vol_widen_factor"]

    s, q, cash = S0, 0, 0.0
    equity_prev = 0.0
    returns = []
    rets_window = deque(maxlen=VOL_WINDOW)
    pending = deque()  # (step, side, price, size)
    markouts, fills, max_inv = [], 0, 0

    base_half = (1.0 / gamma) * math.log(1.0 + gamma / K)  # A-S half of the optimal spread term

    for i in range(steps):
        tau = max(T - i * dt, 1e-6)
        r = s - q * gamma * SIGMA ** 2 * tau  # reservation price
        half = (gamma * SIGMA ** 2 * tau / 2.0 + base_half) * spread_k

        if len(rets_window) == VOL_WINDOW:
            m = sum(rets_window) / VOL_WINDOW
            realized = math.sqrt(sum((x - m) ** 2 for x in rets_window) / VOL_WINDOW) / sq
            if realized > 1.5 * SIGMA:
                half *= vol_widen

        half = max(half, 0.01)
        bid, ask = r - half, r + half
        d_bid, d_ask = max(s - bid, 0.0), max(ask - s, 0.0)

        if q < inv_limit and rng.random() < 1 - math.exp(-A * math.exp(-K * d_bid) * dt):
            cash -= bid * size
            q += size
            fills += 1
            pending.append((i, +1, bid, size))
            if rng.random() < P_INFORMED:
                s -= JUMP
        if q > -inv_limit and rng.random() < 1 - math.exp(-A * math.exp(-K * d_ask) * dt):
            cash += ask * size
            q -= size
            fills += 1
            pending.append((i, -1, ask, size))
            if rng.random() < P_INFORMED:
                s += JUMP

        ds = SIGMA * sq * rng.gauss(0.0, 1.0)
        s += ds
        rets_window.append(ds)
        max_inv = max(max_inv, abs(q))

        while pending and i - pending[0][0] >= MARKOUT_STEPS:
            _, side, px, _sz = pending.popleft()
            markouts.append(side * (s - px))  # buy: mid - price, sell: price - mid

        equity = cash + q * s
        returns.append(equity - equity_prev)
        equity_prev = equity

    # Sharpe on 100-step blocks (per-run, unitless; compare across strategies, not to real-world Sharpe)
    block = 100
    blocks = [sum(returns[j:j + block]) for j in range(0, len(returns) - block + 1, block)]
    mu = sum(blocks) / len(blocks)
    sd = math.sqrt(sum((b - mu) ** 2 for b in blocks) / len(blocks)) or 1e-9
    return {
        "pnl": cash + q * s,
        "sharpe": mu / sd * math.sqrt(len(blocks)),
        "max_inventory": max_inv,
        "fill_rate": fills / steps,
        "avg_markout": (sum(markouts) / len(markouts)) if markouts else 0.0,
    }
