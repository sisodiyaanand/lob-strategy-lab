#!/usr/bin/env python3
"""Build a self-contained HTML report from a runs/*.jsonl log.

Usage:  python report.py                 (uses the newest file in runs/)
        python report.py runs\\file.jsonl
Writes report.html next to this script. Shows only what is in the log; nothing is invented.
"""
import glob
import html
import json
import os
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    rounds = [e for e in entries if "round" in e]
    summary = next((e for e in reversed(entries) if "summary" in e), None)
    return rounds, summary


def chart(rounds):
    W, H, L, R, T, B = 680, 280, 56, 20, 20, 40
    ys = [r["metrics"]["sharpe"] for r in rounds]
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.15 or 1.0
    lo, hi = lo - pad, hi + pad
    n = max(len(rounds) - 1, 1)

    def X(i):
        return L + (W - L - R) * i / n

    def Y(v):
        return T + (H - T - B) * (hi - v) / (hi - lo)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Train Sharpe by round">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
        out.append(f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" class="ax" text-anchor="end">{v:.1f}</text>')
    if lo < 0 < hi:
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>')
    base = ys[0]
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(base):.1f}" y2="{Y(base):.1f}" class="base"/>')
    out.append(f'<text x="{W - R}" y="{Y(base) - 6:.1f}" class="ax" text-anchor="end">baseline {base:.2f}</text>')
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ys))
    out.append(f'<polyline points="{pts}" class="line"/>')
    for i, (r, v) in enumerate(zip(rounds, ys)):
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4.5" class="pt"><title>round {r["round"]}: '
                   f'sharpe {v:.2f}</title></circle>')
        out.append(f'<text x="{X(i):.1f}" y="{H - 16}" class="ax" text-anchor="middle">{r["round"]}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 2}" class="ax" text-anchor="middle">round (0 = baseline)</text>')
    out.append("</svg>")
    return "".join(out)


def esc(x):
    return html.escape(str(x))


def build(rounds, summary):
    first = rounds[0]
    rows = ""
    for r in rounds:
        m = r["metrics"]
        p = r["params"]
        pred = r.get("predicted_sharpe") or "-"
        flag = " (repeat)" if r.get("repeated") else ""
        rows += (f"<tr><td>{r['round']}</td><td class='mono'>spread {p['spread_k']:.2f} / gamma {p['gamma']:.2f} / "
                 f"inv {p['inventory_limit']} / size {p['quote_size']} / vol {p['vol_widen_factor']:.2f}</td>"
                 f"<td>{m['sharpe']:.2f}</td><td>{m['pnl']:.0f}</td><td>{m['fill_rate']:.3f}</td>"
                 f"<td>{esc(pred)}{flag}</td><td class='why'>{esc(r.get('reasoning', ''))}</td></tr>")
    sm = ""
    if summary:
        for name, v in summary["summary"].items():
            m = v["metrics"]
            sm += (f"<tr><td>{esc(name)}</td><td>{m['sharpe']:.2f}</td><td>{m['pnl']:.0f}</td>"
                   f"<td>{m['max_inventory']:.0f}</td><td>{m['fill_rate']:.3f}</td><td>{m['avg_markout']:.3f}</td></tr>")
        sm = ("<h2>Held-out comparison</h2><p class='note'>Chosen on training seeds, reported on "
              f"{summary.get('heldout_seeds', '?')} unseen seeds. Same evaluation budget for every method.</p>"
              "<table><tr><th>Method</th><th>Sharpe</th><th>PnL</th><th>Max inv</th><th>Fill rate</th>"
              f"<th>Markout</th></tr>{sm}</table>")
    llm = esc(summary.get("llm", "?")) if summary else "?"
    sim = esc(first.get("sim", "?"))
    is_mock = (summary and summary.get("llm") == "mock") or first.get("sim") == "mock" \
        or any("[MOCK]" in str(r.get("reasoning", "")) for r in rounds)
    mock_banner = ('<p class="mock">TEST DATA: this run used a mock heuristic instead of a real model '
                   '(and/or a mock simulator). Not a result.</p>') if is_mock else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LOB Strategy Lab report</title>
<style>
:root{{--bg:#fafaf7;--fg:#1c1c1a;--mut:#6b6b66;--card:#fff;--line:#e4e3dc;--acc:#2a5bd7;--base:#b0562a}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151513;--fg:#ecebe5;--mut:#9a9992;--card:#1e1e1b;--line:#33332e;--acc:#7da2ff;--base:#e08a5e}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:900px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:18px;margin:32px 0 8px}}
.sub{{color:var(--mut);margin:0 0 20px}} .note{{color:var(--mut);font-size:13px;margin:0 0 8px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}} th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--mut);font-weight:600}} .mono{{font-family:ui-monospace,Consolas,monospace;font-size:12px}} .why{{color:var(--mut);min-width:220px}}
svg{{width:100%;height:auto}} .grid{{stroke:var(--line)}} .zero{{stroke:var(--mut);stroke-dasharray:2 3}}
.base{{stroke:var(--base);stroke-dasharray:6 4}} .line{{fill:none;stroke:var(--acc);stroke-width:2.5}}
.pt{{fill:var(--acc)}} .ax{{fill:var(--mut);font-size:11px}}
.mock{{background:#fff3cd;color:#664d03;border:1px solid #e6c85a;border-radius:8px;padding:8px 12px;font-weight:600;margin:0 0 12px}}
</style></head><body><main>
<h1>LOB Strategy Lab</h1>
{mock_banner}
<p class="sub">Hypothesis: {esc(first.get('hypothesis', ''))}<br>Simulator: {sim} &middot; Model backend: {llm}</p>
<h2>Train Sharpe by round</h2><div class="card">{chart(rounds)}</div>
<h2>Rounds</h2><div class="card"><table><tr><th>#</th><th>Parameters</th><th>Sharpe</th><th>PnL</th><th>Fill rate</th>
<th>Predicted</th><th>Model's reasoning</th></tr>{rows}</table></div>
{sm}
</main></body></html>"""


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        files = sorted(glob.glob(os.path.join("runs", "*.jsonl")), key=os.path.getmtime)
        if not files:
            sys.exit("no runs/*.jsonl found; run lob_cli_lab.py first")
        path = files[-1]
    rounds, summary = load(path)
    if not rounds:
        sys.exit(f"no rounds in {path}")
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(build(rounds, summary))
    print(f"wrote report.html from {path}" + ("" if summary else "  (no held-out summary in this log)"))


if __name__ == "__main__":
    main()