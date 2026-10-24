// LOB Strategy Lab: C++20 reference Avellaneda-Stoikov market-making simulator.
// Written for the hackathon (new work). Same model as sim_py.py, but fast and deterministic.
//
// Contract: params JSON on stdin (flat object, numbers only, plus "seed")
//           metrics JSON on stdout.
// Build:  g++ -O2 -std=c++20 main.cpp -o lob_sim.exe
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr double S0 = 100.0, SIGMA = 2.0, T = 1.0;
constexpr double A = 1500.0, K = 1.5;
constexpr double P_INFORMED = 0.25, JUMP = 0.35;
constexpr int MARKOUT_STEPS = 20, VOL_WINDOW = 50, STEPS = 20000, BLOCK = 100;
constexpr double PI = 3.14159265358979323846;

struct Params { double spread_k, gamma, vol_widen; int inv_limit, size; uint64_t seed; };
struct Metrics { double pnl, sharpe, max_inventory, fill_rate, avg_markout; };

// Own RNG transforms so results are identical across compilers/platforms.
struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t s) : g(s) {}
    double uni() { return (g() >> 11) * (1.0 / 9007199254740992.0); }
    double gauss() {
        double u1 = uni(), u2 = uni();
        if (u1 < 1e-300) u1 = 1e-300;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * PI * u2);
    }
};

Metrics simulate(const Params& p) {
    Rng rng(p.seed);
    const double dt = T / STEPS, sq = std::sqrt(dt);
    double s = S0, cash = 0.0, eq_prev = 0.0, markout_sum = 0.0;
    long q = 0, fills = 0, markout_n = 0, max_inv = 0;
    std::deque<double> win;
    struct Pend { int step, side; double px; };
    std::deque<Pend> pending;
    std::vector<double> rets;
    rets.reserve(STEPS);
    const double base_half = (1.0 / p.gamma) * std::log(1.0 + p.gamma / K);

    for (int i = 0; i < STEPS; ++i) {
        double tau = std::max(T - i * dt, 1e-6);
        double r = s - q * p.gamma * SIGMA * SIGMA * tau;
        double half = (p.gamma * SIGMA * SIGMA * tau / 2.0 + base_half) * p.spread_k;
        if ((int)win.size() == VOL_WINDOW) {
            double m = 0; for (double x : win) m += x; m /= VOL_WINDOW;
            double v = 0; for (double x : win) v += (x - m) * (x - m);
            double realized = std::sqrt(v / VOL_WINDOW) / sq;
            if (realized > 1.5 * SIGMA) half *= p.vol_widen;
        }
        half = std::max(half, 0.01);
        double bid = r - half, ask = r + half;
        double d_bid = std::max(s - bid, 0.0), d_ask = std::max(ask - s, 0.0);

        if (q < p.inv_limit && rng.uni() < 1 - std::exp(-A * std::exp(-K * d_bid) * dt)) {
            cash -= bid * p.size; q += p.size; ++fills;
            pending.push_back({i, +1, bid});
            if (rng.uni() < P_INFORMED) s -= JUMP;
        }
        if (q > -p.inv_limit && rng.uni() < 1 - std::exp(-A * std::exp(-K * d_ask) * dt)) {
            cash += ask * p.size; q -= p.size; ++fills;
            pending.push_back({i, -1, ask});
            if (rng.uni() < P_INFORMED) s += JUMP;
        }
        double ds = SIGMA * sq * rng.gauss();
        s += ds;
        win.push_back(ds);
        if ((int)win.size() > VOL_WINDOW) win.pop_front();
        max_inv = std::max(max_inv, std::labs(q));

        while (!pending.empty() && i - pending.front().step >= MARKOUT_STEPS) {
            auto pd = pending.front(); pending.pop_front();
            markout_sum += pd.side * (s - pd.px); ++markout_n;
        }
        double eq = cash + q * s;
        rets.push_back(eq - eq_prev);
        eq_prev = eq;
    }
    std::vector<double> blocks;
    for (int j = 0; j + BLOCK <= (int)rets.size(); j += BLOCK) {
        double b = 0; for (int k = 0; k < BLOCK; ++k) b += rets[j + k];
        blocks.push_back(b);
    }
    double mu = 0; for (double b : blocks) mu += b; mu /= blocks.size();
    double var = 0; for (double b : blocks) var += (b - mu) * (b - mu); var /= blocks.size();
    double sd = std::sqrt(var); if (sd < 1e-9) sd = 1e-9;
    return {cash + q * s, mu / sd * std::sqrt((double)blocks.size()), (double)max_inv,
            (double)fills / STEPS, markout_n ? markout_sum / markout_n : 0.0};
}

// Minimal parser for a flat {"key": number, ...} object. Fails loudly on anything else.
std::map<std::string, double> parse_flat_json(const std::string& t) {
    std::map<std::string, double> out;
    size_t i = 0;
    auto fail = [&](const char* m) { std::fprintf(stderr, "bad input JSON: %s\n", m); std::exit(2); };
    while (i < t.size()) {
        size_t a = t.find('"', i);
        if (a == std::string::npos) break;
        size_t b = t.find('"', a + 1);
        if (b == std::string::npos) fail("unterminated key");
        std::string key = t.substr(a + 1, b - a - 1);
        size_t c = t.find(':', b);
        if (c == std::string::npos) fail("missing colon");
        char* end = nullptr;
        double v = std::strtod(t.c_str() + c + 1, &end);
        if (end == t.c_str() + c + 1) fail("value is not a number");
        out[key] = v;
        i = end - t.c_str();
    }
    return out;
}
}  // namespace

int main() {
    std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    auto j = parse_flat_json(in);
    for (const char* k : {"spread_k", "gamma", "inventory_limit", "quote_size", "vol_widen_factor", "seed"})
        if (!j.count(k)) { std::fprintf(stderr, "missing key: %s\n", k); return 2; }
    if (j["gamma"] <= 0 || j["spread_k"] <= 0) { std::fprintf(stderr, "gamma and spread_k must be > 0\n"); return 2; }
    Params p{j["spread_k"], j["gamma"], j["vol_widen_factor"], (int)j["inventory_limit"], (int)j["quote_size"],
             (uint64_t)j["seed"]};
    Metrics m = simulate(p);
    std::printf("{\"pnl\": %.6f, \"sharpe\": %.6f, \"max_inventory\": %.0f, \"fill_rate\": %.6f, \"avg_markout\": %.6f}\n",
                m.pnl, m.sharpe, m.max_inventory, m.fill_rate, m.avg_markout);
    return 0;
}