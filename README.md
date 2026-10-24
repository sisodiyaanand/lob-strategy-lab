\# LOB Strategy Lab



An agent loop where an LLM proposes market-making strategy parameters from a natural-language

hypothesis, a simulator evaluates them, and the model revises using the results.

The simulator is the ground truth. The LLM only proposes.



Built for the Meta Global AI Developer Hackathon (Oct 26 - Nov 4, 2026), Indie track.

Everything in this repo was written during the hackathon period.



\## Status

\- \[x] C++20 Avellaneda-Stoikov simulator (`main.cpp`), deterministic, JSON in/out

\- \[x] Python reference simulator (`sim\_py.py`)

\- \[x] Loop, validation, logging, held-out evaluation (`lob\_cli\_lab.py`)

\- \[ ] Meta Model API (Muse Spark) integration

\- \[ ] Prompt-robustness study

\- \[ ] Results table and write-up



\## Run

&#x20;   g++ -O2 -std=c++20 main.cpp -o lob\_sim.exe

&#x20;   python lob\_cli\_lab.py -H "widen spreads when inventory is skewed and volatility spikes" --mock-llm -b lob\_sim.exe -n 3



`--mock-llm` uses a heuristic stand-in (every line is tagged \[MOCK]); it will be replaced by the model.



\## Evaluation protocol

Strategies are chosen on training seeds and reported on held-out seeds, against a default

Avellaneda-Stoikov baseline and random search with the same budget.

