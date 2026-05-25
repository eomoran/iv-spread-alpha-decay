# IV Spread Alpha Decay Test

This repository tests whether the expected-return relation in Atilgan, Bali, and Demirtas (2015), "Implied Volatility Spreads and Expected Market Returns", still appears in newer S&P 500 option data.

The raw option-chain CSV is intentionally excluded from Git because it is large. Share `OptionDataNM2026.csv` separately if someone needs to reproduce the analysis exactly.

## Contents

- `analyze_iv_spread.py` - end-to-end analysis script.
- `requirements.txt` - Python package dependencies.
- `outputs/iv_spread_decay_report.html` - visual report with results.
- `outputs/workflow_summary.html` - visual methodology summary without results.
- `outputs/iv_spread_class_presentation.pptx` - short 5-10 minute class presentation deck.
- `outputs/iv_spread_results.json` - machine-readable result summary.
- `outputs/daily_iv_spread_returns.csv` - daily signal and return panel.
- `outputs/oos_strategy.csv` - out-of-sample strategy backtest panel.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the raw option data file at:

```text
OptionDataNM2026.csv
```

Expected columns include:

```text
Date, SpotPrice, Expiration, Maturity, ForwardPrice, Strike, Dividend,
CallBid, CallOffer, CallMid, IV_Call, PutBid, PutOffer, PutMid, IV_Put,
RiskFreeRate
```

## Run

```bash
source .venv/bin/activate
python analyze_iv_spread.py
```

This writes the report and analysis outputs into `outputs/`.

To view the HTML reports locally:

```bash
python -m http.server 8899 --bind 127.0.0.1 -d outputs
```

Then open:

- `http://127.0.0.1:8899/iv_spread_decay_report.html`
- `http://127.0.0.1:8899/workflow_summary.html`

## Method Summary

The paper-style implied-volatility spread is:

```text
VS_t = average IV of OTM puts - average IV of ATM calls
```

This implementation uses:

- OTM puts: strike/spot between 0.80 and 0.95.
- ATM calls: strike/spot between 0.95 and 1.05.
- Maturity filter: 10 to 60 calendar days.
- IV filter: 3% to 200%.
- Option mid-price filter: at least $0.125.
- Excess returns: log index returns less the dataset risk-free rate.
- Main multi-day horizon tests: non-overlapping forward returns.

Because the file does not include option volume or open interest, the original paper's volume/open-interest weighting variants cannot be replicated exactly. The report includes inverse percentage bid-ask weighting as a quote-quality proxy, but that is not equivalent to demand or liquidity measured by trading activity.

## Reproducibility Notes

- The local filename says `2026`, but the observed data in the file used for the report span 2017-01-03 through 2021-12-31.
- The strategy backtest is an expanding-window timing rule. It uses only prior data to estimate the forecast equation at each step.
- Cash earns the dataset `RiskFreeRate / 252`.
- The net strategy subtracts 1bp on allocation switches.

## Git Sharing

Recommended Git contents:

- Commit code, requirements, README, workflow summary, result report, and compact output tables.
- Exclude `.venv/`, `.DS_Store`, local server logs, and the raw option CSV.
- Share the raw data separately through a storage link if needed.
