# Professor Agent Instructions

Use these instructions in a local agent or coding assistant instance to reproduce the implied-volatility spread analysis.

## Objective

Test whether the expected-return relation in Atilgan, Bali, and Demirtas (2015), "Implied Volatility Spreads and Expected Market Returns", still appears in newer S&P 500 option data.

The core paper-style signal is:

```text
VS_t = average IV of OTM puts - average IV of ATM calls
```

with:

```text
OTM puts: 0.80 <= Strike / Spot <= 0.95
ATM calls: 0.95 <= Strike / Spot <= 1.05
```

## Files Needed

The repository contains the analysis code, workflow summary, and generated report artifacts.

The raw option data is intentionally not committed to Git. Place the data file in the repository root with this exact name:

```text
OptionDataNM2026.csv
```

Expected CSV columns:

```text
Date, SpotPrice, Expiration, Maturity, ForwardPrice, Strike, Dividend,
CallBid, CallOffer, CallMid, IV_Call, PutBid, PutOffer, PutMid, IV_Put,
RiskFreeRate
```

## Setup Commands

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the Analysis

```bash
source .venv/bin/activate
python analyze_iv_spread.py
```

This writes outputs into:

```text
outputs/
```

Key generated files:

```text
outputs/iv_spread_decay_report.html
outputs/workflow_summary.html
outputs/iv_spread_results.json
outputs/daily_iv_spread_returns.csv
outputs/oos_strategy.csv
```

## View the Reports

Start a local static server:

```bash
python -m http.server 8899 --bind 127.0.0.1 -d outputs
```

Then open:

```text
http://127.0.0.1:8899/iv_spread_decay_report.html
http://127.0.0.1:8899/workflow_summary.html
```

The workflow summary is methodology-only. The decay report includes results.

## Methodology Checklist

The script performs the following steps:

1. Load option-chain data.
2. Filter to usable quotes:
   - 10 to 60 calendar days to expiry.
   - IV between 3% and 200%.
   - Option mid-price at least $0.125.
3. Construct paper-style spread:
   - OTM put IV minus ATM call IV.
4. Construct robustness variants:
   - inverse percentage bid-ask weighted spread;
   - delta-bucket OTM put minus ATM call spread;
   - ATM call-minus-put signal;
   - ATM call-put gated timing rules;
   - cash-threshold timing rules.
5. Compute risk-free-adjusted excess returns:
   - index log return less `RiskFreeRate / 252`;
   - cash allocation earns `RiskFreeRate / 252`.
6. Estimate predictive regressions:
   - daily horizon with Newey-West style inference;
   - multi-day horizons with non-overlapping forward returns.
7. Run an expanding-window out-of-sample timing strategy:
   - fit forecast equation only on prior data;
   - invest in index when forecast excess return is above the threshold;
   - otherwise hold risk-free cash;
   - subtract 1bp on allocation switches in the net strategy.

## Important Limitations

- The data file used here does not contain option volume or open interest.
- Because of that, the paper's volume/open-interest-weighted spread variants cannot be exactly replicated.
- Bid-ask weighting is used only as a quote-quality/liquidity proxy.
- Threshold and gate variants are exploratory unless locked before a future out-of-sample test.
- The current script tests index timing, not a self-financing option portfolio.

## Suggested Agent Prompt

If using an agent, paste this:

```text
You are working in a repository that analyzes whether the implied-volatility spread result from Atilgan, Bali, and Demirtas (2015) still holds in newer S&P 500 option data.

First read README.md and PROFESSOR_AGENT_INSTRUCTIONS.md. Confirm that OptionDataNM2026.csv exists in the repository root. Create a virtual environment if needed, install requirements.txt, run python analyze_iv_spread.py, and open the generated HTML reports in outputs/.

When reviewing results, pay special attention to:
- whether the paper-style OTM put IV minus ATM call IV signal remains negative and significant;
- whether multi-day inference uses non-overlapping returns;
- whether risk-free cash and excess returns use the RiskFreeRate column;
- whether any strategy improvement survives transaction costs and is not just threshold tuning;
- the limitation that volume and open interest are unavailable.
```

