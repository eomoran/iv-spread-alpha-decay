from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from scipy.stats import norm


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "OptionDataNM2026.csv"
OUT_DIR = ROOT / "outputs"
REPORT_PATH = OUT_DIR / "iv_spread_decay_report.html"
RESULTS_PATH = OUT_DIR / "iv_spread_results.json"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = values.astype(float)
    weights = weights.astype(float).replace([np.inf, -np.inf], np.nan)
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def newey_west_regression(y: pd.Series, x: pd.Series, lags: int) -> dict:
    frame = pd.concat({"y": y, "x": x}, axis=1).dropna()
    if len(frame) < 30:
        return {
            "n": len(frame),
            "alpha": np.nan,
            "beta": np.nan,
            "t": np.nan,
            "p": np.nan,
            "r2": np.nan,
        }
    model = sm.OLS(frame["y"], sm.add_constant(frame["x"]))
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": max(0, int(lags))})
    return {
        "n": int(fit.nobs),
        "alpha": float(fit.params["const"]),
        "beta": float(fit.params["x"]),
        "t": float(fit.tvalues["x"]),
        "p": float(fit.pvalues["x"]),
        "r2": float(fit.rsquared),
        "impact_bps_per_vol_point": float(fit.params["x"] * 0.01 * 10000),
    }


def simple_ols_regression(y: pd.Series, x: pd.Series) -> dict:
    frame = pd.concat({"y": y, "x": x}, axis=1).dropna()
    if len(frame) < 5:
        return {"n": len(frame), "alpha": np.nan, "beta": np.nan, "t": np.nan, "p": np.nan, "r2": np.nan}
    yv = frame["y"].to_numpy()
    xv = frame["x"].to_numpy()
    design = np.column_stack([np.ones(len(frame)), xv])
    params = np.linalg.lstsq(design, yv, rcond=None)[0]
    resid = yv - design @ params
    dof = len(frame) - 2
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.inv(design.T @ design)
    beta_se = float(np.sqrt(cov[1, 1]))
    t_stat = float(params[1] / beta_se) if beta_se else np.nan
    # Normal approximation is adequate here for a compact robustness diagnostic.
    p_value = float(math.erfc(abs(t_stat) / np.sqrt(2))) if np.isfinite(t_stat) else np.nan
    sst = float((yv - yv.mean()) @ (yv - yv.mean()))
    r2 = float(1.0 - (resid @ resid) / sst) if sst else np.nan
    return {
        "n": int(len(frame)),
        "alpha": float(params[0]),
        "beta": float(params[1]),
        "t": t_stat,
        "p": p_value,
        "r2": r2,
        "impact_bps_per_vol_point": float(params[1] * 0.01 * 10000),
    }


def make_signal(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["moneyness"] = d["Strike"] / d["SpotPrice"]
    d["call_spread_pct"] = (d["CallOffer"] - d["CallBid"]) / d["CallMid"].replace(0, np.nan)
    d["put_spread_pct"] = (d["PutOffer"] - d["PutBid"]) / d["PutMid"].replace(0, np.nan)
    d["years_to_expiry"] = d["Maturity"] / 365.0
    sqrt_t = np.sqrt(d["years_to_expiry"].clip(lower=1e-8))
    carry_call = (d["RiskFreeRate"] - d["Dividend"] + 0.5 * d["IV_Call"] ** 2) * d["years_to_expiry"]
    carry_put = (d["RiskFreeRate"] - d["Dividend"] + 0.5 * d["IV_Put"] ** 2) * d["years_to_expiry"]
    call_d1 = (np.log(d["SpotPrice"] / d["Strike"]) + carry_call) / (d["IV_Call"] * sqrt_t)
    put_d1 = (np.log(d["SpotPrice"] / d["Strike"]) + carry_put) / (d["IV_Put"] * sqrt_t)
    discount_div = np.exp(-d["Dividend"] * d["years_to_expiry"])
    d["call_delta"] = discount_div * norm.cdf(call_d1)
    d["put_delta"] = discount_div * (norm.cdf(put_d1) - 1.0)

    base = d[(d["Maturity"].between(10, 60))].copy()
    put = base[
        base["moneyness"].between(0.80, 0.95)
        & base["IV_Put"].between(0.03, 2.00)
        & (base["PutMid"] >= 0.125)
    ].copy()
    call = base[
        base["moneyness"].between(0.95, 1.05)
        & base["IV_Call"].between(0.03, 2.00)
        & (base["CallMid"] >= 0.125)
    ].copy()
    atm_put = base[
        base["moneyness"].between(0.95, 1.05)
        & base["IV_Put"].between(0.03, 2.00)
        & (base["PutMid"] >= 0.125)
    ].copy()
    delta_put = base[
        base["put_delta"].abs().between(0.10, 0.40)
        & base["IV_Put"].between(0.03, 2.00)
        & (base["PutMid"] >= 0.125)
    ].copy()
    delta_call = base[
        base["call_delta"].between(0.45, 0.55)
        & base["IV_Call"].between(0.03, 2.00)
        & (base["CallMid"] >= 0.125)
    ].copy()

    put["q_weight"] = 1.0 / put["put_spread_pct"].clip(lower=0.0001)
    call["q_weight"] = 1.0 / call["call_spread_pct"].clip(lower=0.0001)
    atm_put["q_weight"] = 1.0 / atm_put["put_spread_pct"].clip(lower=0.0001)

    put_ew = put.groupby("Date")["IV_Put"].mean().rename("put_iv_ew")
    call_ew = call.groupby("Date")["IV_Call"].mean().rename("call_iv_ew")
    atm_put_ew = atm_put.groupby("Date")["IV_Put"].mean().rename("atm_put_iv_ew")
    delta_put_ew = delta_put.groupby("Date")["IV_Put"].mean().rename("delta_put_iv_ew")
    delta_call_ew = delta_call.groupby("Date")["IV_Call"].mean().rename("delta_call_iv_ew")
    put_qw = put.groupby("Date").apply(
        lambda g: weighted_mean(g["IV_Put"], g["q_weight"]), include_groups=False
    ).rename("put_iv_qw")
    call_qw = call.groupby("Date").apply(
        lambda g: weighted_mean(g["IV_Call"], g["q_weight"]), include_groups=False
    ).rename("call_iv_qw")
    atm_put_qw = atm_put.groupby("Date").apply(
        lambda g: weighted_mean(g["IV_Put"], g["q_weight"]), include_groups=False
    ).rename("atm_put_iv_qw")
    put_count = put.groupby("Date").size().rename("n_put")
    call_count = call.groupby("Date").size().rename("n_call")
    delta_put_count = delta_put.groupby("Date").size().rename("n_delta_put")
    delta_call_count = delta_call.groupby("Date").size().rename("n_delta_call")

    daily = (
        d.groupby("Date", as_index=True)
        .agg(
            spot=("SpotPrice", "first"),
            risk_free=("RiskFreeRate", "first"),
            raw_rows=("Strike", "size"),
        )
        .join(
            [
                put_ew,
                call_ew,
                atm_put_ew,
                delta_put_ew,
                delta_call_ew,
                put_qw,
                call_qw,
                atm_put_qw,
                put_count,
                call_count,
                delta_put_count,
                delta_call_count,
            ],
            how="left",
        )
        .sort_index()
    )
    daily["vs_ew"] = daily["put_iv_ew"] - daily["call_iv_ew"]
    daily["vs_qw"] = daily["put_iv_qw"] - daily["call_iv_qw"]
    daily["vs_delta_ew"] = daily["delta_put_iv_ew"] - daily["delta_call_iv_ew"]
    daily["atm_cp_ew"] = daily["call_iv_ew"] - daily["atm_put_iv_ew"]
    daily["atm_cp_qw"] = daily["call_iv_qw"] - daily["atm_put_iv_qw"]
    daily["log_ret_1d"] = np.log(daily["spot"].shift(-1) / daily["spot"])
    daily["rf_1d"] = daily["risk_free"] / 252.0
    daily["excess_1d"] = daily["log_ret_1d"] - daily["rf_1d"]
    for h in [5, 10, 21]:
        daily[f"log_ret_{h}d"] = np.log(daily["spot"].shift(-h) / daily["spot"])
        daily[f"rf_{h}d"] = daily["risk_free"].rolling(h, min_periods=h).sum().shift(-(h - 1)) / 252.0
        daily[f"excess_{h}d"] = daily[f"log_ret_{h}d"] - daily[f"rf_{h}d"]
    daily["next_year"] = daily.index.year
    return daily


def strategy_backtest(
    daily: pd.DataFrame, signal_col: str = "vs_ew", cash_threshold_bps: float = 0.0
) -> pd.DataFrame:
    frame = daily[[signal_col, "excess_1d", "log_ret_1d", "rf_1d"]].dropna().copy()
    start = len(frame) // 2
    forecasts = []
    dates = []
    for i in range(start, len(frame) - 1):
        train = frame.iloc[:i]
        fit = sm.OLS(train["excess_1d"], sm.add_constant(train[signal_col])).fit()
        pred = float(fit.params["const"] + fit.params[signal_col] * frame.iloc[i][signal_col])
        forecasts.append(pred)
        dates.append(frame.index[i])
    bt = frame.loc[dates].copy()
    bt["forecast_excess"] = forecasts
    threshold_return = -cash_threshold_bps / 10000.0
    bt["invested"] = (bt["forecast_excess"] > threshold_return).astype(int)
    bt["strategy_log_ret_gross"] = np.where(
        bt["invested"].eq(1), bt["log_ret_1d"], bt["rf_1d"]
    )
    bt["switch"] = bt["invested"].diff().abs().fillna(0)
    bt["strategy_log_ret_net"] = bt["strategy_log_ret_gross"] - bt["switch"] * 0.0001
    bt["buy_hold_log_ret"] = bt["log_ret_1d"]
    bt["cash_log_ret"] = bt["rf_1d"]
    for col in ["strategy_log_ret_gross", "strategy_log_ret_net", "buy_hold_log_ret", "cash_log_ret"]:
        bt[f"{col}_growth"] = 100 * np.exp(bt[col].cumsum())
    return bt


def threshold_sensitivity(daily: pd.DataFrame, signal_col: str = "vs_ew") -> list[dict]:
    rows = []
    for threshold_bps in [0, 1, 2.5, 5, 10, 15, 25]:
        bt = strategy_backtest(daily, signal_col, threshold_bps)
        stats = ann_stats(bt["strategy_log_ret_net"], bt["cash_log_ret"])
        rows.append(
            {
                "threshold_bps": threshold_bps,
                "cash_days": int((bt["invested"] == 0).sum()),
                "switches": int(bt["switch"].sum()),
                "final_growth": float(bt["strategy_log_ret_net_growth"].iloc[-1]),
                "ann_return": stats["ann_return"],
                "ann_vol": stats["ann_vol"],
                "excess_sharpe": stats["excess_sharpe"],
            }
        )
    return rows


def gated_strategy_backtest(
    daily: pd.DataFrame,
    signal_col: str = "vs_ew",
    gate_col: str = "atm_cp_ew",
    cash_threshold_bps: float = 0.0,
    gate_cutoff: float = 0.0,
) -> pd.DataFrame:
    frame = daily[[signal_col, gate_col, "excess_1d", "log_ret_1d", "rf_1d"]].dropna().copy()
    start = len(frame) // 2
    forecasts = []
    dates = []
    for i in range(start, len(frame) - 1):
        train = frame.iloc[:i]
        fit = sm.OLS(train["excess_1d"], sm.add_constant(train[signal_col])).fit()
        pred = float(fit.params["const"] + fit.params[signal_col] * frame.iloc[i][signal_col])
        forecasts.append(pred)
        dates.append(frame.index[i])
    bt = frame.loc[dates].copy()
    bt["forecast_excess"] = forecasts
    threshold_return = -cash_threshold_bps / 10000.0
    bearish_paper_signal = bt["forecast_excess"] <= threshold_return
    bullish_atm_signal = bt[gate_col] > gate_cutoff
    bt["invested"] = (~bearish_paper_signal | bullish_atm_signal).astype(int)
    bt["strategy_log_ret_gross"] = np.where(
        bt["invested"].eq(1), bt["log_ret_1d"], bt["rf_1d"]
    )
    bt["switch"] = bt["invested"].diff().abs().fillna(0)
    bt["strategy_log_ret_net"] = bt["strategy_log_ret_gross"] - bt["switch"] * 0.0001
    bt["buy_hold_log_ret"] = bt["log_ret_1d"]
    bt["cash_log_ret"] = bt["rf_1d"]
    for col in ["strategy_log_ret_gross", "strategy_log_ret_net", "buy_hold_log_ret", "cash_log_ret"]:
        bt[f"{col}_growth"] = 100 * np.exp(bt[col].cumsum())
    return bt


def gate_sensitivity(daily: pd.DataFrame) -> list[dict]:
    cases = [
        ("Baseline sign rule", "none", 0.0, None),
        ("ATM CP gate", "atm_cp_ew", 0.0, 0.0),
        ("ATM CP gate + 10bp threshold", "atm_cp_ew", 10.0, 0.0),
        ("ATM CP gate + 15bp threshold", "atm_cp_ew", 15.0, 0.0),
    ]
    rows = []
    for label, gate_col, threshold_bps, cutoff in cases:
        if gate_col == "none":
            bt = strategy_backtest(daily, "vs_ew", threshold_bps)
        else:
            bt = gated_strategy_backtest(daily, "vs_ew", gate_col, threshold_bps, cutoff or 0.0)
        stats = ann_stats(bt["strategy_log_ret_net"], bt["cash_log_ret"])
        rows.append(
            {
                "rule": label,
                "cash_days": int((bt["invested"] == 0).sum()),
                "switches": int(bt["switch"].sum()),
                "final_growth": float(bt["strategy_log_ret_net_growth"].iloc[-1]),
                "ann_return": stats["ann_return"],
                "ann_vol": stats["ann_vol"],
                "excess_sharpe": stats["excess_sharpe"],
            }
        )
    return rows


def ann_stats(returns: pd.Series, rf_returns: pd.Series | None = None) -> dict:
    if rf_returns is None:
        frame = pd.DataFrame({"r": returns}).dropna()
        frame["rf"] = 0.0
    else:
        frame = pd.concat({"r": returns, "rf": rf_returns}, axis=1).dropna()
    r = frame["r"]
    if r.empty:
        return {"mean_daily_bps": np.nan, "ann_return": np.nan, "ann_vol": np.nan, "excess_sharpe": None}
    simple = np.exp(r) - 1
    rf_simple = np.exp(frame["rf"]) - 1
    excess = simple - rf_simple
    ann_ret = float(np.exp(r.mean() * 252) - 1)
    ann_vol = float(simple.std(ddof=1) * np.sqrt(252))
    excess_vol = float(excess.std(ddof=1))
    return {
        "mean_daily_bps": float(simple.mean() * 10000),
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "excess_sharpe": float(excess.mean() / excess_vol * np.sqrt(252)) if excess_vol > 1e-12 else None,
    }


def plot_to_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def drawdown(growth: pd.Series) -> pd.Series:
    return growth / growth.cummax() - 1.0


def build_report(
    daily: pd.DataFrame,
    regressions: list[dict],
    bt: pd.DataFrame,
    stats: dict,
    threshold_rows: list[dict],
    gate_rows: list[dict],
) -> str:
    valid = daily.dropna(subset=["vs_ew", "excess_1d"]).copy()
    split_stats = []
    for label, start, end in [
        ("2017-2019", "2017-01-01", "2019-12-31"),
        ("2020-2021", "2020-01-01", "2021-12-31"),
    ]:
        sub = valid.loc[start:end]
        reg = newey_west_regression(sub["excess_1d"], sub["vs_ew"], 5)
        split_stats.append((label, reg))

    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Scatter(x=daily.index, y=daily["vs_ew"] * 100, name="EW IV spread", line=dict(color="#c2410c", width=1.6)), secondary_y=False)
    fig1.add_trace(go.Scatter(x=daily.index, y=daily["spot"], name="S&P 500 spot", line=dict(color="#334155", width=1.2)), secondary_y=True)
    fig1.update_layout(title="Put-minus-call implied volatility spread", template="plotly_white", height=380, margin=dict(l=40, r=40, t=55, b=35))
    fig1.update_yaxes(title_text="Volatility spread, vol points", secondary_y=False)
    fig1.update_yaxes(title_text="Spot index", secondary_y=True)

    fig2 = go.Figure()
    scatter_df = valid.copy()
    scatter_df["vs_bucket"] = pd.qcut(scatter_df["vs_ew"], 20, duplicates="drop")
    bucket = scatter_df.groupby("vs_bucket", observed=True).agg(vs=("vs_ew", "mean"), ret=("excess_1d", "mean"), n=("excess_1d", "size"))
    fig2.add_trace(go.Bar(x=bucket["vs"] * 100, y=bucket["ret"] * 10000, marker_color="#0f766e", name="Binned next-day excess return"))
    fig2.update_layout(title="Return by IV-spread bucket", xaxis_title="Average IV spread, vol points", yaxis_title="Next-day excess return, bps", template="plotly_white", height=330, margin=dict(l=45, r=25, t=55, b=45))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=bt.index, y=bt["strategy_log_ret_net_growth"], name="Forecast-timed strategy, net 1bp switches", line=dict(color="#0f766e", width=2)))
    fig3.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_log_ret_growth"], name="Buy and hold index", line=dict(color="#334155", width=2)))
    fig3.add_trace(go.Scatter(x=bt.index, y=bt["cash_log_ret_growth"], name="Risk-free cash", line=dict(color="#64748b", width=1.4, dash="dot")))
    fig3.update_layout(title="Out-of-sample expanding-window market timing", yaxis_title="$100 growth", template="plotly_white", height=380, margin=dict(l=45, r=25, t=55, b=35))

    threshold_backtests = {
        "Sign rule": strategy_backtest(daily, "vs_ew", 0),
        "10bp cash threshold": strategy_backtest(daily, "vs_ew", 10),
        "15bp cash threshold": strategy_backtest(daily, "vs_ew", 15),
        "25bp cash threshold": strategy_backtest(daily, "vs_ew", 25),
    }
    fig4 = go.Figure()
    threshold_colors = ["#0f766e", "#1d4ed8", "#9333ea", "#c2410c"]
    for (name, alt_bt), color in zip(threshold_backtests.items(), threshold_colors):
        fig4.add_trace(go.Scatter(x=alt_bt.index, y=alt_bt["strategy_log_ret_net_growth"], name=name, line=dict(color=color, width=1.9)))
    fig4.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_log_ret_growth"], name="Buy and hold", line=dict(color="#334155", width=2, dash="dot")))
    fig4.update_layout(title="Threshold strategy growth", yaxis_title="$100 growth", template="plotly_white", height=380, margin=dict(l=45, r=25, t=55, b=35))

    fig5 = go.Figure()
    for (name, alt_bt), color in zip(threshold_backtests.items(), threshold_colors):
        fig5.add_trace(go.Scatter(x=alt_bt.index, y=drawdown(alt_bt["strategy_log_ret_net_growth"]) * 100, name=name, line=dict(color=color, width=1.8)))
    fig5.add_trace(go.Scatter(x=bt.index, y=drawdown(bt["buy_hold_log_ret_growth"]) * 100, name="Buy and hold", line=dict(color="#334155", width=2, dash="dot")))
    fig5.update_layout(title="Threshold strategy drawdowns", yaxis_title="Drawdown, %", template="plotly_white", height=340, margin=dict(l=45, r=25, t=55, b=35))

    gate_backtests = {
        "ATM CP gate": gated_strategy_backtest(daily, "vs_ew", "atm_cp_ew", 0, 0),
        "ATM CP gate + 10bp": gated_strategy_backtest(daily, "vs_ew", "atm_cp_ew", 10, 0),
        "ATM CP gate + 15bp": gated_strategy_backtest(daily, "vs_ew", "atm_cp_ew", 15, 0),
    }
    fig6 = go.Figure()
    gate_colors = ["#0f766e", "#1d4ed8", "#c2410c"]
    for (name, alt_bt), color in zip(gate_backtests.items(), gate_colors):
        fig6.add_trace(go.Scatter(x=alt_bt.index, y=alt_bt["strategy_log_ret_net_growth"], name=name, line=dict(color=color, width=1.9)))
    fig6.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_log_ret_growth"], name="Buy and hold", line=dict(color="#334155", width=2, dash="dot")))
    fig6.update_layout(title="ATM call-put gated strategy growth", yaxis_title="$100 growth", template="plotly_white", height=380, margin=dict(l=45, r=25, t=55, b=35))

    rows = []
    signal_labels = {
        "vs_ew": "EW OTM put - ATM call",
        "vs_qw": "Inverse bid-ask weighted OTM put - ATM call",
        "vs_delta_ew": "Delta-bucket OTM put - ATM call",
        "atm_cp_ew": "ATM call - ATM put",
        "atm_cp_qw": "Inverse bid-ask weighted ATM call - ATM put",
    }
    for signal in ["vs_ew", "vs_qw", "vs_delta_ew", "atm_cp_ew", "atm_cp_qw"]:
        for horizon in [1, 5, 10, 21]:
            if horizon == 1:
                reg = next(
                    r for r in regressions if r["signal"] == signal and r["horizon"] == "1d"
                )
            else:
                sub = daily[[signal, f"excess_{horizon}d"]].dropna().iloc[::horizon]
                reg = simple_ols_regression(sub[f"excess_{horizon}d"], sub[signal])
            rows.append(
                f"<tr><td>{horizon}d</td><td>{signal_labels.get(signal, signal)}</td><td>{reg['n']:,}</td>"
                f"<td>{reg['beta']:.4f}</td><td>{reg['t']:.2f}</td><td>{reg['p']:.3f}</td>"
                f"<td>{reg['impact_bps_per_vol_point']:.2f}</td><td>{reg['r2']:.3%}</td></tr>"
            )

    split_rows = []
    for label, reg in split_stats:
        split_rows.append(
            f"<tr><td>{label}</td><td>{reg['n']:,}</td><td>{reg['beta']:.4f}</td>"
            f"<td>{reg['t']:.2f}</td><td>{reg['p']:.3f}</td><td>{reg['impact_bps_per_vol_point']:.2f}</td></tr>"
        )

    stat_cards = "".join(
        f"<div class='metric'><span>{name}</span><strong>{value}</strong></div>"
        for name, value in [
            ("Sample", f"{daily.index.min().date()} to {daily.index.max().date()}"),
            ("Trading days", f"{daily.index.nunique():,}"),
            ("Mean \\(IV^{p}_{OTM}-IV^{c}_{ATM}\\)", f"{valid['vs_ew'].mean()*100:.2f} vol pts"),
            ("1D beta impact", f"{regressions[0]['impact_bps_per_vol_point']:.2f} bps / +1 vol pt"),
            ("1D t-stat", f"{regressions[0]['t']:.2f}"),
            ("OOS net final $", f"{bt['strategy_log_ret_net_growth'].iloc[-1]:.2f}"),
        ]
    )

    def fmt_sharpe(value: float | None) -> str:
        return "n/a" if value is None or pd.isna(value) else f"{value:.2f}"

    strategy_rows = "".join(
        f"<tr><td>{label}</td><td>{s['mean_daily_bps']:.2f}</td><td>{s['ann_return']:.2%}</td><td>{s['ann_vol']:.2%}</td><td>{fmt_sharpe(s['excess_sharpe'])}</td></tr>"
        for label, s in stats.items()
    )

    threshold_table_rows = "".join(
        f"<tr><td>{row['threshold_bps']:g}</td><td>{row['cash_days']:,}</td><td>{row['switches']:,}</td>"
        f"<td>{row['final_growth']:.2f}</td><td>{row['ann_return']:.2%}</td><td>{row['ann_vol']:.2%}</td><td>{fmt_sharpe(row['excess_sharpe'])}</td></tr>"
        for row in threshold_rows
    )
    gate_table_rows = "".join(
        f"<tr><td>{row['rule']}</td><td>{row['cash_days']:,}</td><td>{row['switches']:,}</td>"
        f"<td>{row['final_growth']:.2f}</td><td>{row['ann_return']:.2%}</td><td>{row['ann_vol']:.2%}</td><td>{fmt_sharpe(row['excess_sharpe'])}</td></tr>"
        for row in gate_rows
    )

    paper_rows = "".join(
        [
            "<tr><td>Paper univariate 1-day beta</td><td>-0.0112 to -0.0295</td><td>Our EW: -0.1084; QW: -0.0655</td></tr>",
            "<tr><td>Paper 1-day t-stats</td><td>Statistically significant, about -2.7 to -4.1 in reported tables</td><td>Our EW: -1.55; QW: -1.01</td></tr>",
            "<tr><td>Paper market-timing growth</td><td>$100 grows to $158 gross; $151.6 after costs; S&P 500 grows to $129</td><td>Our $100 grows to $140.7 gross; $139.0 net; buy-and-hold grows to $159.8</td></tr>",
            "<tr><td>Paper daily strategy return</td><td>0.032% gross; 0.030% after bid-ask costs; passive 0.021%</td><td>Our net strategy: 0.0585%; buy-and-hold: 0.0862%</td></tr>",
        ]
    )

    verdict = (
        "The post-publication sample does not reproduce the paper's strong daily negative predictive relation. "
        "The coefficient is small and statistically weak in this dataset; the sign only appears intermittently by subperiod. "
        "The simple expanding-window timing rule also fails to dominate buy-and-hold after a 1bp switching cost; its cash leg uses the dataset's RiskFreeRate."
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IV Spread Alpha Decay Test</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']]
  }},
  chtml: {{ scale: 0.98 }},
  startup: {{ typeset: true }}
}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<script>{get_plotlyjs()}</script>
<style>
:root {{
  color-scheme: light;
  --ink:#172033; --muted:#64748b; --line:#d9e2ec; --bg:#f8fafc; --panel:#ffffff;
  --accent:#0f766e; --warm:#c2410c; --blue:#1d4ed8;
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }}
header {{ padding:34px 42px 20px; background:#ffffff; border-bottom:1px solid var(--line); }}
h1 {{ margin:0; font-size:clamp(28px, 4vw, 52px); letter-spacing:0; line-height:1.03; max-width:980px; }}
.sub {{ margin-top:14px; max-width:980px; font-size:17px; color:var(--muted); line-height:1.45; }}
main {{ padding:26px 42px 48px; max-width:1280px; margin:0 auto; }}
.metrics {{ display:grid; grid-template-columns:repeat(6, minmax(0, 1fr)); gap:12px; margin-bottom:20px; }}
.metric {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; min-height:86px; }}
.metric span {{ display:block; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
.metric strong {{ display:block; margin-top:9px; font-size:22px; line-height:1.1; }}
.grid {{ display:grid; grid-template-columns:1.2fr .8fr; gap:18px; align-items:start; }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; margin-bottom:18px; }}
h2 {{ margin:0 0 10px; font-size:19px; letter-spacing:0; }}
p {{ line-height:1.55; color:#334155; }}
.callout {{ border-left:4px solid var(--warm); padding:12px 14px; background:#fff7ed; color:#7c2d12; border-radius:6px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ padding:9px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
th {{ color:#475569; font-weight:650; background:#f8fafc; }}
.paper-table th, .paper-table td {{ white-space:normal; text-align:left; vertical-align:top; line-height:1.35; }}
.paper-table td:first-child {{ width:28%; }}
.paper-table td:nth-child(2), .paper-table td:nth-child(3) {{ width:36%; }}
.sensitivity-table {{ font-size:13px; }}
.sensitivity-table th, .sensitivity-table td {{ white-space:normal; }}
.note {{ color:var(--muted); font-size:13px; }}
.equations {{ display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:12px; }}
.equations div {{ background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:13px; min-height:96px; }}
.equations strong {{ display:block; margin-bottom:6px; }}
.equations mjx-container[display="true"] {{ margin:.35em 0 .2em; overflow-x:auto; overflow-y:hidden; }}
.method {{ display:grid; grid-template-columns:1fr; gap:12px; }}
.method div {{ background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:13px; }}
.method strong {{ display:block; margin-bottom:6px; }}
@media (max-width: 980px) {{
  header, main {{ padding-left:18px; padding-right:18px; }}
  .metrics {{ grid-template-columns:repeat(2, minmax(0,1fr)); }}
  .grid, .method, .equations {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body>
<header>
  <h1>Implied volatility spread alpha decay test</h1>
  <div class="sub">Post-publication S&P 500 option-chain test of Atilgan, Bali, and Demirtas (2015): OTM put IV minus ATM call IV as a predictor of future index returns.</div>
</header>
<main>
  <div class="metrics">{stat_cards}</div>
	  <section>
	    <h2>Bottom line</h2>
	    <p class="callout">{verdict}</p>
	    <p class="note">Data caveat: the input file name says 2026, but the actual data end on {daily.index.max().date()}. The test therefore covers 2017-2021 only.</p>
	  </section>
  <section>
    <h2>Model equations</h2>
    <div class="equations">
      <div><strong>Paper-style spread</strong>\\[VS_t=\\overline{{\\sigma}}^p_t(K/S\\in[0.80,0.95])-\\overline{{\\sigma}}^c_t(K/S\\in[0.95,1.05])\\]</div>
      <div><strong>Predictive regression</strong>\\[R^e_{{t+h}}=\\alpha+\\beta VS_t+\\varepsilon_{{t+h}},\\quad R^e_{{t+h}}=\\log(S_{{t+h}}/S_t)-\\sum_{{j=1}}^h r^f_{{t+j}}/252\\]</div>
      <div><strong>Bid-ask liquidity proxy</strong>\\[w_i=\\left(\\frac{{Ask_i-Bid_i}}{{Mid_i}}\\right)^{{-1}}\\]</div>
      <div><strong>Excess Sharpe</strong>\\[SR=\\sqrt{{252}}\\,\\frac{{\\mathbb{{E}}[r_t-r^f_t]}}{{\\operatorname{{sd}}(r_t-r^f_t)}}\\]</div>
      <div><strong>Estimated option delta</strong>\\[\\Delta_c=e^{{-qT}}\\Phi(d_1),\\quad \\Delta_p=e^{{-qT}}(\\Phi(d_1)-1)\\]</div>
      <div><strong>Timing rule</strong>\\[I_t=\\mathbf{{1}}\\left\\{{\\widehat{{R}}^e_{{t+1}}> -\\theta\\right\\}}\\]</div>
    </div>
  </section>
	  <div class="grid">
    <section>
      {plot_to_html(fig1)}
    </section>
    <section>
      <h2>Replication choices</h2>
      <div class="method">
	        <div><strong>Paper signal</strong>\\(VS_t=IV^p_{{OTM}}-IV^c_{{ATM}}\\). OTM puts use strike/spot \\(0.80\\le K/S\\le0.95\\); ATM calls use \\(0.95\\le K/S\\le1.05\\).</div>
	        <div><strong>Delta robustness</strong>Delta-bucket version estimates Black-Scholes \\(\\Delta\\), then uses \\(10\\)-\\(40\\) delta puts and \\(45\\)-\\(55\\) delta calls. This is not the paper's primary construction.</div>
	        <div><strong>ATM call-put gate</strong>\\(IV^c_{{ATM}}-IV^p_{{ATM}}\\) is tested separately. For the gate, a positive ATM call-minus-put signal keeps the strategy invested even when the paper-style forecast is bearish.</div>
	        <div><strong>Filters</strong>10-60 days to expiry, IV 3%-200%, option mid at least $0.125.</div>
	        <div><strong>Risk-free handling</strong>Excess returns subtract \\(r^f_t/252\\). The out-of-sample cash allocation earns \\(r^f_t/252\\).</div>
	        <div><strong>Bid-ask proxy</strong>Quote-quality-weighted spread means inverse percentage bid-ask weighting: \\(w_i=((Ask_i-Bid_i)/Mid_i)^{{-1}}\\), computed separately for OTM puts and ATM calls.</div>
        <div><strong>Dataset limit</strong>No volume or open interest, so the paper's HVVS, HOVS, VWVS and OWVS cannot be replicated exactly. The bid-ask proxy is a liquidity approximation, not a substitute for signed demand or actual trading activity.</div>
      </div>
    </section>
  </div>
  <div class="grid">
    <section>{plot_to_html(fig2)}</section>
    <section>
      <h2>Subperiod stability</h2>
      <table><thead><tr><th>Period</th><th>N</th><th>Beta</th><th>t-stat</th><th>p</th><th>bps / +1 vol pt</th></tr></thead><tbody>{''.join(split_rows)}</tbody></table>
      <p class="note">A negative beta matches the paper's direction. Statistical strength is Newey-West adjusted.</p>
    </section>
  </div>
  <section>
    <h2>Predictive regressions</h2>
    <table><thead><tr><th>Horizon</th><th>Signal</th><th>N</th><th>Beta</th><th>t-stat</th><th>p</th><th>bps / +1 vol pt</th><th>R²</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <p class="note">These rows use non-overlapping forward returns for each horizon, matching the paper's horizon construction more closely. Daily rows are unchanged because 1-day returns do not overlap.</p>
  </section>
  <section>
    <h2>Trading interpretation</h2>
    <p>Statistical significance is useful for guarding against noise, but it is not sufficient for a tradable strategy. Trading evidence also needs the signal to be executable after bid-ask costs, turnover, slippage, borrowing or financing, taxes, capacity limits, latency, and regime changes. A statistically weak signal can still be useful as a risk-control input if it reliably improves drawdowns after costs; a statistically significant signal can be useless if costs or instability consume it.</p>
    <p class="note">The alternative-strategy charts below focus on implementability: final growth, drawdowns, switching frequency, and risk-adjusted excess return.</p>
  </section>
  <section>
    <h2>Paper comparison</h2>
    <table class="paper-table"><thead><tr><th>Item</th><th>Atilgan et al. (2015)</th><th>This file, 2017-2021</th></tr></thead><tbody>{paper_rows}</tbody></table>
    <p class="note">Paper values are drawn from the univariate regression discussion and out-of-sample strategy discussion. This is a directional benchmark rather than an exact replication because this file has no option volume or open interest.</p>
  </section>
  <div class="grid">
    <section>{plot_to_html(fig3)}</section>
    <section>
      <h2>Out-of-sample timing stats</h2>
      <table><thead><tr><th>Series</th><th>Daily bps</th><th>Ann return</th><th>Ann vol</th><th>Excess Sharpe</th></tr></thead><tbody>{strategy_rows}</tbody></table>
      <p class="note">The strategy follows the paper's expanding-window idea: invest in the index when forecast excess return is positive, otherwise cash earning the dataset RiskFreeRate. Net version subtracts 1bp on allocation switches.</p>
    </section>
  </div>
  <section>
    <h2>Cash-threshold sensitivity</h2>
    <table class="sensitivity-table"><thead><tr><th>Cash threshold, bps</th><th>Cash days</th><th>Switches</th><th>Final $</th><th>Ann return</th><th>Ann vol</th><th>Excess Sharpe</th></tr></thead><tbody>{threshold_table_rows}</tbody></table>
    <p class="note">Threshold is in daily excess-return basis points. A 5bp threshold means the strategy stays in the index unless the expanding-window forecast is below -5bp.</p>
  </section>
  <div class="grid">
    <section>{plot_to_html(fig4)}</section>
    <section>{plot_to_html(fig5)}</section>
  </div>
  <section>
    <h2>ATM call-put gate</h2>
    <table class="sensitivity-table"><thead><tr><th>Rule</th><th>Cash days</th><th>Switches</th><th>Final $</th><th>Ann return</th><th>Ann vol</th><th>Excess Sharpe</th></tr></thead><tbody>{gate_table_rows}</tbody></table>
    <p class="note">The gate treats positive ATM call IV minus ATM put IV as bullish. When bullish, the strategy stays in the index instead of following a bearish paper-spread forecast into cash.</p>
  </section>
  <section>{plot_to_html(fig6)}</section>
</main>
</body>
</html>"""
    return html


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["Date", "Expiration"])
    daily = make_signal(df)
    daily.to_csv(OUT_DIR / "daily_iv_spread_returns.csv")

    regressions = []
    for signal in ["vs_ew", "vs_qw", "vs_delta_ew", "atm_cp_ew", "atm_cp_qw"]:
        for h, lags in [(1, 5), (5, 10), (10, 15), (21, 25)]:
            reg = newey_west_regression(daily[f"excess_{h}d"], daily[signal], lags)
            reg["signal"] = signal
            reg["horizon"] = f"{h}d"
            regressions.append(reg)

    bt = strategy_backtest(daily, "vs_ew")
    bt.to_csv(OUT_DIR / "oos_strategy.csv")
    threshold_rows = threshold_sensitivity(daily, "vs_ew")
    gate_rows = gate_sensitivity(daily)

    stats = {
        "Strategy gross": ann_stats(bt["strategy_log_ret_gross"], bt["cash_log_ret"]),
        "Strategy net": ann_stats(bt["strategy_log_ret_net"], bt["cash_log_ret"]),
        "Buy and hold": ann_stats(bt["buy_hold_log_ret"], bt["cash_log_ret"]),
        "Cash": ann_stats(bt["cash_log_ret"], bt["cash_log_ret"]),
    }

    results = {
        "sample_start": str(daily.index.min().date()),
        "sample_end": str(daily.index.max().date()),
        "n_days": int(daily.index.nunique()),
        "signal_summary": daily[
            [
                "vs_ew",
                "vs_qw",
                "vs_delta_ew",
                "atm_cp_ew",
                "atm_cp_qw",
                "n_put",
                "n_call",
                "n_delta_put",
                "n_delta_call",
            ]
        ].describe().to_dict(),
        "regressions": regressions,
        "strategy_stats": stats,
        "threshold_sensitivity": threshold_rows,
        "gate_sensitivity": gate_rows,
        "oos_final_growth": {
            "strategy_gross": float(bt["strategy_log_ret_gross_growth"].iloc[-1]),
            "strategy_net": float(bt["strategy_log_ret_net_growth"].iloc[-1]),
            "buy_hold": float(bt["buy_hold_log_ret_growth"].iloc[-1]),
            "cash": float(bt["cash_log_ret_growth"].iloc[-1]),
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(
        build_report(daily, regressions, bt, stats, threshold_rows, gate_rows),
        encoding="utf-8",
    )
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
