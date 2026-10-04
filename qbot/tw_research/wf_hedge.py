"""C 部分：線性模型低換手＋0050 避險的 alpha，walk-forward 2013–2020。

預先承諾見 config.PRE_REGISTRATION_C（報告開頭原文貼上）。只檢驗一個設定：
OLS＋Alpha158（每年用 2008…Y−1 重訓）、TopkDropout topk 50 / n_drop 1、每月以持股加權 beta 對 0050 避險。

執行：python qbot/tw_research/wf_hedge.py
輸出：outputs/tw_research/wf_hedge/、outputs/tw_research/WF_HEDGE_REPORT.md
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_research import analysis, config, factors  # noqa: E402

OUT = config.OUTPUT_DIR.joinpath("wf_hedge")
REPORT = config.OUTPUT_DIR.joinpath("WF_HEDGE_REPORT.md")


# ---------------------------------------------------------------- 模型
def train_end(year: int) -> str:
    return f"{year - 1}-{config.WF_TRAIN_END_MMDD}"


def handler_config(year: int) -> dict:
    return {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": config.WF_TRAIN_START,
            "end_time": f"{year}-12-31",
            "fit_start_time": config.WF_TRAIN_START,
            "fit_end_time": train_end(year),
            "instruments": config.UNIVERSE,
            "infer_processors": [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ],
        },
    }


def fit_predict_year(year: int) -> Tuple[pd.Series, pd.Series]:
    from qlib.contrib.model.linear import LinearModel
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    dataset = init_instance_by_config({
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config(year),
            "segments": {"train": (config.WF_TRAIN_START, train_end(year)),
                         "test": (f"{year}-01-01", f"{year}-12-31")},
        },
    })
    model = LinearModel(estimator="ols")
    with R.start(experiment_name="tw_wf_hedge", recorder_name=str(year)):
        model.fit(dataset)
        pred = model.predict(dataset, segment="test")
    label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
    label = label.iloc[:, 0] if isinstance(label, pd.DataFrame) else label
    return pred.rename("score"), label.rename("label")


# ---------------------------------------------------------------- 回測
def qlib_backtest(pred: pd.Series, start: str, end: str, limit: float, n_drop: int,
                  fee_multiplier: float = 1.0) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, pd.Series]]:
    from qlib.backtest import backtest

    fee = config.FEE_RATE * config.FEE_DISCOUNT * fee_multiplier
    portfolio, _ = backtest(
        start_time=start,
        end_time=end,
        strategy={"class": "TopkDropoutStrategy", "module_path": "qlib.contrib.strategy",
                  "kwargs": {"signal": pred.to_frame("score"), "topk": config.WF_TOPK, "n_drop": n_drop}},
        executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                  "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}},
        benchmark=config.BENCHMARK,
        account=config.B2_ACCOUNT,
        exchange_kwargs={"freq": "day", "limit_threshold": limit, "deal_price": "close",
                         "open_cost": fee, "close_cost": fee + config.SELL_TAX_STOCK,
                         "min_cost": config.MIN_FEE * (fee_multiplier > 0)},
    )
    report, positions = portfolio["1day"]
    out = pd.DataFrame({"gross": report["return"], "cost": report["cost"], "turnover": report["turnover"]})
    out["net"] = out["gross"] - out["cost"]
    weights = {pd.Timestamp(d): pd.Series(p.get_stock_weight_dict(only_stock=False), dtype=float)
               for d, p in positions.items()}
    return out, weights


def segmented_backtest(pred: pd.Series, n_drop: int, fee_multiplier: float = 1.0):
    reports, weights, starts = [], {}, []
    for start, end, limit in config.WF_SEGMENTS:
        rep, w = qlib_backtest(pred, start, end, limit, n_drop, fee_multiplier)
        reports.append(rep)
        weights.update(w)
        starts.append(rep.index[0])
    return pd.concat(reports).sort_index(), weights, starts


# ---------------------------------------------------------------- 避險
def stock_betas(stock_ret: pd.DataFrame, bench_ret: pd.Series, date, symbols) -> pd.Series:
    """date（含）以前 BETA_WINDOW 個交易日的 OLS beta；有效觀測不足記為 1，截斷在 BETA_CLIP。"""
    window = stock_ret.loc[:date].iloc[-config.BETA_WINDOW:]
    b = bench_ret.reindex(window.index)
    out = {}
    for s in symbols:
        if s not in window:
            out[s] = 1.0
            continue
        pair = pd.DataFrame({"y": window[s], "x": b}).dropna()
        if len(pair) < config.BETA_MIN_OBS or pair["x"].var() == 0:
            out[s] = 1.0
            continue
        out[s] = float(np.cov(pair["y"], pair["x"], ddof=1)[0, 1] / pair["x"].var())
    return pd.Series(out, dtype=float).clip(*config.BETA_CLIP)


def update_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每月第一個交易日。"""
    months = dates.to_period("M")
    return dates[~months.duplicated(keep="first")]


def hedge_ratios(dates: pd.DatetimeIndex, weights: Dict[pd.Timestamp, pd.Series], stock_ret: pd.DataFrame,
                 bench_ret: pd.Series, segment_starts=()) -> pd.Series:
    """每月第一個交易日收盤算 H = Σ w_i β_i，從隔日起適用到下一次更新日（含）。

    回測分段的第一天還是現金（收盤才建倉），當天 H = 0。
    """
    H = pd.Series(0.0, index=dates)
    ups = list(update_dates(dates))
    for k, u in enumerate(ups):
        w = weights.get(u, pd.Series(dtype=float))
        h = float((w * stock_betas(stock_ret, bench_ret, u, w.index)).sum()) if len(w) else 0.0
        nxt = ups[k + 1] if k + 1 < len(ups) else dates[-1] + pd.Timedelta(days=1)
        H[(dates > u) & (dates <= nxt)] = h
    for s in segment_starts:
        if s in H.index:
            H.loc[s] = 0.0
    return H


def hedged_returns(net: pd.Series, bench_ret: pd.Series, H: pd.Series,
                   cost_annual: float = config.HEDGE_COST_ANNUAL) -> pd.Series:
    b = bench_ret.reindex(net.index).fillna(0.0)
    Hh = H.reindex(net.index).fillna(0.0)
    return net - Hh * b - Hh.abs() * cost_annual / analysis.TRADING_DAYS


# ---------------------------------------------------------------- 統計與判定
def evaluate(h: pd.Series, bench_ret: pd.Series, period=config.WF_PERIOD) -> dict:
    h = h.loc[period[0]:period[1]].dropna()
    mu, se, t = analysis.newey_west_t(h.to_numpy(), config.NW_LAG_DAILY)
    ann, ann_se = mu * analysis.TRADING_DAYS, se * analysis.TRADING_DAYS
    years = h.groupby(h.index.year).mean() * analysis.TRADING_DAYS
    halves = [float(h.loc[a:b].mean() * analysis.TRADING_DAYS) for a, b in config.WF_HALVES]
    k = int(np.ceil(len(h) * config.C_CONCENTRATION_TOP))
    conc = float(h.drop(h.nlargest(k).index).mean() * analysis.TRADING_DAYS)
    b = bench_ret.reindex(h.index)
    ok = b.notna()
    beta = float(np.cov(h[ok], b[ok], ddof=1)[0, 1] / b[ok].var())
    return {"E": ann, "se": ann_se, "t": t, "p": analysis.one_sided_p(t, len(h)), "upper95": ann + 1.645 * ann_se,
            "mde": 2 * ann_se, "years": years, "n_positive_years": int((years > 0).sum()),
            "half1": halves[0], "half2": halves[1], "concentration_E": conc, "realized_beta": beta,
            "n_days": int(len(h))}


def verdict_c(st: dict, ew_alpha: float) -> Tuple[str, list]:
    flags = []
    if abs(st["realized_beta"]) > 0.2:
        flags.append("避險失效")
    if (st["p"] < 0.05 and st["E"] >= config.C_MATERIALITY
            and st["n_positive_years"] >= config.C_MIN_POSITIVE_YEARS
            and st["half1"] > 0 and st["half2"] > 0 and st["concentration_E"] > 0):
        v = "(+)"
        if ew_alpha <= 0:
            flags.append("優勢可能來自大小型股的相對表現，而非選股")
    elif st["upper95"] < config.C_MATERIALITY:
        v = "(−)"
    else:
        v = "(?)"
    return v, flags


def regression_alpha(y: pd.Series, x: pd.Series) -> Tuple[float, float, float]:
    """y 對 x 迴歸：回傳 (beta, 年化 alpha, alpha 的 NW t)。"""
    d = pd.DataFrame({"y": y, "x": x}).dropna()
    beta = float(np.cov(d["y"], d["x"], ddof=1)[0, 1] / d["x"].var())
    a = d["y"] - beta * d["x"]
    mu, _, t = analysis.newey_west_t(a.to_numpy(), config.NW_LAG_DAILY)
    return beta, mu * analysis.TRADING_DAYS, t


# ---------------------------------------------------------------- 主流程
def load_predictions() -> Tuple[pd.Series, pd.Series]:
    OUT.mkdir(parents=True, exist_ok=True)
    preds, labels = [], []
    for year in config.WF_YEARS:
        path = OUT.joinpath(f"pred_{year}.pkl")
        if path.exists():
            with open(path, "rb") as f:
                p, lab = pickle.load(f)
        else:
            p, lab = fit_predict_year(year)
            with open(path, "wb") as f:
                pickle.dump((p, lab), f)
        print(f"WF {year}: {len(p)} predictions", flush=True)
        preds.append(p)
        labels.append(lab)
    return pd.concat(preds).sort_index(), pd.concat(labels).sort_index()


def load_returns(symbols) -> Tuple[pd.DataFrame, pd.Series]:
    from qlib.data import D

    from tw_research import data

    df = D.features(sorted(symbols), ["$close"], start_time="2011-10-01", end_time="2021-01-31")
    close = df["$close"].unstack(level="instrument").sort_index()
    stock_ret = close.pct_change(fill_method=None)
    b = data.load_t1_bars()["close"]
    bench_ret = b.pct_change().loc["2011-10-01":"2021-01-31"]
    return stock_ret, bench_ret


def run():
    factors.init_qlib()
    pred, label = load_predictions()
    stock_ret, bench_ret = load_returns(pred.index.get_level_values("instrument").unique())
    ew = factors.universe_ew_returns(config.WF_PERIOD[0], config.WF_PERIOD[1])

    ic = factors.daily_rank_ic(pred, label)
    w = ic["k"] - 1
    ic_year = (ic["ic"] * w).groupby(ic.index.year).sum() / w.groupby(ic.index.year).sum()

    runs = {}
    for n_drop in (config.WF_N_DROP, 2, 5):
        rep, weights, starts = segmented_backtest(pred, n_drop)
        H = hedge_ratios(rep.index, weights, stock_ret, bench_ret, starts)
        runs[n_drop] = {"report": rep, "H": H, "starts": starts}
        print(f"backtest n_drop={n_drop} done", flush=True)

    main = runs[config.WF_N_DROP]
    rep, H = main["report"], main["H"]
    h = hedged_returns(rep["net"], bench_ret, H)
    st = evaluate(h, bench_ret)
    sub = evaluate(h, bench_ret, config.WF_SUBPERIOD)
    ew_beta, ew_alpha, ew_t = regression_alpha(rep["net"], ew)
    b50_beta, b50_alpha, b50_t = regression_alpha(rep["net"], bench_ret.reindex(rep.index))
    v, flags = verdict_c(st, ew_alpha)

    sens = {}
    for c in (0.0, 0.01, 0.02):
        sens[f"避險成本 {c:.0%}"] = evaluate(hedged_returns(rep["net"], bench_ret, H, c), bench_ret)
    for n_drop in (2, 5):
        r = runs[n_drop]
        sens[f"n_drop {n_drop}"] = evaluate(hedged_returns(r["report"]["net"], bench_ret, r["H"]), bench_ret)

    result = {"stats": st, "sub": sub, "verdict": v, "flags": flags, "ic": ic, "ic_year": ic_year,
              "h": h, "H": H, "report": rep, "ew": ew, "bench_ret": bench_ret,
              "unhedged": {"vs_ew": (ew_beta, ew_alpha, ew_t), "vs_0050": (b50_beta, b50_alpha, b50_t),
                           "excess_vs_ew": float((rep["net"] - ew.reindex(rep.index)).mean() * 252),
                           "excess_vs_0050": float((rep["net"] - bench_ret.reindex(rep.index)).mean() * 252)},
              "sens": sens}
    with open(OUT.joinpath("result.pkl"), "wb") as f:
        pickle.dump(result, f)
    write_report(result)
    print(f"verdict {v} {flags}  E={st['E']:.4f} t={st['t']:.2f}", flush=True)


# ---------------------------------------------------------------- 報告
def pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.2f}%"


def write_report(r: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tw_research import report as rp  # 沿用同一套圖表樣式

    st, sub = r["stats"], r["sub"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cum = r["h"].loc[config.WF_PERIOD[0]:config.WF_PERIOD[1]].cumsum()
    ax.plot(cum.index, cum.to_numpy(), color=rp.SERIES[0], linewidth=1.8)
    ax.axhline(0, color=rp.AXIS, linewidth=1)
    ax.axvline(pd.Timestamp(config.WF_SUBPERIOD[0]), color=rp.MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(pd.Timestamp(config.WF_SUBPERIOD[0]), ax.get_ylim()[1], " 2018–2020 子期", color=rp.INK2,
            fontsize=8, va="top")
    ax.set_ylabel("累積避險後淨報酬（算術加總）")
    ax.yaxis.set_major_formatter(rp.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    rp.FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(rp.FIG.joinpath("wf_hedge_cum.png"), dpi=150)
    plt.close(fig)

    years = pd.DataFrame({"避險後 h（年化）": st["years"].map(pct), "rank IC": r["ic_year"].map(lambda v: f"{v:+.4f}")})
    years.index.name = "年"
    sens = pd.DataFrame({k: {"E": pct(v["E"]), "t": f"{v['t']:.2f}", "上界": pct(v["upper95"])}
                         for k, v in r["sens"].items()}).T
    sens.index.name = "敏感度（只報告）"
    u = r["unhedged"]
    flags = "、".join(r["flags"]) if r["flags"] else "無"
    text = f"""# 線性模型低換手＋0050 避險（walk-forward 2013–2020）

由 `qbot/tw_research/wf_hedge.py` 產生。

<details><summary><b>預先承諾（原文，寫於任何 2013–2020 的模型預測或組合報酬計算之前）</b></summary>

{config.PRE_REGISTRATION_C}
</details>

## 判定：{r['verdict']}

附加標記：{flags}

| 指標 | 全期 2013–2020 | 子期 2018–2020（描述） |
|---|---|---|
| E（避險後淨 alpha，年化） | {pct(st['E'])} | {pct(sub['E'])} |
| NW 標準誤 | {pct(st['se'])} | {pct(sub['se'])} |
| t | {st['t']:.2f} | {sub['t']:.2f} |
| 單尾 p | {st['p']:.3f} | {sub['p']:.3f} |
| 95% 上界 | {pct(st['upper95'])} | {pct(sub['upper95'])} |
| MDE（2×SE；事前估計 {pct(config.C_MDE_PRIOR)}） | {pct(st['mde'])} | {pct(sub['mde'])} |
| 正報酬年數 | {st['n_positive_years']} / {len(st['years'])} | {sub['n_positive_years']} / {len(sub['years'])} |
| 前半 2013–2016／後半 2017–2020 | {pct(st['half1'])}／{pct(st['half2'])} | — |
| 剔除最好 1% 交易日後的 E | {pct(st['concentration_E'])} | {pct(sub['concentration_E'])} |
| 避險後對 0050 的實現 beta | {st['realized_beta']:.3f} | {sub['realized_beta']:.3f} |

![累積](figures/wf_hedge_cum.png)

{rp.md_table(years)}

未避險組合（描述）：相對 0050 超額 {pct(u['excess_vs_0050'])}、beta {u['vs_0050'][0]:.2f}、alpha {pct(u['vs_0050'][1])}（t {u['vs_0050'][2]:.2f}）；
相對 top200 等權超額 {pct(u['excess_vs_ew'])}、beta {u['vs_ew'][0]:.2f}、alpha {pct(u['vs_ew'][1])}（t {u['vs_ew'][2]:.2f}）。
平均避險比率 {r['H'][r['H'] > 0].mean():.2f}；每日換手（買賣合計）{r['report']['turnover'].mean():.1%}；年化交易成本 {pct(r['report']['cost'].mean() * 252)}。

{rp.md_table(sens)}
"""
    REPORT.write_text(text, encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    run()
