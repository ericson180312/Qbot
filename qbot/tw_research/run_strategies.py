"""A 部分：Qbot 擇時策略在台股的回測與檢定。

    T1  0050 擇時：規則 vs 重排持有段的隨機擇時（ΔSharpe）
    T2  個股波段：每季流動性前 100 名，每筆交易 vs 同股隨機進場日（每筆超額報酬）

執行：python qbot/tw_research/run_strategies.py [--t2_limit N]
輸出：outputs/tw_research/strategies/*.csv、*.pkl
"""

import argparse
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_research import analysis, config, data, engine  # noqa: E402
from tw_research.strategies import STRATEGIES, STRATEGY_BY_CODE  # noqa: E402

OUT = config.OUTPUT_DIR.joinpath("strategies")


# ---------------------------------------------------------------- T1
def run_t1():
    symbol = config.T1_SYMBOL
    bars = data.load_t1_bars()
    n_bad = int(data.bad_days(bars).sum())
    start = bars.index[config.WARMUP_DAYS]
    end = pd.Timestamp(config.DATA_END)
    windows = {
        "full": (start, end),
        "A": (start, pd.Timestamp(config.TRACK_A[1])),
        "B": (pd.Timestamp(config.TRACK_B[0]), end),
    }
    buy = config.buy_cost_rate()
    sell = config.sell_cost_rate(symbol)
    oo = analysis.open_to_open_log_returns(bars)

    rows, curves, years = [], {}, {}
    bh_held = pd.Series(True, index=bars.index)
    bh = analysis.position_log_returns(bh_held.loc[start:], oo.loc[start:end].dropna(), buy, sell)
    curves["買進持有 0050"] = bh
    for strat in STRATEGIES:
        res = engine.run(bars, strat, symbol)
        trades = engine.trades_from_held(res.held)
        row = {"code": strat.code, "strategy": strat.__name__, "source": strat.source,
               "blocked_buys": res.blocked_buys, "delayed_sells": res.delayed_sells}
        for w, (a, b) in windows.items():
            test = analysis.t1_test(res.held, bars, a, b, buy, sell)
            for key in ["rule_sharpe", "null_median", "delta_sharpe", "sd0", "mde", "p", "upper95",
                        "positive_year_share", "exposure"]:
                row[f"{w}_{key}"] = test[key]
            if w == "full":
                curves[strat.code] = test["rule_log_returns"]
                years[strat.code] = test["year_excess"]
                summ = analysis.summarize(test["rule_log_returns"], res.held)
                row.update({f"full_{k}": v for k, v in summ.items()})
        in_window = trades[(trades["entry_date"] >= start)]
        tlr = analysis.trade_net_log_returns(in_window, bars, buy, sell)
        row["n_trades"] = len(tlr)
        drop = np.sort(tlr)[::-1][config.T1_CONCENTRATION_TOP:]
        row["concentration_ok"] = bool(drop.sum() > 0)
        bt_vals = res.values.loc[start:end]
        row["bt_CAGR"] = float((bt_vals.iloc[-1] / bt_vals.iloc[0]) ** (252 / len(bt_vals)) - 1)
        rows.append(row)
        print(f"T1 {strat.code} done", flush=True)

    t1 = pd.DataFrame(rows).set_index("code")
    adj = analysis.holm(t1["full_p"].to_dict())
    t1["holm_p"] = [adj[c][0] for c in t1.index]
    t1["holm_reject"] = [adj[c][1] for c in t1.index]
    t1["verdict"] = [
        analysis.verdict(r.holm_reject, r.full_delta_sharpe, r.full_upper95, r.A_delta_sharpe,
                         r.B_delta_sharpe, r.full_positive_year_share, r.concentration_ok,
                         config.T1_MATERIALITY)
        for r in t1.itertuples()
    ]
    bh_summary = analysis.summarize(bh, bh_held)
    bh_summary["bad_days"] = n_bad
    bh_summary["first_bar"] = str(bars.index[0].date())
    return t1, pd.DataFrame(curves), pd.DataFrame(years), bh_summary, (start, end)


# ---------------------------------------------------------------- T2
def _t2_symbol(symbol: str, spans: pd.DataFrame, codes):
    bars = data.load_bars(symbol)
    if len(bars) < 60:
        return symbol, {}, None, []
    member = data.membership(spans, symbol, bars.index)
    first = member.idxmax() if member.any() else None
    if first is None:
        return symbol, {}, None, []
    # 指標暖身從入選前 300 個交易日開始；出場最晚看到資料結束
    lo = max(0, bars.index.get_loc(first) - 300)
    bars = bars.iloc[lo:]
    member = member.iloc[lo:]
    buy, sell = config.buy_cost_rate(), config.sell_cost_rate(symbol)
    oo = analysis.open_to_open_log_returns(bars)
    trades_out = {}
    daily = {"bh": np.expm1(oo.where(member))}
    seed_base = int(symbol[2:]) if symbol[2:].isdigit() else 0
    errors = []
    for i, code in enumerate(codes):
        strat = STRATEGY_BY_CODE[code]
        try:
            res = engine.run(bars, strat, symbol, in_universe=member)
        except Exception as exc:  # 記錄下來寫進報告，不讓單一股票讓整批作廢
            errors.append((symbol, code, repr(exc)))
            continue
        trades = engine.trades_from_held(res.held)
        rng = np.random.default_rng([config.NULL_SEED, i, seed_base])
        table = analysis.t2_trade_null(trades, bars, buy, sell, rng)
        table["symbol"] = symbol
        trades_out[code] = table
        lr = analysis.position_log_returns(res.held, oo, buy, sell)
        daily[code] = np.expm1(lr.where(member))
    return symbol, trades_out, pd.DataFrame(daily), errors


def run_t2(limit=None, workers: int = 10):
    spans = data.universe_spans(top_n=config.T2_TOP_N)
    symbols = sorted(spans["symbol"].unique())
    if limit:
        symbols = symbols[:limit]
    codes = [s.code for s in STRATEGIES]
    trades = {c: [] for c in codes}
    sums, counts = None, None
    errors = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_t2_symbol, s, spans, codes) for s in symbols]
        for k, fut in enumerate(as_completed(futures), 1):
            symbol, tables, daily, errs = fut.result()
            errors.extend(errs)
            for c, tab in tables.items():
                trades[c].append(tab)
            if daily is not None:
                s = daily.fillna(0.0)
                n = daily.notna().astype(float)
                sums = s if sums is None else sums.add(s, fill_value=0.0)
                counts = n if counts is None else counts.add(n, fill_value=0.0)
            if k % 50 == 0 or k == len(futures):
                print(f"T2 {k}/{len(futures)} symbols, {time.time() - t0:.0f}s", flush=True)
    trades = {c: pd.concat(v, ignore_index=True) for c, v in trades.items() if v}
    portfolio = (sums / counts).sort_index()

    rows = []
    for c, tab in trades.items():
        row = {"code": c, "strategy": STRATEGY_BY_CODE[c].__name__,
               "closed_trades": int(len(tab)), "excluded_bad_day": int(tab["spans_bad_day"].sum())}
        for w, (a, b) in {"full": (config.DATA_START, config.DATA_END), "A": config.TRACK_A,
                          "B": config.TRACK_B}.items():
            res = analysis.t2_test(tab, a, b)
            for key, v in res.items():
                row[f"{w}_{key}"] = v
        conc = analysis.t2_test(analysis.drop_top(tab, config.T2_CONCENTRATION_TOP),
                                config.DATA_START, config.DATA_END)
        row["concentration_E"] = conc["E"]
        row["concentration_ok"] = bool(conc["E"] > 0)
        rows.append(row)
    t2 = pd.DataFrame(rows).set_index("code")
    adj = analysis.holm(t2["full_p"].to_dict())
    t2["holm_p"] = [adj[c][0] for c in t2.index]
    t2["holm_reject"] = [adj[c][1] for c in t2.index]
    t2["verdict"] = [
        analysis.verdict(r.holm_reject, r.full_E, r.full_upper95, r.A_E, r.B_E,
                         r.full_positive_year_share, r.concentration_ok, config.T2_MATERIALITY)
        for r in t2.itertuples()
    ]
    t2.attrs["errors"] = errors
    return t2, trades, portfolio, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2_limit", type=int, default=None, help="只跑前 N 檔（除錯用）")
    parser.add_argument("--skip_t1", action="store_true")
    parser.add_argument("--skip_t2", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_t1:
        t1, curves, years, bh_summary, window = run_t1()
        t1.to_csv(OUT.joinpath("t1_results.csv"), encoding="utf-8-sig")
        with open(OUT.joinpath("t1.pkl"), "wb") as f:
            pickle.dump({"t1": t1, "curves": curves, "years": years, "bh": bh_summary, "window": window}, f)
    if not args.skip_t2:
        t2, trades, portfolio, errors = run_t2(args.t2_limit)
        t2.to_csv(OUT.joinpath("t2_results.csv"), encoding="utf-8-sig")
        with open(OUT.joinpath("t2.pkl"), "wb") as f:
            pickle.dump({"t2": t2, "trades": trades, "portfolio": portfolio, "errors": errors}, f)
        print(f"T2 errors: {len(errors)}", errors[:10], flush=True)


if __name__ == "__main__":
    main()
