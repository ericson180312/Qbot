"""B2 事後探索（不參與判定）：預測力的結構，以及忽略成本、降低換股頻率時的表現。

測試期在預先承諾的判定之後才看這些數字，所以這裡的結果只能描述、不能當成發現；
任何想採用的設定都要用 2026-09-30 之後的新資料另立預先承諾。

    1. IC 的年度分布與衰減：預測分數對未來 h 日報酬的 rank IC（h = 1, 2, 5, 10, 20, 60）
    2. 十分位：依預測分十組，隔日報酬的平均數與中位數（IC 與平均數可能方向不同）
    3. TopkDropout 的換手 × 成本交叉表（沿用 run_factors 已跑的回測）
    4. 每 H 日重選前 50 名等權、分 H 批錯開持有（H = 1, 5, 10, 20, 60），扣 / 不扣成本

執行：python qbot/tw_research/explore_b2.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_research import analysis, config, factors  # noqa: E402

OUT = config.OUTPUT_DIR.joinpath("explore_b2")
HORIZONS = [1, 2, 5, 10, 20, 60]
HOLDS = [1, 5, 10, 20, 60]
TOPK = config.B2_TOPK


def load_results():
    out = {}
    for name in factors.MODELS:
        with open(config.OUTPUT_DIR.joinpath("factors", f"b2_{name}.pkl"), "rb") as f:
            out[name] = pickle.load(f)
    return out


def load_closes(pred_index: pd.MultiIndex) -> pd.DataFrame:
    """測試期出現過的股票的還原收盤價（不受股票池區間限制，持有中離開股票池仍有報酬）。"""
    from qlib.data import D

    symbols = sorted(pred_index.get_level_values("instrument").unique())
    df = D.features(symbols, ["$close"], start_time="2020-12-01", end_time="2026-10-01")
    return df["$close"].unstack(level="instrument").sort_index()


def ic_by_horizon(pred: pd.Series, close: pd.DataFrame) -> pd.DataFrame:
    """預測 t 日分數 vs t+1 收盤到 t+1+h 收盤的報酬（與 qlib 的 label、成交時點一致）。"""
    c = close.ffill()
    rows = []
    for h in HORIZONS:
        fwd = c.shift(-(1 + h)) / c.shift(-1) - 1
        fwd = fwd.stack().rename("fwd")
        fwd.index = fwd.index.set_names(["datetime", "instrument"])
        y = fwd.reindex(pred.index)
        ic = factors.daily_rank_ic(pred, y)
        st = factors.ic_stats(ic, lag=max(config.NW_LAG_DAILY, h))
        rows.append({"h": h, "ic_w": st["ic_w"], "icir": st["icir"], "t": st["t"], "days": st["n_days"]})
    return pd.DataFrame(rows).set_index("h")


def ic_by_year(ic: pd.DataFrame) -> pd.Series:
    w = ic["k"] - 1
    return (ic["ic"] * w).groupby(ic.index.year).sum() / w.groupby(ic.index.year).sum()


def deciles(pred: pd.Series, label: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"s": pred, "y": label}).dropna()
    d["y_dm"] = d["y"] - d.groupby(level=0)["y"].transform("mean")
    d["dec"] = d.groupby(level=0)["s"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    g = d.groupby("dec")["y_dm"]
    return pd.DataFrame({"mean_bp": g.mean() * 1e4, "median_bp": g.median() * 1e4,
                         "up9_share": d.groupby("dec")["y"].apply(lambda s: (s > 0.09).mean())})


def turnover_grid(bts: dict, ew: pd.Series) -> pd.DataFrame:
    rows = []
    for tag, n_drop in [("ndrop1", 1), ("ndrop2", 2), ("main", 5)]:
        b = bts[tag]
        one_way = b["turnover"].mean() / 2
        g = summarize_excess(b["gross"], ew, config.NW_LAG_DAILY)
        n = summarize_excess(b["net"], ew, config.NW_LAG_DAILY)
        rows.append({
            "n_drop": n_drop,
            "每日單邊換手": one_way,
            "平均持有天數": 1 / one_way if one_way > 0 else np.nan,
            "毛超額（不扣成本）": g["ann"],
            "毛 t": g["t"],
            "beta": g["beta"],
            "毛 alpha": g["alpha"],
            "毛 alpha t": g["alpha_t"],
            "年化成本": float(b["cost"].mean() * 252),
            "淨超額": n["ann"],
            "淨 t": n["t"],
            "淨 alpha": n["alpha"],
            "兩平來回成本": g["ann"] / (one_way * 252) if one_way > 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("n_drop")


def staggered_topk(pred: pd.Series, close: pd.DataFrame, hold: int, buy: float, sell: float) -> pd.DataFrame:
    """每 hold 日依 t 日分數選前 TOPK 名等權，t+1 收盤成交，持有 hold 日（期間不再平衡）。

    分 hold 批、每批錯開一天起跑，日報酬取各批平均（避免起跑日的運氣）。
    不處理漲跌停與整張限制；成本 = 買進成交額 × buy + 賣出成交額 × sell。
    """
    ret = close.ffill().pct_change()
    dates = pred.index.get_level_values(0).unique().sort_values()
    scores = pred.unstack(level="instrument")
    pos = {d: i for i, d in enumerate(ret.index)}
    gross_all, net_all = [], []
    for offset in range(hold):
        gross = pd.Series(0.0, index=ret.index)
        cost = pd.Series(0.0, index=ret.index)
        prev_w = pd.Series(dtype=float)
        for k in range(offset, len(dates), hold):
            t = dates[k]
            s = scores.loc[t].dropna()
            pick = s.nlargest(TOPK).index
            i0 = pos[t] + 1  # t+1 收盤成交
            if i0 >= len(ret.index):
                break
            i1 = min(i0 + hold, len(ret.index) - 1)
            r = ret.iloc[i0 + 1:i1 + 1][pick].fillna(0.0)
            if r.empty:
                break
            growth = (1 + r).cumprod()
            value = growth.mean(axis=1)  # 等權買進後不再平衡
            daily = value / value.shift(1).fillna(1.0) - 1
            gross.iloc[i0 + 1:i1 + 1] = daily.to_numpy()
            new_w = pd.Series(1 / len(pick), index=pick)
            delta = new_w.subtract(prev_w, fill_value=0.0)
            cost.iloc[i0] = delta.clip(lower=0).sum() * buy + (-delta.clip(upper=0)).sum() * sell
            prev_w = growth.iloc[-1] / growth.iloc[-1].sum()
        gross_all.append(gross)
        net_all.append((1 + gross) * (1 - cost) - 1)
    g = pd.concat(gross_all, axis=1).mean(axis=1)
    n = pd.concat(net_all, axis=1).mean(axis=1)
    start = ret.index[pos[dates[0]] + 2]
    return pd.DataFrame({"gross": g, "net": n}).loc[start:config.BACKTEST_END]


def summarize_excess(series: pd.Series, ew: pd.Series, lag: int) -> dict:
    """相對等權的超額，以及對等權報酬迴歸後的 beta 與 alpha（扣掉市場曝險差異後的部分）。"""
    m = ew.reindex(series.index)
    ok = series.notna() & m.notna()
    y, x = series[ok], m[ok]
    ex = y - x
    mu, se, t = analysis.newey_west_t(ex.to_numpy(), lag)
    beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    alpha = y - beta * x
    a_mu, _, a_t = analysis.newey_west_t(alpha.to_numpy(), lag)
    halves = [float(ex.loc[a:b].mean() * 252) for a, b in config.TEST_HALVES]
    return {"ann": mu * 252, "t": t, "half1": halves[0], "half2": halves[1],
            "beta": beta, "alpha": a_mu * 252, "alpha_t": a_t}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    factors.init_qlib()
    results = load_results()
    with open(config.OUTPUT_DIR.joinpath("factors", "b2.pkl"), "rb") as f:
        ew = pickle.load(f)["ew"]
    any_pred = next(iter(results.values()))["pred"]
    close = load_closes(any_pred.index)
    buy, sell = config.buy_cost_rate(), config.sell_cost_rate()

    tables = {"ic_h": {}, "ic_year": {}, "deciles": {}, "grid": {}, "hold": {}}
    for name, r in results.items():
        tables["ic_h"][name] = ic_by_horizon(r["pred"], close)
        tables["ic_year"][name] = ic_by_year(r["ic"])
        tables["deciles"][name] = deciles(r["pred"], r["label"])
        tables["grid"][name] = turnover_grid(r["backtests"], ew)
        rows = []
        for h in HOLDS:
            bt = staggered_topk(r["pred"], close, h, buy, sell)
            g = summarize_excess(bt["gross"], ew, max(config.NW_LAG_DAILY, h))
            n = summarize_excess(bt["net"], ew, max(config.NW_LAG_DAILY, h))
            rows.append({"H": h, "毛超額": g["ann"], "毛 t": g["t"], "beta": g["beta"], "毛 alpha": g["alpha"],
                         "毛 alpha t": g["alpha_t"], "淨超額": n["ann"], "淨 t": n["t"], "淨 alpha": n["alpha"],
                         "淨 alpha t": n["alpha_t"], "淨前半": n["half1"], "淨後半": n["half2"]})
            print(f"{name} H={h} done", flush=True)
        tables["hold"][name] = pd.DataFrame(rows).set_index("H")
    with open(OUT.joinpath("explore_b2.pkl"), "wb") as f:
        pickle.dump(tables, f)
    for key, d in tables.items():
        pd.concat(d, axis=0 if key != "ic_year" else 1).to_csv(OUT.joinpath(f"{key}.csv"), encoding="utf-8-sig")
    print("done")


if __name__ == "__main__":
    main()
