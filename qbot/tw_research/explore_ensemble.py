"""B2 事後探索（不參與判定）：四個模型等權組合後，對未來 h 日報酬的 IC。

只用 B2 測試期（2021-01-01～2026-09-30）已存的預測；不碰 2013–2020（C 的預先承諾期間）。
組合方式事先固定、不在測試期調權重：
    主版本   每日把各模型分數轉成橫斷面百分位名次，四個等權平均
    穩健性   每日把各模型分數轉成橫斷面 z 分數，四個等權平均

執行：python qbot/tw_research/explore_ensemble.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_research import analysis, config, factors  # noqa: E402
from tw_research.explore_b2 import HORIZONS, ic_by_horizon, ic_by_year, load_closes, load_results  # noqa: E402

OUT = config.OUTPUT_DIR.joinpath("explore_b2")


def cross_sectional(scores: pd.DataFrame, how: str) -> pd.DataFrame:
    g = scores.groupby(level="datetime")
    if how == "rank":
        return g.rank(pct=True)
    return (scores - g.transform("mean")) / g.transform("std")


def score_correlation(scores: pd.DataFrame) -> pd.DataFrame:
    """每日橫斷面 Spearman 相關的時間平均。"""
    names = list(scores.columns)
    out = pd.DataFrame(1.0, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ic = factors.daily_rank_ic(scores[a], scores[b])
            out.loc[a, b] = out.loc[b, a] = ic["ic"].mean()
    return out


def forward_returns(close: pd.DataFrame, h: int) -> pd.Series:
    c = close.ffill()
    fwd = (c.shift(-(1 + h)) / c.shift(-1) - 1).stack()
    fwd.index = fwd.index.set_names(["datetime", "instrument"])
    return fwd


def paired_ic_diff(a: pd.Series, b: pd.Series, y: pd.Series, lag: int) -> dict:
    """同一天、同一批股票上兩個分數的 IC 差，NW t。"""
    ia = factors.daily_rank_ic(a, y)["ic"]
    ib = factors.daily_rank_ic(b, y)["ic"]
    d = (ia - ib).dropna()
    mu, _, t = analysis.newey_west_t(d.to_numpy(), lag)
    return {"diff": mu, "t": t}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    factors.init_qlib()
    results = load_results()
    scores = pd.DataFrame({name: r["pred"] for name, r in results.items()}).dropna()
    close = load_closes(scores.index)

    ens = {
        "組合（名次平均）": cross_sectional(scores, "rank").mean(axis=1),
        "組合（z 分數平均）": cross_sectional(scores, "z").mean(axis=1),
    }
    all_scores = {**{k: scores[k] for k in scores.columns}, **ens}

    ic_h = pd.concat({k: ic_by_horizon(s, close) for k, s in all_scores.items()}, axis=1)
    corr = score_correlation(scores)

    label = results["M1_Linear_A158"]["label"]
    ic_daily = {k: factors.daily_rank_ic(s, label) for k, s in all_scores.items()}
    ic_year = pd.DataFrame({k: ic_by_year(v) for k, v in ic_daily.items()})

    rows = []
    main_ens = ens["組合（名次平均）"]
    for h in HORIZONS:
        y = forward_returns(close, h).reindex(scores.index)
        lag = max(config.NW_LAG_DAILY, h)
        for name in scores.columns:
            r = paired_ic_diff(main_ens, scores[name], y, lag)
            rows.append({"h": h, "對照": name, "IC 差（組合 − 單一）": r["diff"], "t": r["t"]})
    diffs = pd.DataFrame(rows).set_index(["h", "對照"])

    ic_h.to_csv(OUT.joinpath("ensemble_ic_h.csv"), encoding="utf-8-sig")
    corr.to_csv(OUT.joinpath("ensemble_score_corr.csv"), encoding="utf-8-sig")
    ic_year.to_csv(OUT.joinpath("ensemble_ic_year.csv"), encoding="utf-8-sig")
    diffs.to_csv(OUT.joinpath("ensemble_ic_diff.csv"), encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("== 分數相關（每日橫斷面 Spearman 的平均）")
    print(corr.round(3).to_string())
    print("== IC（ic_w）與 t，依 h")
    print(ic_h.xs("ic_w", axis=1, level=1).round(4).to_string())
    print(ic_h.xs("t", axis=1, level=1).round(2).to_string())
    print("== ICIR")
    print(ic_h.xs("icir", axis=1, level=1).round(3).to_string())
    print("== 每年 IC（h=1）")
    print(ic_year.round(4).to_string())
    print("== 組合（名次平均）對單一模型的 IC 差")
    print(diffs.unstack("對照").round(4).to_string())


if __name__ == "__main__":
    main()
