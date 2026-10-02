"""B 部分：Alpha158 單因子篩選與 qlib 多因子模型回測。

執行：python qbot/tw_research/run_factors.py [--skip_b1] [--skip_b2] [--models M1_Linear_A158,...]
輸出：outputs/tw_research/factors/*.csv、*.pkl
"""

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_research import analysis, config, factors  # noqa: E402

OUT = config.OUTPUT_DIR.joinpath("factors")


def run_b1():
    t = time.time()
    df = factors.load_alpha158_raw(config.DISCOVERY[0], config.DATA_END)
    print(f"B1 loaded {df.shape} in {time.time() - t:.0f}s", flush=True)
    df, n_bad = factors.clean_label(df)
    print(f"B1 labels beyond limit set to NaN: {n_bad}", flush=True)
    screen = factors.single_factor_screen(df)
    judged = factors.judge_b1(screen)
    judged.to_csv(OUT.joinpath("b1_single_factor.csv"), encoding="utf-8-sig")

    # 描述：通過的因子在測試期的五分位平均報酬（alphalens）
    quantiles = {}
    passed = judged.index[judged["passed"]].tolist()
    if passed:
        import alphalens

        test = factors.segment(df, config.TEST)
        label = test["LABEL0"]
        for name in passed:
            fac = test[name] * judged.loc[name, "sign"]
            clean = pd.DataFrame({"factor": fac, "1D": label}).dropna()
            clean["factor_quantile"] = clean.groupby(level=0)["factor"].transform(
                lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
            clean.index = clean.index.set_names(["date", "asset"])
            mean_ret, _ = alphalens.performance.mean_return_by_quantile(clean, by_date=False, demeaned=True)
            quantiles[name] = mean_ret["1D"]
    with open(OUT.joinpath("b1.pkl"), "wb") as f:
        pickle.dump({"judged": judged, "quantiles": quantiles, "bad_labels": n_bad,
                     "n_labels": int(df["LABEL0"].notna().sum())}, f)
    del df
    gc.collect()
    return judged


def run_b2(models):
    results = {}
    ew = factors.universe_ew_returns(config.TEST[0], config.BACKTEST_END)
    for name in models:
        t = time.time()
        pred, label = factors.fit_predict(name)
        ic = factors.daily_rank_ic(pred, label)
        stats = factors.ic_stats(ic)
        stats["p"] = analysis.one_sided_p(stats["t"], stats["n_days"])
        bts = {}
        for tag, fee_mult, n_drop in [("main", 1.0, config.B2_N_DROP), ("fee0", 0.0, config.B2_N_DROP),
                                      ("fee_full", 1 / config.FEE_DISCOUNT, config.B2_N_DROP),
                                      ("ndrop1", 1.0, 1), ("ndrop2", 1.0, 2)]:
            bts[tag] = factors.topk_backtest(pred, fee_multiplier=fee_mult, n_drop=n_drop)
        results[name] = {"pred": pred, "label": label, "ic": ic, "ic_stats": stats, "backtests": bts}
        with open(OUT.joinpath(f"b2_{name}.pkl"), "wb") as f:
            pickle.dump(results[name], f)
        print(f"B2 {name} done in {time.time() - t:.0f}s", flush=True)
        gc.collect()
    return results, ew


def judge_all_b2(results, ew):
    adj = analysis.holm({k: v["ic_stats"]["p"] for k, v in results.items()})
    rows = []
    for name, r in results.items():
        bt = r["backtests"]["main"]
        excess = (bt["net"] - ew.reindex(bt.index)).dropna()
        j = factors.judge_b2(adj[name], excess)
        row = {"model": name, **{f"ic_{k}": v for k, v in r["ic_stats"].items()},
               "ic_holm_p": adj[name][0], "ic_holm_reject": adj[name][1], **j}
        for tag, b in r["backtests"].items():
            row[f"{tag}_ann_net"] = float(b["net"].mean() * analysis.TRADING_DAYS)
            row[f"{tag}_ann_excess_vs_ew"] = float((b["net"] - ew.reindex(b.index)).mean() * analysis.TRADING_DAYS)
        row["ann_gross_excess_vs_ew"] = float((bt["gross"] - ew.reindex(bt.index)).mean() * analysis.TRADING_DAYS)
        row["daily_turnover"] = float(bt["turnover"].mean())
        row["ann_cost"] = float(bt["cost"].mean() * analysis.TRADING_DAYS)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")

    # 基準診斷：各模型對 M1（線性）的每日 IC 差
    base = results.get("M1_Linear_A158")
    if base is not None:
        for name, r in results.items():
            if name == "M1_Linear_A158":
                continue
            diff = (r["ic"]["ic"] - base["ic"]["ic"]).dropna()
            mu, se, t = analysis.newey_west_t(diff.to_numpy(), config.NW_LAG_DAILY)
            table.loc[name, "ic_diff_vs_linear"] = mu
            table.loc[name, "ic_diff_vs_linear_t"] = t
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_b1", action="store_true")
    parser.add_argument("--skip_b2", action="store_true")
    parser.add_argument("--models", default=",".join(factors.MODELS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    factors.init_qlib()
    if not args.skip_b1:
        run_b1()
    if not args.skip_b2:
        models = [m for m in args.models.split(",") if m]
        results, ew = run_b2(models)
        # 全部四個模型都跑完才做 Holm；分批跑時從 pkl 讀回
        for name in factors.MODELS:
            path = OUT.joinpath(f"b2_{name}.pkl")
            if name not in results and path.exists():
                with open(path, "rb") as f:
                    results[name] = pickle.load(f)
        table = judge_all_b2(results, ew)
        table.to_csv(OUT.joinpath("b2_models.csv"), encoding="utf-8-sig")
        with open(OUT.joinpath("b2.pkl"), "wb") as f:
            pickle.dump({"table": table, "ew": ew,
                         "backtests": {k: v["backtests"] for k, v in results.items()},
                         "ic": {k: v["ic"] for k, v in results.items()}}, f)


if __name__ == "__main__":
    main()
