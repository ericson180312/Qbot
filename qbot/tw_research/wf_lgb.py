"""D 部分：低換手 LightGBM＋台指期避險，walk-forward 2013–2020。

預先承諾見 config.PRE_REGISTRATION_D（報告開頭原文貼上）。只檢驗一個主設定：
LightGBM＋Alpha158（每年 Y 用 2008…Y−2 訓練、Y−1 做 early stopping）、TopkDropout topk 50 / n_drop 1、
每月以持股加權 beta 賣台指期避險。判準門檻與 C 相同；避險、回測、統計函式沿用 wf_hedge.py。

執行：python qbot/tw_research/wf_lgb.py
輸出：outputs/tw_research/wf_lgb/、outputs/tw_research/WF_LGB_REPORT.md
"""

import pickle
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_research import config, factors, taifex  # noqa: E402
from tw_research import wf_hedge as wf  # noqa: E402

OUT = config.OUTPUT_DIR.joinpath("wf_lgb")
REPORT = config.OUTPUT_DIR.joinpath("WF_LGB_REPORT.md")
SCENARIOS = {  # 名稱 -> (買進費率, 賣出費率, 最低費用)；None = 散戶股票成本（主設定）
    "股票成本（主設定）": None,
    "個股期貨每邊 0.05%": (config.D_SSF_COSTS[0], config.D_SSF_COSTS[0], 0.0),
    "個股期貨每邊 0.15%": (config.D_SSF_COSTS[1], config.D_SSF_COSTS[1], 0.0),
}


def segments(year: int) -> dict:
    return {
        "train": (config.WF_TRAIN_START, f"{year - 2}-{config.WF_TRAIN_END_MMDD}"),
        "valid": (f"{year - 1}-01-01", f"{year - 1}-{config.WF_TRAIN_END_MMDD}"),
        "test": (f"{year}-01-01", f"{year}-12-31"),
    }


def handler_config(year: int) -> dict:
    seg = segments(year)
    return {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {  # 前處理用 Alpha158 預設（同 B2 的 M2）
            "start_time": config.WF_TRAIN_START,
            "end_time": f"{year}-12-31",
            "fit_start_time": seg["train"][0],
            "fit_end_time": seg["train"][1],
            "instruments": config.UNIVERSE,
        },
    }


def fit_predict_year(year: int) -> Tuple[pd.Series, pd.Series]:
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    dataset = init_instance_by_config({
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {"handler": handler_config(year), "segments": segments(year)},
    })
    model = init_instance_by_config({"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
                                     "kwargs": dict(factors._LGB_158, seed=config.MODEL_SEED)})
    with R.start(experiment_name="tw_wf_lgb", recorder_name=str(year)):
        model.fit(dataset)
        pred = model.predict(dataset, segment="test")
    label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
    label = label.iloc[:, 0] if isinstance(label, pd.DataFrame) else label
    best = getattr(model.model, "best_iteration", None)
    return pred.rename("score"), label.rename("label"), best


def load_predictions():
    OUT.mkdir(parents=True, exist_ok=True)
    preds, labels, best = [], [], {}
    for year in config.D_YEARS:
        path = OUT.joinpath(f"pred_{year}.pkl")
        if path.exists():
            with open(path, "rb") as f:
                p, lab, b = pickle.load(f)
        else:
            p, lab, b = fit_predict_year(year)
            with open(path, "wb") as f:
                pickle.dump((p, lab, b), f)
        best[year] = b
        print(f"WF-LGB {year}: {len(p)} predictions, best iteration {b}", flush=True)
        preds.append(p)
        labels.append(lab)
    return pd.concat(preds).sort_index(), pd.concat(labels).sort_index(), best


def load_stock_returns(symbols) -> pd.DataFrame:
    from qlib.data import D

    df = D.features(sorted(symbols), ["$close"], start_time="2011-10-01", end_time="2021-01-31")
    return df["$close"].unstack(level="instrument").sort_index().pct_change(fill_method=None)


def run():
    factors.init_qlib()
    pred, label, best = load_predictions()
    stock_ret = load_stock_returns(pred.index.get_level_values("instrument").unique())
    tx_ret = taifex.load_tx_returns()
    from tw_research import data

    r0050 = data.load_t1_bars()["close"].pct_change()
    ew = factors.universe_ew_returns(config.WF_PERIOD[0], config.WF_PERIOD[1])

    ic = factors.daily_rank_ic(pred, label)
    w = ic["k"] - 1
    ic_year = (ic["ic"] * w).groupby(ic.index.year).sum() / w.groupby(ic.index.year).sum()

    runs = {}
    for name, costs in SCENARIOS.items():
        n_drops = (config.D_N_DROP, 2, 5) if costs is None else (config.D_N_DROP,)
        for n_drop in n_drops:
            rep, weights, starts = wf.segmented_backtest(pred, n_drop, costs=costs)
            H = wf.hedge_ratios(rep.index, weights, stock_ret, tx_ret, starts)
            runs[(name, n_drop)] = {"report": rep, "H": H, "weights": weights, "starts": starts}
            print(f"backtest {name} n_drop={n_drop} done", flush=True)

    main = runs[("股票成本（主設定）", config.D_N_DROP)]
    rep, H = main["report"], main["H"]
    h = wf.hedged_returns(rep["net"], tx_ret, H, config.D_HEDGE_COST_ANNUAL)
    st = wf.evaluate(h, tx_ret)
    sub = wf.evaluate(h, tx_ret, config.WF_SUBPERIOD)
    ew_beta, ew_alpha, ew_t = wf.regression_alpha(rep["net"], ew)
    tx_beta, tx_alpha, tx_t = wf.regression_alpha(rep["net"], tx_ret.reindex(rep.index))
    v, flags = wf.verdict_c(st, ew_alpha)

    sens = {}
    for (name, n_drop), r in runs.items():
        if (name, n_drop) == ("股票成本（主設定）", config.D_N_DROP):
            continue
        hh = wf.hedged_returns(r["report"]["net"], tx_ret, r["H"], config.D_HEDGE_COST_ANNUAL)
        sens[f"{name}，n_drop {n_drop}"] = wf.evaluate(hh, tx_ret)
    for c in (0.0, 0.01):
        sens[f"台指期避險成本 {c:.0%}"] = wf.evaluate(wf.hedged_returns(rep["net"], tx_ret, H, c), tx_ret)
    H50 = wf.hedge_ratios(rep.index, main["weights"], stock_ret, r0050, main["starts"])
    sens["改用 0050 避險（C 的方法）"] = wf.evaluate(
        wf.hedged_returns(rep["net"], r0050, H50, config.HEDGE_COST_ANNUAL), r0050)
    gross_h = wf.hedged_returns(rep["gross"], tx_ret, H, 0.0)
    sens["完全不計成本（毛報酬、避險免費）"] = wf.evaluate(gross_h, tx_ret)

    result = {"stats": st, "sub": sub, "verdict": v, "flags": flags, "ic": ic, "ic_year": ic_year,
              "best_iteration": best, "h": h, "H": H, "report": rep, "ew": ew, "tx_ret": tx_ret,
              "unhedged": {"vs_ew": (ew_beta, ew_alpha, ew_t), "vs_tx": (tx_beta, tx_alpha, tx_t),
                           "excess_vs_ew": float((rep["net"] - ew.reindex(rep.index)).mean() * 252),
                           "excess_vs_tx": float((rep["net"] - tx_ret.reindex(rep.index)).mean() * 252)},
              "sens": sens,
              "costs": {k: float(r["report"]["cost"].mean() * 252) for k, r in runs.items()},
              "turnover": {k: float(r["report"]["turnover"].mean()) for k, r in runs.items()}}
    with open(OUT.joinpath("result.pkl"), "wb") as f:
        pickle.dump(result, f)
    write_report(result)
    print(f"verdict {v} {flags}  E={st['E']:.4f} t={st['t']:.2f}", flush=True)


def write_report(r: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tw_research import report as rp

    pct = wf.pct
    st, sub = r["stats"], r["sub"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cum = r["h"].loc[config.WF_PERIOD[0]:config.WF_PERIOD[1]].cumsum()
    ax.plot(cum.index, cum.to_numpy(), color=rp.SERIES[0], linewidth=1.8)
    ax.axhline(0, color=rp.AXIS, linewidth=1)
    ax.set_ylabel("累積避險後淨報酬（算術加總）")
    ax.yaxis.set_major_formatter(rp.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    rp.FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(rp.FIG.joinpath("wf_lgb_cum.png"), dpi=150)
    plt.close(fig)

    years = pd.DataFrame({"避險後 h（年化）": st["years"].map(pct),
                          "rank IC": r["ic_year"].map(lambda v: f"{v:+.4f}"),
                          "early stopping 最佳樹數": pd.Series(r["best_iteration"])})
    years.index.name = "年"
    sens = pd.DataFrame({k: {"E": pct(v["E"]), "t": f"{v['t']:.2f}", "上界": pct(v["upper95"]),
                             "正報酬年數": f"{v['n_positive_years']}/{len(v['years'])}"}
                         for k, v in r["sens"].items()}).T
    sens.index.name = "敏感度（只報告）"
    cost_tab = pd.DataFrame({"年化交易成本": {f"{k[0]}，n_drop {k[1]}": pct(v) for k, v in r["costs"].items()},
                             "每日換手（買賣合計）": {f"{k[0]}，n_drop {k[1]}": f"{r['turnover'][k]:.1%}"
                                             for k in r["costs"]}})
    cost_tab.index.name = "情境"
    u = r["unhedged"]
    flags = "、".join(r["flags"]) if r["flags"] else "無"
    text = f"""# 低換手 LightGBM＋台指期避險（walk-forward 2013–2020）

由 `qbot/tw_research/wf_lgb.py` 產生。

<details><summary><b>預先承諾（原文，寫於任何 LightGBM 在 2013–2020 的預測或組合報酬計算之前）</b></summary>

{config.PRE_REGISTRATION_D}
</details>

## 判定：{r['verdict']}

附加標記：{flags}

| 指標 | 全期 2013–2020 | 子期 2018–2020（描述） |
|---|---|---|
| E（避險後淨報酬，年化） | {pct(st['E'])} | {pct(sub['E'])} |
| NW 標準誤 | {pct(st['se'])} | {pct(sub['se'])} |
| t | {st['t']:.2f} | {sub['t']:.2f} |
| 單尾 p | {st['p']:.3f} | {sub['p']:.3f} |
| 95% 上界 | {pct(st['upper95'])} | {pct(sub['upper95'])} |
| MDE（2×SE；事前估計 {pct(config.D_MDE_PRIOR)}） | {pct(st['mde'])} | {pct(sub['mde'])} |
| 正報酬年數 | {st['n_positive_years']} / {len(st['years'])} | {sub['n_positive_years']} / {len(sub['years'])} |
| 前半 2013–2016／後半 2017–2020 | {pct(st['half1'])}／{pct(st['half2'])} | — |
| 剔除最好 1% 交易日後的 E | {pct(st['concentration_E'])} | {pct(sub['concentration_E'])} |
| 避險後對台指期的實現 beta | {st['realized_beta']:.3f} | {sub['realized_beta']:.3f} |

![累積](figures/wf_lgb_cum.png)

{rp.md_table(years)}

未避險組合（描述）：相對台指期超額 {pct(u['excess_vs_tx'])}、beta {u['vs_tx'][0]:.2f}、alpha {pct(u['vs_tx'][1])}（t {u['vs_tx'][2]:.2f}）；
相對 top200 等權超額 {pct(u['excess_vs_ew'])}、beta {u['vs_ew'][0]:.2f}、alpha {pct(u['vs_ew'][1])}（t {u['vs_ew'][2]:.2f}）。
平均避險比率 {r['H'][r['H'] > 0].mean():.2f}。

{rp.md_table(sens)}

{rp.md_table(cost_tab)}
"""
    REPORT.write_text(text, encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    run()
