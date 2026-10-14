"""E 部分：台積電中立增強型指數＋60 日標籤（walk-forward 2013–2026/09）。

預先承諾見 config.PRE_REGISTRATION_E（報告開頭原文貼上）。只檢驗一件事：四個模型的訓練目標從隔天報酬（L1）
改成和每季持有期一致的 60 日報酬（L60）之後，增強型指數（explore_enhanced 的規則，不改）的加減碼相對基準 B 有沒有超額。

模型同 etf_sim（特徵、前處理、參數、seed），只改標籤（config.E_LABEL，前處理一律 DropnaLabel＋CSRankNorm）
與切分（訓練、early stopping 都止於 9/20，標籤不跨進下一段）。

執行：python qbot/tw_research/wf_label60.py（--report-only 只重寫報告）
輸出：outputs/tw_research/label60/、outputs/tw_research/LABEL60_REPORT.md
"""

import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_research import analysis, config, etf_sim, explore_enhanced as xe  # noqa: E402

OUT = config.OUTPUT_DIR.joinpath("label60")
REPORT = config.OUTPUT_DIR.joinpath("LABEL60_REPORT.md")
END = config.E_PERIOD[1]
LABEL_PROCESSORS = [{"class": "DropnaLabel"}, {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}]
ETF_TOP_N = 20


# ---------------------------------------------------------------- walk-forward 預測
def cutoff(year: int) -> str:
    return f"{year}-{config.E_TRAIN_END_MMDD}"


def gbdt_segments(year: int) -> dict:
    test_end = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(END)).strftime("%Y-%m-%d")
    return {"train": (config.WF_TRAIN_START, cutoff(year - 2)),
            "valid": (f"{year - 1}-01-01", cutoff(year - 1)),
            "test": (f"{year}-01-01", test_end)}


def linear_handler(year: int) -> dict:
    from tw_research import wf_hedge

    cfg = wf_hedge.handler_config(year)  # RobustZScoreNorm、Fillna；label 已是 DropnaLabel＋CSRankNorm
    cfg["kwargs"].update(fit_end_time=cutoff(year - 1), label=config.E_LABEL)
    return cfg


def shared_handler(kind: str):
    """M2–M4 的前處理沒有需要 fit 的步驟，整段載入一次、各年共用（同 etf_sim._shared_handler，只換標籤）。"""
    from qlib.utils import init_instance_by_config

    kwargs = {"start_time": config.WF_TRAIN_START, "end_time": END, "instruments": config.UNIVERSE,
              "fit_start_time": config.WF_TRAIN_START, "fit_end_time": cutoff(2010),
              "label": config.E_LABEL, "learn_processors": LABEL_PROCESSORS}
    if kind == "alpha360":
        kwargs["infer_processors"] = []
    cls = "Alpha360" if kind == "alpha360" else "Alpha158"
    return init_instance_by_config({"class": cls, "module_path": "qlib.contrib.data.handler", "kwargs": kwargs})


def fit_linear(year: int) -> dict:
    from qlib.contrib.model.linear import LinearModel
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    dataset = init_instance_by_config({
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {"handler": linear_handler(year),
                   "segments": {"train": (config.WF_TRAIN_START, cutoff(year - 1)),
                                "test": (f"{year}-01-01", f"{year}-12-31")}},
    })
    model = LinearModel(estimator="ols")
    with R.start(experiment_name="tw_label60", recorder_name=f"M1_{year}"):
        model.fit(dataset)
        pred = model.predict(dataset, segment="test")
    return {"score": pred.rename("score"), "n_trees": None}


def fit_gbdt(handler, model_cfg: dict, year: int, key: str) -> dict:
    from qlib.data.dataset import DatasetH
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    dataset = DatasetH(handler=handler, segments=gbdt_segments(year))
    model = init_instance_by_config(model_cfg)
    with R.start(experiment_name="tw_label60", recorder_name=f"{key}_{year}"):
        model.fit(dataset)
        pred = model.predict(dataset, segment="test")
    best = getattr(getattr(model, "model", None), "best_iteration", None)
    if best is not None:
        best = int(best) + (1 if "XGB" in model_cfg["class"] else 0)  # LightGBM 從 1 起算、XGBoost 從 0 起算
    return {"score": pred.rename("score"), "n_trees": best}


def predictions():
    """回傳（{模型: 分數}, {模型: {年: early stopping 的樹數}}）。"""
    cache = OUT.joinpath("preds")
    cache.mkdir(parents=True, exist_ok=True)
    out, trees = {}, {}
    parts = []
    for y in config.E_YEARS:
        parts.append(etf_sim._cached(cache.joinpath(f"M1_{y}.pkl"), lambda y=y: fit_linear(y)))
        print(f"M1 {y} ok", flush=True)
    out["M1"], trees["M1"] = pd.concat([p["score"] for p in parts]), {}
    handlers = {}
    for key, (kind, cfg) in etf_sim.GBDT_MODELS.items():
        parts = []
        for y in config.E_YEARS:
            def fit(y=y, kind=kind, cfg=cfg, key=key):
                if kind not in handlers:
                    handlers.clear()  # Alpha158 與 Alpha360 不同時留在記憶體
                    handlers[kind] = shared_handler(kind)
                return fit_gbdt(handlers[kind], cfg, y, key)
            parts.append(etf_sim._cached(cache.joinpath(f"{key}_{y}.pkl"), fit))
            print(f"{key} {y} ok", flush=True)
        out[key] = pd.concat([p["score"] for p in parts])
        trees[key] = {y: p["n_trees"] for y, p in zip(config.E_YEARS, parts)}
    return {k: v.loc[config.E_PERIOD[0]:END].sort_index() for k, v in out.items()}, trees


# ---------------------------------------------------------------- 判定
def effect(a: pd.Series) -> dict:
    """a：日超額報酬（增強 − B）。E 是算術平均 × 252。"""
    a = a.dropna()
    mu, se, t = analysis.newey_west_t(a.to_numpy(), config.NW_LAG_DAILY)
    td = analysis.TRADING_DAYS
    yearly = a.groupby(a.index.year).mean() * td
    keep = a[a < a.quantile(1 - config.E_CONCENTRATION_TOP)]
    return {"E": mu * td, "se": se * td, "t": t, "p": analysis.one_sided_p(t, len(a)), "mde": 2 * se * td,
            "upper95": (mu + 1.645 * se) * td,
            "halves": [float(a.loc[x:y].mean() * td) for x, y in config.E_HALVES],
            "years": yearly, "n_positive_years": int((yearly > 0).sum()),
            "concentration_E": float(keep.mean() * td)}


def verdict(st: dict, holm_reject: bool) -> str:
    if (holm_reject and st["E"] >= config.E_MATERIALITY and all(h > 0 for h in st["halves"])
            and st["n_positive_years"] >= config.E_MIN_POSITIVE_YEARS and st["concentration_E"] > 0):
        return "(+)"
    if st["upper95"] < config.E_MATERIALITY:
        return "(−)"
    return "(?)"


def judge(active: Dict[str, pd.Series]) -> Dict[str, dict]:
    stats = {k: effect(a) for k, a in active.items()}
    adj = analysis.holm({k: s["p"] for k, s in stats.items()})
    for k, s in stats.items():
        s["p_holm"], s["holm_reject"] = adj[k]
        s["verdict"] = verdict(s, s["holm_reject"])
    return stats


# ---------------------------------------------------------------- 執行
def quarterly_ic(ctx, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """每季：分數日的分數 vs 本次成交日到下次成交日的報酬（股票池全部、市值前 50）。"""
    plan = xe.rebalance_plan(ctx, xe.FREQ)
    rows = []
    for (s, t0, caps), (_, t1, _) in zip(plan[:-1], plan[1:]):
        pool = ctx.scores["M1"].loc[s].dropna().index
        fwd = (ctx.close.loc[t1] / ctx.close.loc[t0] - 1).reindex(pool)
        for name, table in tables.items():
            sc = table.loc[s].reindex(pool)
            for group, names in (("全部", pool), ("市值前 50", caps.index)):
                d = pd.DataFrame({"s": sc.reindex(names), "f": fwd.reindex(names)}).dropna()
                rows.append({"分數": name, "組": group, "季": s, "IC": d["s"].corr(d["f"], method="spearman")})
    return pd.DataFrame(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = xe.load_context()
    preds, trees = predictions()
    l60 = {k: p.unstack(level="instrument").reindex(ctx.trade_days) for k, p in preds.items()}

    series, turnover = {}, {}

    def go(name, targets):
        series[name], turnover[name] = xe.simulate_weights(targets, ctx.ret, ctx.ok, ctx.buy, ctx.sell)

    go("B", xe.build_targets(ctx, xe.FREQ, xe.TILT))
    for sname, sp in xe.SENSITIVITY.items():
        go(f"B|{sname}", xe.build_targets(ctx, sp["freq"], sp["tilt"]))
    for k in etf_sim.MODEL_NAMES:
        go(f"L60|{k}", xe.build_targets(ctx, xe.FREQ, xe.TILT, xe.table_scores(l60[k])))
        go(f"L1|{k}", xe.build_targets(ctx, xe.FREQ, xe.TILT, xe.table_scores(ctx.scores[k])))
        for sname, sp in xe.SENSITIVITY.items():
            go(f"L60|{k}|{sname}", xe.build_targets(ctx, sp["freq"], sp["tilt"], xe.table_scores(l60[k])))
        print(f"{k} portfolios done", flush=True)
    rand = xe.random_tilts(ctx)
    print("random done", flush=True)

    ic = quarterly_ic(ctx, {**{f"L60|{k}": t for k, t in l60.items()},
                            **{f"L1|{k}": ctx.scores[k] for k in etf_sim.MODEL_NAMES}})

    sdates = etf_sim.score_dates(ctx.trade_days, xe.FREQ)
    etf = {}
    for k in etf_sim.MODEL_NAMES:
        for tag, table in (("L60", l60[k]), ("L1", ctx.scores[k])):
            etf[f"{tag}|{k}"] = etf_sim.simulate(etf_sim.model_picks(table, ctx.ok, sdates, ETF_TOP_N),
                                                 ctx.ret, ctx.buy, ctx.sell)
    ew_picks = {}
    for s in sdates:
        t = etf_sim.next_trade_day(ctx.ok, s)
        if t is None:
            break
        ew_picks[t] = [x for x in ctx.scores["M1"].loc[s].dropna().index if ctx.ok.at[t, x]]
    etf["股票池等權"] = etf_sim.simulate(ew_picks, ctx.ret, ctx.buy, ctx.sell)

    with open(OUT.joinpath("label60.pkl"), "wb") as f:
        pickle.dump({"series": series, "turnover": turnover, "random": rand, "ic": ic, "etf": etf,
                     "trees": trees, "bench0050": ctx.bench0050}, f)
    print("done", flush=True)


# ---------------------------------------------------------------- 報告
def _pct(x, d=2, sign=True):
    return "—" if x is None or not np.isfinite(x) else (f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%")


def write_report():
    import matplotlib

    matplotlib.use("Agg")
    from tw_research import report as rp

    with open(OUT.joinpath("label60.pkl"), "rb") as f:
        res = pickle.load(f)
    series, turnover, rand, ic, etf = res["series"], res["turnover"], res["random"], res["ic"], res["etf"]
    b, b50 = series["B"], res["bench0050"]
    names = etf_sim.MODEL_NAMES
    years = (pd.Timestamp(END) - pd.Timestamp(config.E_PERIOD[0])).days / 365.25

    active = {k: (series[f"L60|{k}"] - b).dropna() for k in names}
    st = judge(active)
    rand_ir = np.array([xe.active_stats(r, b)["IR"] for r in rand])

    main = []
    for k, v in names.items():
        s = st[k]
        main.append({"模型": v, "判定": s["verdict"], "E（年化）": _pct(s["E"]), "NW 標準誤": _pct(s["se"], sign=False),
                     "t": f"{s['t']:.2f}", "單尾 p": f"{s['p']:.3f}", "Holm p": f"{s['p_holm']:.3f}",
                     "95% 上界": _pct(s["upper95"]), "MDE": _pct(s["mde"], sign=False),
                     "2013–2020／2021–2026": f"{_pct(s['halves'][0])}／{_pct(s['halves'][1])}",
                     "E>0 年數": f"{s['n_positive_years']}/{len(s['years'])}",
                     "剔除最好 1% 日": _pct(s["concentration_E"])})
    main = pd.DataFrame(main).set_index("模型")

    yearly = pd.DataFrame({v: st[k]["years"] for k, v in names.items()})
    yearly.index.name = "年"
    yearly = yearly.map(lambda x: _pct(x, 1))

    desc = []
    for k, v in names.items():
        a60, a1 = st[k], effect((series[f"L1|{k}"] - b).dropna())
        diff = effect((series[f"L60|{k}"] - series[f"L1|{k}"]).dropna())
        x = xe.active_stats(series[f"L60|{k}"], b)
        desc.append({"模型": v, "E：L60": _pct(a60["E"]), "E：L1": _pct(a1["E"]),
                     "L60 − L1": f"{_pct(diff['E'])}（t {diff['t']:.2f}）",
                     "年換手 L60／L1": f"{turnover[f'L60|{k}'].sum() / years:.0%}／{turnover[f'L1|{k}'].sum() / years:.0%}",
                     "超額的 beta": f"{x['β']:+.3f}", "扣 beta 後": _pct(x["α"]),
                     "隨機分數百分位": f"{np.mean(rand_ir < x['IR']):.0%}"})
    desc = pd.DataFrame(desc).set_index("模型")

    ic_rows = []
    for (name, group), d in ic.groupby(["分數", "組"], sort=False):
        tag, k = name.split("|")
        mu, _, t = analysis.newey_west_t(d["IC"].to_numpy(), 1)
        ic_rows.append({"模型": names[k], "組": group, "標籤": tag, "IC": f"{mu:.3f}（t {t:.1f}）"})
    ic_t = pd.DataFrame(ic_rows).pivot_table(index=["模型", "組"], columns="標籤", values="IC", aggfunc="first",
                                             sort=False)[["L60", "L1"]]
    ic_t.index = [f"{m}｜{g}" for m, g in ic_t.index]
    ic_t.index.name = "模型｜股票"

    sens = []
    for k, v in names.items():
        for sname in xe.SENSITIVITY:
            s = effect((series[f"L60|{k}|{sname}"] - series[f"B|{sname}"]).dropna())
            sens.append({"模型": v, "設定": sname, "E（年化）": _pct(s["E"]), "t": f"{s['t']:.2f}",
                         "E>0 年數": f"{s['n_positive_years']}/{len(s['years'])}"})
    sens = pd.DataFrame(sens).set_index("模型")

    def cagr(r):
        return analysis.summarize(np.log1p(r.dropna()))

    etf_rows = []
    for name, r in {"0050（含息）": b50, "股票池等權（每季）": etf["股票池等權"],
                    **{f"{v}｜{tag}": etf[f"{tag}|{k}"] for k, v in names.items() for tag in ("L60", "L1")}}.items():
        s = cagr(r)
        etf_rows.append({"組合": name, "年化報酬": _pct(s["CAGR"], 1, False), "Sharpe": f"{s['Sharpe']:.2f}",
                         "最大回撤": _pct(s["MaxDD"], 1, False)})
    etf_t = pd.DataFrame(etf_rows).set_index("組合")

    tree_t = pd.DataFrame(res["trees"]).drop(columns="M1", errors="ignore").rename(columns=names)
    tree_t.index.name = "預測年"

    bs = cagr(b)
    text = f"""# 台積電中立增強型指數＋60 日標籤（walk-forward 2013–2026/09）

由 `qbot/tw_research/wf_label60.py` 產生。

<details><summary><b>預先承諾（原文，寫於任何 60 日標籤模型的訓練、預測或組合報酬計算之前）</b></summary>

{config.PRE_REGISTRATION_E}
</details>

## 判定（4 項，Holm 校正；材料性門檻 {_pct(config.E_MATERIALITY, 1, False)}，事前 MDE {_pct(config.E_MDE_PRIOR, 1, False)}）

基準 B（台積電 ≤25% 的市值前 50）：年化 {_pct(bs['CAGR'], 1, False)}、Sharpe {bs['Sharpe']:.2f}。
E 是增強組合減 B 的日超額報酬算術平均 × 252。

{rp.md_table(main)}

逐年 E：

{rp.md_table(yearly)}

## 描述（不參與判定）

和隔天報酬版本（L1，同樣規則重算）比較、換手、beta、隨機分數百分位（{xe.RANDOM_DRAWS} 次）：

{rp.md_table(desc)}

每季 rank IC（分數日 → 本次成交日到下次成交日的報酬；NW t，lag 1）：

{rp.md_table(ic_t)}

前 {ETF_TOP_N} 名等權、每季換倉的 ETF（`etf_sim` 規則）：

{rp.md_table(etf_t)}

early stopping 的樹數（M1 是 OLS）：

{rp.md_table(tree_t)}

## 敏感度（只報告，不改判定）

{rp.md_table(sens)}
"""
    REPORT.write_text(text, encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    if "--report-only" not in sys.argv:
        run()
    write_report()
