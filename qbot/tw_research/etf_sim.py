"""把四個多因子模型模擬成 ETF：前 N 名等權、定期換倉，和 0050 比較（描述性回測，不是預先承諾的檢驗）。

模型都每年重訓一次（walk-forward），預測 2013-01 至 2026-09 全部是樣本外：
    M1 Linear(OLS)+Alpha158   訓練 2008…Y−1（同 C 部分，wf_hedge.py）
    M2 LightGBM+Alpha158      訓練 2008…Y−2、Y−1 早停（同 D 部分，wf_lgb.py）
    M3 XGBoost+Alpha158       同 M2 的切分
    M4 LightGBM+Alpha360      同 M2 的切分
2013–2020 的 M1、M2 預測沿用 C、D 已存的檔案；其餘年份在這裡補訓並快取。

ETF 規則：每個換倉期最後一個交易日依分數取前 N 名，隔日收盤等權買進（當日漲停的跳過、改買下一名），
持有到下次換倉、期間不再平衡；成本 = 買進成交額 × 手續費 + 賣出成交額 ×（手續費 + 證交稅）。
不扣經理費（0050 的報酬已扣它自己的費用，所以這裡對模擬 ETF 略為有利）。
隨機 ETF：同樣的 N 與換倉日，從當天有分數（屬於 top200）的股票隨機抽 N 檔，用來分辨選股能力與股票池偏差。

執行：python qbot/tw_research/etf_sim.py
輸出：outputs/tw_research/etf_sim/、outputs/tw_research/ETF_SIM_REPORT.md
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_research import analysis, config, factors  # noqa: E402

OUT = config.OUTPUT_DIR.joinpath("etf_sim")
REPORT = config.OUTPUT_DIR.joinpath("ETF_SIM_REPORT.md")
YEARS = tuple(range(2013, 2027))
START, END = "2013-01-01", config.DATA_END
TOP_NS = (10, 20, 30)
FREQS = {"每月": "M", "每季": "Q", "每半年": "H"}
SUBPERIODS = {"2013–2020": ("2013-01-01", "2020-12-31"), "2021–2026/09": ("2021-01-01", END)}
RANDOM_DRAWS = 300
RANDOM_SEED = 20261014
MODEL_NAMES = {"M1": "M1 線性", "M2": "M2 LightGBM", "M3": "M3 XGBoost", "M4": "M4 Alpha360"}


# ---------------------------------------------------------------- walk-forward 預測
def _gbdt_segments(year: int) -> dict:
    end = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(END)).strftime("%Y-%m-%d")
    return {
        "train": (config.WF_TRAIN_START, f"{year - 2}-{config.WF_TRAIN_END_MMDD}"),
        "valid": (f"{year - 1}-01-01", f"{year - 1}-{config.WF_TRAIN_END_MMDD}"),
        "test": (f"{year}-01-01", end),
    }


def _shared_handler(kind: str):
    """M2–M4 的前處理沒有需要 fit 的步驟（label 只做逐日橫斷面標準化），所以整段載入一次、各年共用。"""
    from qlib.utils import init_instance_by_config

    kwargs = {"start_time": config.WF_TRAIN_START, "end_time": END, "instruments": config.UNIVERSE,
              # 沒有需要 fit 的前處理，fit 區間不影響結果；仍填一個在所有訓練期內的區間
              "fit_start_time": config.WF_TRAIN_START, "fit_end_time": "2010-12-20"}
    if kind == "alpha360":
        kwargs.update(infer_processors=[], learn_processors=[
            {"class": "DropnaLabel"}, {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}])
        return init_instance_by_config({"class": "Alpha360", "module_path": "qlib.contrib.data.handler",
                                        "kwargs": kwargs})
    return init_instance_by_config({"class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                                    "kwargs": kwargs})


def _gbdt_fit_predict(handler, model_cfg: dict, year: int, tag: str) -> pd.Series:
    from qlib.data.dataset import DatasetH
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    dataset = DatasetH(handler=handler, segments=_gbdt_segments(year))
    model = init_instance_by_config(model_cfg)
    with R.start(experiment_name="tw_etf_sim", recorder_name=f"{tag}_{year}"):
        model.fit(dataset)
        pred = model.predict(dataset, segment="test")
    return pred.rename("score")


def _cached(path: Path, fn):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    out = fn()
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def _old_pred(folder: str, year: int) -> pd.Series:
    with open(config.OUTPUT_DIR.joinpath(folder, f"pred_{year}.pkl"), "rb") as f:
        return pickle.load(f)[0]


def predictions() -> Dict[str, pd.Series]:
    from tw_research import wf_hedge

    cache = OUT.joinpath("preds")
    cache.mkdir(parents=True, exist_ok=True)
    out = {}

    # M1：2013–2020 沿用 C；之後每年照 C 的方式重訓（前處理在訓練期 fit，所以每年重新載入）
    preds = []
    for y in YEARS:
        if y <= 2020:
            preds.append(_old_pred("wf_hedge", y))
        else:
            preds.append(_cached(cache.joinpath(f"M1_{y}.pkl"), lambda y=y: wf_hedge.fit_predict_year(y)[0]))
        print(f"M1 {y} ok", flush=True)
    out["M1"] = pd.concat(preds)

    gbdt = {
        "M2": ("alpha158", {"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
                            "kwargs": dict(factors._LGB_158, seed=config.MODEL_SEED)}),
        "M3": ("alpha158", factors.MODELS["M3_XGB_A158"][0]),
        "M4": ("alpha360", {"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
                            "kwargs": dict(factors._LGB_360, seed=config.MODEL_SEED)}),
    }
    handlers = {}
    for key, (kind, cfg) in gbdt.items():
        preds = []
        for y in YEARS:
            if key == "M2" and y <= 2020:
                preds.append(_old_pred("wf_lgb", y))  # D 部分的同一套切分與參數
            else:
                def fit(y=y, kind=kind, cfg=cfg, key=key):
                    if kind not in handlers:
                        handlers.clear()  # Alpha158 與 Alpha360 不同時留在記憶體
                        handlers[kind] = _shared_handler(kind)
                    return _gbdt_fit_predict(handlers[kind], cfg, y, key)
                preds.append(_cached(cache.joinpath(f"{key}_{y}.pkl"), fit))
            print(f"{key} {y} ok", flush=True)
        out[key] = pd.concat(preds)
    return {k: v.loc[START:END].sort_index() for k, v in out.items()}


# ---------------------------------------------------------------- ETF 模擬
def score_dates(trade_days: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """換倉的分數日：第一個交易日（建倉）＋每個期間的最後一個交易日。"""
    if freq == "M":
        key = trade_days.to_period("M")
    elif freq == "Q":
        key = trade_days.to_period("Q")
    else:
        key = pd.Index([f"{d.year}H{1 if d.month <= 6 else 2}" for d in trade_days])
    last = trade_days[~pd.Index(key).duplicated(keep="last")]
    return pd.DatetimeIndex([trade_days[0]]).append(last[last > trade_days[0]]).unique()


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    """停牌日報酬為 0（價格沿用前一日），復牌當天反映停牌期間的變動。"""
    return close.ffill().pct_change(fill_method=None)


def simulate(picks: Dict[pd.Timestamp, List[str]], ret: pd.DataFrame, buy: float, sell: float) -> pd.Series:
    """picks: 交易日 -> 當天收盤要持有的股票。回傳每日淨報酬（第一個交易日只有建倉成本）。"""
    days = list(picks)
    idx = ret.index
    out = pd.Series(0.0, index=idx)
    prev_w = pd.Series(dtype=float)
    pos = {d: i for i, d in enumerate(idx)}
    for k, d in enumerate(days):
        i0 = pos[d]
        i1 = pos[days[k + 1]] if k + 1 < len(days) else len(idx) - 1
        names = picks[d]
        new_w = pd.Series(1.0 / len(names), index=names)
        delta = new_w.subtract(prev_w, fill_value=0.0)
        cost = delta.clip(lower=0).sum() * buy + (-delta.clip(upper=0)).sum() * sell
        out.iloc[i0] = (1 + out.iloc[i0]) * (1 - cost) - 1  # 成本在成交日收盤扣
        r = ret.iloc[i0 + 1:i1 + 1][names].fillna(0.0)
        if r.empty:
            prev_w = new_w
            continue
        growth = (1 + r).cumprod()
        value = growth.mean(axis=1)
        out.iloc[i0 + 1:i1 + 1] = (value / value.shift(1).fillna(1.0) - 1).to_numpy()
        prev_w = growth.iloc[-1] / growth.iloc[-1].sum()
    first = pos[days[0]]
    return out.iloc[first:]


def tradeable(close: pd.DataFrame) -> pd.DataFrame:
    """可買進：當天有收盤價、且收盤相對前收沒有漲停。"""
    chg = close / close.ffill().shift(1) - 1
    thr = pd.Series(np.where(close.index >= pd.Timestamp(config.LIMIT_CHANGE_DATE),
                             config.LIMIT_THRESHOLD_NEW, config.LIMIT_THRESHOLD_OLD), index=close.index)
    return close.notna() & ~chg.ge(thr, axis=0)


_ACTIVE_DAYS = {}


def next_trade_day(ok: pd.DataFrame, s):
    """分數日之後第一個有個股可交易的日子。加權指數日曆裡有幾天颱風休市（個股沒有成交），要跳過。"""
    key = id(ok)
    if key not in _ACTIVE_DAYS:
        _ACTIVE_DAYS[key] = ok.index[ok.any(axis=1).to_numpy()]
    days = _ACTIVE_DAYS[key]
    j = days.searchsorted(s, side="right")
    return days[j] if j < len(days) else None


def model_picks(scores: pd.DataFrame, ok: pd.DataFrame, sdates, n: int) -> Dict[pd.Timestamp, List[str]]:
    picks = {}
    for s in sdates:
        t = next_trade_day(ok, s)
        if t is None:
            break
        ranked = scores.loc[s].dropna().sort_values(ascending=False).index
        ranked = [x for x in ranked if x in ok.columns and ok.at[t, x]]
        picks[t] = ranked[:n]
    return picks


def random_picks(scores: pd.DataFrame, ok: pd.DataFrame, sdates, n: int, rng) -> Dict[pd.Timestamp, List[str]]:
    picks = {}
    for s in sdates:
        t = next_trade_day(ok, s)
        if t is None:
            break
        pool = [x for x in scores.loc[s].dropna().index if x in ok.columns and ok.at[t, x]]
        picks[t] = list(rng.choice(pool, size=n, replace=False))
    return picks


def metrics(r: pd.Series, bench: pd.Series) -> dict:
    r = r.dropna()
    b = bench.reindex(r.index).fillna(0.0)
    lr, lb = np.log1p(r), np.log1p(b)
    s, sb = analysis.summarize(lr), analysis.summarize(lb)
    ex = r - b
    beta = float(np.cov(r, b, ddof=1)[0, 1] / b.var())
    yr = pd.DataFrame({"r": lr, "b": lb}).groupby(r.index.year).sum()
    return {"CAGR": s["CAGR"], "Sharpe": s["Sharpe"], "MaxDD": s["MaxDD"], "Vol": s["Vol"],
            "0050 CAGR": sb["CAGR"], "0050 Sharpe": sb["Sharpe"], "0050 MaxDD": sb["MaxDD"],
            "ΔCAGR": s["CAGR"] - sb["CAGR"], "ΔSharpe": s["Sharpe"] - sb["Sharpe"], "beta": beta,
            "TE": float(ex.std() * np.sqrt(analysis.TRADING_DAYS)),
            "IR": float(ex.mean() / ex.std() * np.sqrt(analysis.TRADING_DAYS)) if ex.std() > 0 else np.nan,
            "贏 0050 年數": int((yr["r"] > yr["b"]).sum()), "年數": int(len(yr))}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    factors.init_qlib()
    preds = predictions()
    from qlib.data import D

    from tw_research import data

    symbols = sorted(set().union(*[p.index.get_level_values("instrument").unique() for p in preds.values()]))
    df = D.features(symbols, ["$close"], start_time="2012-12-01", end_time=END)
    close = df["$close"].unstack(level="instrument").sort_index().loc[:END]
    ok = tradeable(close)
    ret = daily_returns(close)
    bench = data.load_t1_bars()["close"].pct_change().loc[START:END]
    buy, sell = config.buy_cost_rate(), config.sell_cost_rate()
    trade_days = close.loc[START:END].index
    scores = {k: p.unstack(level="instrument").reindex(trade_days) for k, p in preds.items()}

    rows, series = [], {}
    for freq_name, freq in FREQS.items():
        sdates = score_dates(trade_days, freq)
        for n in TOP_NS:
            for key, sc in scores.items():
                r = simulate(model_picks(sc, ok, sdates, n), ret, buy, sell)
                series[(key, n, freq_name)] = r
                for per, (a, b) in {"全期 2013–2026/09": (START, END), **SUBPERIODS}.items():
                    rows.append({"模型": MODEL_NAMES[key], "檔數": n, "換倉": freq_name, "期間": per,
                                 **metrics(r.loc[a:b], bench)})
            rng = np.random.default_rng([RANDOM_SEED, n, len(sdates)])
            rand = []
            for _ in range(RANDOM_DRAWS):
                rr = simulate(random_picks(scores["M1"], ok, sdates, n, rng), ret, buy, sell)
                rand.append({per: metrics(rr.loc[a:b], bench)
                             for per, (a, b) in {"全期 2013–2026/09": (START, END), **SUBPERIODS}.items()})
            series[("random", n, freq_name)] = rand
            print(f"{freq_name} done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT.joinpath("etf_grid.csv"), index=False, encoding="utf-8-sig")
    with open(OUT.joinpath("etf_sim.pkl"), "wb") as f:
        pickle.dump({"table": table, "series": series, "bench": bench}, f)
    print("done", flush=True)


# ---------------------------------------------------------------- 報告
REPRESENTATIVE = (20, "每季")  # 淨值圖固定用這一組（事先選定，不是挑最好的）
FULL = "全期 2013–2026/09"


def _pct(x, d=1):
    return "—" if not np.isfinite(x) else f"{x * 100:.{d}f}%"


def random_frame(series, n, freq_name, per) -> pd.DataFrame:
    return pd.DataFrame([d[per] for d in series[("random", n, freq_name)]])


def _heatmap(t, series, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from tw_research import report as rp

    freqs = list(FREQS)
    rows_lbl, mat = [], []
    for m in MODEL_NAMES.values():
        for n in TOP_NS:
            rows_lbl.append(f"{m}｜前 {n}")
            sel = t[(t["期間"] == FULL) & (t["模型"] == m) & (t["檔數"] == n)].set_index("換倉")
            mat.append([float(sel.loc[f, "ΔSharpe"]) for f in freqs])
    for n in TOP_NS:
        rows_lbl.append(f"隨機選股中位數｜前 {n}")
        mat.append([float(random_frame(series, n, f, FULL)["ΔSharpe"].median()) for f in freqs])
    mat = np.array(mat)
    lim = float(np.nanmax(np.abs(mat)))
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows_lbl) + 1.4))
    ax.imshow(mat, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.2f}".replace("−", "-"), ha="center", va="center", fontsize=8, color=rp.INK)
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels(freqs)
    ax.set_yticks(range(len(rows_lbl)))
    ax.set_yticklabels(rows_lbl, fontsize=8)
    ax.tick_params(axis="y", colors=rp.INK)
    ax.grid(False)
    for y in (2.5, 5.5, 8.5, 11.5):
        ax.axhline(y, color=rp.SURFACE, linewidth=3)
    ax.set_title("Sharpe 減 0050 的 Sharpe（2013–2026/09；藍＝贏、紅＝輸）", loc="left", fontsize=9, color=rp.INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _equity(series, bench, path):
    import matplotlib.pyplot as plt

    from tw_research import report as rp

    n, fq = REPRESENTATIVE
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (k, name) in enumerate(MODEL_NAMES.items()):
        c = (1 + series[(k, n, fq)].loc[START:END]).cumprod()
        ax.plot(c.index, c.to_numpy(), color=rp.SERIES[i], linewidth=1.6, label=name)  # 線尾會重疊，只用圖例
    b = (1 + bench.reindex(c.index).fillna(0)).cumprod()
    ax.plot(b.index, b.to_numpy(), color=rp.INK, linewidth=1.6, label="0050（含息）")
    ax.annotate("0050", (b.index[-1], b.iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8,
                color=rp.INK, va="center")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(rp.FixedLocator([1, 2, 5, 10, 20]))
    ax.yaxis.set_major_formatter(rp.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(rp.FixedLocator([]))
    ax.set_ylabel("淨值（起點＝1，對數刻度）")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.margins(x=0.12)
    ax.set_title(f"前 {n} 名、{fq}換倉（事先固定的代表設定）", loc="left", fontsize=9, color=rp.INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _model_table(t, per) -> pd.DataFrame:
    freqs = list(FREQS)
    order = list(MODEL_NAMES.values())
    sub = t[t["期間"] == per].copy()
    sub["_m"] = sub["模型"].map(order.index)
    sub["_f"] = sub["換倉"].map(freqs.index)
    sub = sub.sort_values(["_m", "檔數", "_f"])
    return pd.DataFrame({
        "檔數": sub["檔數"].to_numpy(), "換倉": sub["換倉"].to_numpy(),
        "年化報酬": sub["CAGR"].map(_pct).to_numpy(), "Sharpe": sub["Sharpe"].map(lambda v: f"{v:.2f}").to_numpy(),
        "最大回撤": sub["MaxDD"].map(_pct).to_numpy(),
        "報酬差 vs 0050": sub["ΔCAGR"].map(lambda v: f"{v * 100:+.1f}%").to_numpy(),
        "Sharpe 差 vs 0050": sub["ΔSharpe"].map(lambda v: f"{v:+.2f}").to_numpy(),
        "beta": sub["beta"].map(lambda v: f"{v:.2f}").to_numpy(),
        "贏 0050 年數": (sub["贏 0050 年數"].astype(str) + "/" + sub["年數"].astype(str)).to_numpy(),
        "隨機 ETF 百分位": sub["隨機 ETF 百分位"].map(lambda v: f"{v:.0%}").to_numpy(),
    }, index=pd.Index(sub["模型"].to_numpy(), name="模型"))


def _random_table(series, per) -> pd.DataFrame:
    rows = []
    for n in TOP_NS:
        for f in FREQS:
            rand = random_frame(series, n, f, per)
            rows.append({"檔數": n, "換倉": f, "年化報酬中位數": _pct(rand["CAGR"].median()),
                         "Sharpe 中位數": f"{rand['Sharpe'].median():.2f}",
                         "Sharpe 5%～95%": f"{rand['Sharpe'].quantile(0.05):.2f}～{rand['Sharpe'].quantile(0.95):.2f}",
                         "Sharpe 贏 0050 的比例": f"{(rand['ΔSharpe'] > 0).mean():.0%}"})
    return pd.DataFrame(rows).set_index("檔數")


def write_report():
    import matplotlib

    matplotlib.use("Agg")
    from tw_research import report as rp

    with open(OUT.joinpath("etf_sim.pkl"), "rb") as f:
        res = pickle.load(f)
    t, series, bench = res["table"], res["series"], res["bench"]
    t["隨機 ETF 百分位"] = [
        float((random_frame(series, r["檔數"], r["換倉"], r["期間"])["Sharpe"] < r["Sharpe"]).mean())
        for _, r in t.iterrows()
    ]
    rp.FIG.mkdir(parents=True, exist_ok=True)
    _heatmap(t, series, rp.FIG.joinpath("etf_delta_sharpe.png"))
    _equity(series, bench, rp.FIG.joinpath("etf_equity.png"))

    def bench_line(per):
        r = t[t["期間"] == per].iloc[0]
        return (f"0050（含息）：年化報酬 {_pct(r['0050 CAGR'])}、Sharpe {r['0050 Sharpe']:.2f}、"
                f"最大回撤 {_pct(r['0050 MaxDD'])}。")

    fullt = t[t["期間"] == FULL]
    best = fullt.loc[fullt["Sharpe"].idxmax()]
    beat = {per: int((t[t["期間"] == per]["ΔSharpe"] > 0).sum()) for per in [FULL, *SUBPERIODS]}
    sections = []
    for per in [FULL, *SUBPERIODS]:
        sections.append(f"## {per}\n\n{bench_line(per)}\n\n{rp.md_table(_model_table(t, per))}\n\n"
                        f"隨機選股 ETF（同股票池、同檔數與換倉日，各 {RANDOM_DRAWS} 次）：\n\n"
                        f"{rp.md_table(_random_table(series, per))}\n")
    text = f"""# 四個多因子模型模擬成 ETF：和 0050 比較

由 `qbot/tw_research/etf_sim.py` 產生。**描述性回測，不是預先承諾的檢驗**：36 組設定裡表現最好的那組有選擇偏差，
要和下方「隨機選股 ETF」的分布一起看。

**設定**：模型每年用過去的資料重新訓練（walk-forward），2013-01 至 2026-09 全部是樣本外預測。
每個換倉期最後一個交易日依分數取前 N 名，隔日收盤等權買進（當天漲停買不到就換下一名），持有到下次換倉；
扣散戶交易成本（手續費 0.1425%×2.8 折、賣出證交稅 0.3%），**不扣 ETF 經理費**（0050 的報酬已扣它自己的費用）。
股票池是每季流動性前 200 名，但清單只含現存公司（存活者偏差）；隨機選股 ETF 有同樣的偏差，可以用來對照。

## 摘要

- {bench_line(FULL)}
- 36 組中全期 Sharpe 高於 0050 的有 **{beat[FULL]} 組**（2013–2020：{beat['2013–2020']} 組；2021–2026/09：{beat['2021–2026/09']} 組）。
- 全期 Sharpe 最高的是 {best['模型']}、前 {best['檔數']} 名、{best['換倉']}換倉：Sharpe {best['Sharpe']:.2f}
  （0050 {best['0050 Sharpe']:.2f}），年化報酬 {_pct(best['CAGR'])}（0050 {_pct(best['0050 CAGR'])}），
  在同設定的隨機選股 ETF 中贏過 {best['隨機 ETF 百分位']:.0%}。

![Sharpe 差](figures/etf_delta_sharpe.png)

![代表設定的淨值](figures/etf_equity.png)

{chr(10).join(sections)}
## 欄位說明

- **報酬差、Sharpe 差**：模擬 ETF 減 0050。Sharpe 未扣無風險利率。
- **beta**：日報酬對 0050 的敏感度。
- **贏 0050 年數**：日曆年含息報酬高於 0050 的年數（2026 只到 9 月）。
- **隨機 ETF 百分位**：同樣檔數與換倉頻率下，Sharpe 贏過幾成的隨機選股 ETF；50% 代表和亂挑差不多。
"""
    REPORT.write_text(text, encoding="utf-8")
    t.to_csv(OUT.joinpath("etf_grid.csv"), index=False, encoding="utf-8-sig")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    if "--report-only" not in sys.argv:
        run()
    write_report()
