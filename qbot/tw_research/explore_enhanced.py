"""事後探索（描述性，不參與判定；2013-01 至 2026-09 全部是已看過的期間）：台積電中立的增強型指數。

回答「若要做一檔基金，怎麼用模型的選股能力」：不讓模型決定台積電與大小型股的比重（那是大賭注、模型沒有資訊），
只在大型股之間依分數小幅加減碼。設計在執行前固定，不依結果調整：

- 基準 B：每季最後一個交易日，取當時股票池（top200，以 M1 有分數的股票為準）中市值代理前 50 名；
  台積電權重 = min(市值權重, 25%)（2026-04 金管會「台積電條款」下主動基金的上限），其餘 49 檔依市值分配剩下的權重。
  下一個有成交的交易日收盤調整到目標權重，之後隨價格漂移到下次調整；扣散戶成本（同 etf_sim）。
- 增強 E（M1–M4 各一）：同樣的 50 檔、台積電權重和 B 相同；其餘 49 檔依模型分數排名加減碼
  a = TILT ×（u − 0.5），u =（排名 − 0.5）／n，分數最高 +1.5%、最低 −1.5%；權重 = max(基準 + a, 0)，
  再把 49 檔等比例縮放到 1 − 台積電權重。沒有分數的股票不加減碼。
- 無法交易（當天漲停或沒有成交）的股票：維持漂移後的權重，其餘依目標比例分配。
- 隨機對照：分數換成隨機數，300 次，看 E 對 B 的資訊比率落在哪個百分位。
- 敏感度（只報告）：加減碼加倍（±3%）；每半年調整。
- 另列兩個參考基準：台積電不設上限（CW50）、完全不含台積電，用來拆出台積電的影響。

市值代理 = Yahoo 未還原股利的收盤價 × python_AlphaGPT 在 2026-10-13 抓的發行股數（沒有歷史股數與自由流通調整，
而且用到資料期間之後的股數）；股票池只含現存公司。兩者對 B 與 E 的影響大致相同，E − B 受影響較小。

執行：python qbot/tw_research/explore_enhanced.py（需要 etf_sim 的預測快取；--report-only 只重寫報告）
輸出：outputs/tw_research/enhanced/、outputs/tw_research/ENHANCED_INDEX_REPORT.md
"""

import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_research import analysis, config, etf_sim  # noqa: E402

OUT = config.OUTPUT_DIR.joinpath("enhanced")
REPORT = config.OUTPUT_DIR.joinpath("ENHANCED_INDEX_REPORT.md")
SHARES_SOURCE = Path(r"C:\Users\User\Desktop\Python_projects\python_AlphaGPT\share_counts_cache.json")
SHARES_CACHE = config.BASE_DIR.joinpath("share_counts_2026-10-13.json")
TSMC = "TW2330"
TSMC_CAP = 0.25
N_BENCH = 50
TILT = 0.03  # 分數最高 +1.5%、最低 −1.5%
FREQ = "Q"
SENSITIVITY = {"加減碼加倍（±3%）": {"tilt": 0.06, "freq": "Q"}, "每半年調整": {"tilt": TILT, "freq": "H"}}
REFERENCE_CAPS = {"CW50（台積電不設上限）": 1.0, "CW50 不含台積電": 0.0}
RANDOM_DRAWS = 300
RANDOM_SEED = 20261015
PERIODS = {"全期 2013–2026/09": (etf_sim.START, etf_sim.END), **etf_sim.SUBPERIODS}
FULL = "全期 2013–2026/09"


# ---------------------------------------------------------------- 權重
def bench_weights(caps: pd.Series, tsmc_cap: float = TSMC_CAP) -> pd.Series:
    """市值加權，台積電最多 tsmc_cap，超出的部分依市值分給其他股票。"""
    w = caps / caps.sum()
    if TSMC in w.index and w[TSMC] > tsmc_cap:
        rest = w.drop(TSMC)
        w = pd.concat([pd.Series({TSMC: tsmc_cap}), rest / rest.sum() * (1 - tsmc_cap)])
    return w


def tilt_weights(bw: pd.Series, scores: pd.Series, tilt: float = TILT) -> pd.Series:
    """台積電以外的股票依分數排名加減碼，權重不為負，總和維持 1 − 台積電權重。"""
    others = bw.drop(TSMC, errors="ignore")
    s = scores.reindex(others.index)
    u = (s.rank(method="first") - 0.5) / s.notna().sum()
    w = (others + (tilt * (u - 0.5)).fillna(0.0)).clip(lower=0.0)
    w = w / w.sum() * others.sum()
    return pd.concat([bw.reindex([TSMC]).dropna(), w])


# ---------------------------------------------------------------- 模擬
def simulate_weights(targets: Dict[pd.Timestamp, pd.Series], ret: pd.DataFrame, ok: pd.DataFrame,
                     buy: float, sell: float):
    """targets: 交易日 -> 當天收盤的目標權重。回傳（每日淨報酬, 每次調整的買進比例）。"""
    days = list(targets)
    idx = ret.index
    pos = {d: i for i, d in enumerate(idx)}
    out = pd.Series(0.0, index=idx)
    cur = pd.Series(dtype=float)
    bought = {}
    for k, d in enumerate(days):
        i0 = pos[d]
        i1 = pos[days[k + 1]] if k + 1 < len(days) else len(idx) - 1
        tgt = targets[d][targets[d] > 0]
        names = cur.index.union(tgt.index)
        can = ok.loc[d].reindex(names).fillna(False).astype(bool)
        locked = cur.reindex(names).fillna(0.0)[~can]
        free = tgt.reindex(names).fillna(0.0)[can]
        new = cur if free.sum() <= 0 else pd.concat([locked, free / free.sum() * (1 - locked.sum())])
        new = new[new > 0]
        delta = new.subtract(cur, fill_value=0.0)
        bought[d] = float(delta.clip(lower=0).sum())
        cost = bought[d] * buy + float((-delta.clip(upper=0)).sum()) * sell
        out.iloc[i0] = (1 + out.iloc[i0]) * (1 - cost) - 1
        r = ret.iloc[i0 + 1:i1 + 1][new.index].fillna(0.0)
        if r.empty:
            cur = new
            continue
        growth = (1 + r).cumprod()
        value = growth.mul(new, axis=1).sum(axis=1)
        out.iloc[i0 + 1:i1 + 1] = (value / value.shift(1).fillna(1.0) - 1).to_numpy()
        last = growth.iloc[-1] * new
        cur = last / last.sum()
    return out.iloc[pos[days[0]]:], pd.Series(bought)


# ---------------------------------------------------------------- 資料
def load_caps(symbols, index: pd.DatetimeIndex) -> pd.DataFrame:
    if not SHARES_CACHE.exists():
        shutil.copy(SHARES_SOURCE, SHARES_CACHE)
    sh = json.loads(SHARES_CACHE.read_text(encoding="utf-8"))
    shares = {**sh["tpex"]["shares"], **sh["twse"]["shares"]}
    src = config.BASE_DIR.joinpath("source")
    raw = {}
    for s in symbols:
        f = src.joinpath(f"{s}.csv")
        if f.exists() and s[2:] in shares:
            raw[s] = pd.read_csv(f, usecols=["date", "close"], index_col="date", parse_dates=True)["close"]
    raw = pd.DataFrame(raw).reindex(index).ffill()
    return raw.mul(pd.Series({s: shares[s[2:]] for s in raw.columns}))


def stats(r: pd.Series) -> dict:
    s = analysis.summarize(np.log1p(r.dropna()))
    return {"CAGR": s["CAGR"], "Sharpe": s["Sharpe"], "MaxDD": s["MaxDD"]}


def active_stats(r: pd.Series, b: pd.Series) -> dict:
    x = (r - b.reindex(r.index)).dropna()
    _, _, t = analysis.newey_west_t(x.to_numpy(), config.NW_LAG_DAILY)
    yr = pd.DataFrame({"r": np.log1p(r), "b": np.log1p(b.reindex(r.index))}).groupby(r.index.year).sum()
    bb = b.reindex(x.index)
    beta = float(np.cov(x, bb, ddof=1)[0, 1] / bb.var())
    return {"ΔCAGR": stats(r)["CAGR"] - stats(b)["CAGR"], "β": beta,
            "α": float((x - beta * bb).mean() * analysis.TRADING_DAYS),
            "TE": float(x.std() * np.sqrt(analysis.TRADING_DAYS)),
            "IR": float(x.mean() / x.std() * np.sqrt(analysis.TRADING_DAYS)), "NW t": t,
            "贏的年數": int((yr["r"] > yr["b"]).sum()), "年數": int(len(yr))}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    from tw_research import data, factors

    factors.init_qlib()
    preds = etf_sim.predictions()
    from qlib.data import D

    symbols = sorted(set().union(*[p.index.get_level_values("instrument").unique() for p in preds.values()]))
    close = D.features(symbols, ["$close"], start_time="2012-12-01", end_time=etf_sim.END)["$close"]
    close = close.unstack(level="instrument").sort_index().loc[:etf_sim.END]
    ok = etf_sim.tradeable(close)
    ret = etf_sim.daily_returns(close)
    cap = load_caps(symbols, close.index)
    bench0050 = data.load_t1_bars()["close"].pct_change().loc[etf_sim.START:etf_sim.END]
    buy, sell = config.buy_cost_rate(), config.sell_cost_rate()
    trade_days = close.loc[etf_sim.START:etf_sim.END].index
    scores = {k: p.unstack(level="instrument").reindex(trade_days) for k, p in preds.items()}

    def plan(freq):
        """[(分數日, 成交日, 前 50 檔市值)]"""
        rows = []
        for s in etf_sim.score_dates(trade_days, freq):
            t = etf_sim.next_trade_day(ok, s)
            if t is None:
                break
            pool = scores["M1"].loc[s].dropna().index
            rows.append((s, t, cap.loc[s].reindex(pool).dropna().nlargest(N_BENCH)))
        return rows

    def run_set(freq, tilt, score_fn, tsmc_cap=TSMC_CAP):
        targets = {}
        for s, t, caps in plan_cache[freq]:
            bw = bench_weights(caps, tsmc_cap)
            targets[t] = bw if score_fn is None else tilt_weights(bw, score_fn(s, bw.index), tilt)
        return targets

    plan_cache = {f: plan(f) for f in {FREQ, *[v["freq"] for v in SENSITIVITY.values()]}}
    series, turnover, weights = {}, {}, {}

    def go(name, targets):
        series[name], turnover[name] = simulate_weights(targets, ret, ok, buy, sell)
        weights[name] = targets

    go("B", run_set(FREQ, TILT, None))
    for name, c in REFERENCE_CAPS.items():
        go(name, run_set(FREQ, TILT, None, tsmc_cap=c))
    for sname, sp in SENSITIVITY.items():
        go(f"B|{sname}", run_set(sp["freq"], sp["tilt"], None))
    for key in scores:
        go(key, run_set(FREQ, TILT, lambda s, names, key=key: scores[key].loc[s].reindex(names)))
        for sname, sp in SENSITIVITY.items():
            go(f"{key}|{sname}", run_set(sp["freq"], sp["tilt"],
                                         lambda s, names, key=key: scores[key].loc[s].reindex(names)))
        print(f"{key} done", flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    rand = []
    for _ in range(RANDOM_DRAWS):
        r, _ = simulate_weights(run_set(FREQ, TILT, lambda s, names: pd.Series(rng.random(len(names)), index=names)),
                                ret, ok, buy, sell)
        rand.append(r)
    print("random done", flush=True)
    with open(OUT.joinpath("enhanced.pkl"), "wb") as f:
        pickle.dump({"series": series, "turnover": turnover, "random": rand, "bench0050": bench0050,
                     "max_other_weight": {k: max(float(w.drop(TSMC, errors="ignore").max()) for w in v.values())
                                          for k, v in weights.items()}}, f)
    print("done", flush=True)


# ---------------------------------------------------------------- 報告
def _pct(x, d=1, sign=False):
    return "—" if not np.isfinite(x) else (f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%")


def _chart(series, rand, path):
    import matplotlib.pyplot as plt

    from tw_research import report as rp

    b = series["B"]
    cum = lambda r: np.log1p(r - b.reindex(r.index)).cumsum()  # noqa: E731
    band = pd.DataFrame({i: cum(r) for i, r in enumerate(rand)})
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.fill_between(band.index, band.quantile(0.05, axis=1), band.quantile(0.95, axis=1), color=rp.BENCH,
                    alpha=0.25, linewidth=0, label="隨機分數 5%～95%")
    for i, (k, name) in enumerate(etf_sim.MODEL_NAMES.items()):
        c = cum(series[k])
        ax.plot(c.index, c.to_numpy(), color=rp.SERIES[i], linewidth=1.6, label=name)
    ax.axhline(0, color=rp.INK, linewidth=0.8)
    ax.set_ylabel("累積超額（對數，對基準 B）")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("加減碼部位的累積超額報酬（每季調整、±1.5%）", loc="left", fontsize=9, color=rp.INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report():
    import matplotlib

    matplotlib.use("Agg")
    from tw_research import report as rp

    with open(OUT.joinpath("enhanced.pkl"), "rb") as f:
        res = pickle.load(f)
    series, turnover, rand, b50 = res["series"], res["turnover"], res["random"], res["bench0050"]
    years = (pd.Timestamp(etf_sim.END) - pd.Timestamp(etf_sim.START)).days / 365.25
    names = {"B": "基準 B（台積電 ≤25%）", **{k: k for k in REFERENCE_CAPS},
             **{k: f"增強｜{v}" for k, v in etf_sim.MODEL_NAMES.items()}}

    def yr_wins(r, b):
        yr = pd.DataFrame({"r": np.log1p(r), "b": np.log1p(b.reindex(r.index).fillna(0))}).groupby(r.index.year).sum()
        return f"{int((yr['r'] > yr['b']).sum())}/{len(yr)}"

    lad = []
    for k, label in [("0050", "0050（含息）"), *names.items()]:
        row = {"組合": label}
        for per, (a, z) in PERIODS.items():
            r = (b50 if k == "0050" else series[k]).loc[a:z]
            s = stats(r)
            row[f"{per} 年化"] = _pct(s["CAGR"])
            row[f"{per} Sharpe"] = f"{s['Sharpe']:.2f}"
        full = (b50 if k == "0050" else series[k]).loc[etf_sim.START:etf_sim.END]
        row["最大回撤"] = _pct(stats(full)["MaxDD"])
        row["年換手（買進）"] = "—" if k == "0050" else _pct(turnover[k].sum() / years, 0)
        row["贏 0050 年數"] = "—" if k == "0050" else yr_wins(full, b50)
        lad.append(row)
    lad = pd.DataFrame(lad).set_index("組合")

    rand_stats = {per: pd.DataFrame([active_stats(r.loc[a:z], series["B"].loc[a:z]) for r in rand])
                  for per, (a, z) in PERIODS.items()}
    rand_full = rand_stats[FULL]
    act = []
    for k, v in etf_sim.MODEL_NAMES.items():
        for per, (a, z) in PERIODS.items():
            s = active_stats(series[k].loc[a:z], series["B"].loc[a:z])
            act.append({"模型": v, "期間": per, "超額年化": _pct(s["ΔCAGR"], 1, True), "追蹤誤差": _pct(s["TE"]),
                        "資訊比率": f"{s['IR']:.2f}", "NW t": f"{s['NW t']:.2f}",
                        "贏 B 年數": f"{s['贏的年數']}/{s['年數']}",
                        "隨機分數百分位": f"{np.mean(rand_stats[per]['IR'].to_numpy() < s['IR']):.0%}",
                        "超額的 beta": f"{s['β']:+.3f}", "扣 beta 後": _pct(s["α"], 1, True)})
    act = pd.DataFrame(act).set_index("模型")

    sens = []
    for k, v in etf_sim.MODEL_NAMES.items():
        for sname in SENSITIVITY:
            s = active_stats(series[f"{k}|{sname}"], series[f"B|{sname}"])
            sens.append({"模型": v, "設定": sname, "超額年化": _pct(s["ΔCAGR"], 1, True), "追蹤誤差": _pct(s["TE"]),
                         "資訊比率": f"{s['IR']:.2f}", "NW t": f"{s['NW t']:.2f}",
                         "贏 B 年數": f"{s['贏的年數']}/{s['年數']}"})
    sens = pd.DataFrame(sens).set_index("模型")

    rp.FIG.mkdir(parents=True, exist_ok=True)
    _chart(series, rand, rp.FIG.joinpath("enhanced_active.png"))
    q = rand_full.quantile([0.05, 0.5, 0.95])
    max_w = max(res["max_other_weight"][k] for k in ["B", *etf_sim.MODEL_NAMES])
    text = f"""# 台積電中立的增強型指數（事後探索）

由 `qbot/tw_research/explore_enhanced.py` 產生。**描述性回測，不參與判定**：2013-01 至 2026-09 全部是已看過的期間，
模型預測是 walk-forward 樣本外（同 `etf_sim.py`），但這個設計是看過 ETF 模擬與差距拆解之後才提出的。
設計在執行前固定（見程式開頭），沒有依結果調整。

**設計**：每季取市值代理前 50 檔，台積電 = min(市值權重, 25%)，其餘依市值加權 → 基準 B。
增強組合的台積電權重和 B 相同，其餘 49 檔依模型分數排名加減碼（最高 +1.5%、最低 −1.5%，權重不為負）。
隔一個交易日收盤成交、期間不再平衡，扣散戶成本，不扣經理費。

**限制**：市值 = 未還原收盤價 × 2026-10 的發行股數（沒有歷史股數與自由流通調整）；股票池只含現存公司；
台積電條款是 2026-04 才有的規定，之前的年份是假設性的。B 與增強組合裡台積電以外單一個股的目標權重最高 {_pct(max_w)}
（法規上限 10%；這個設計沒有另設 10% 上限，少數調整日略為超過）。

## 各組合和 0050

{rp.md_table(lad)}

## 加減碼本身：增強組合減基準 B

隨機分數百分位：同樣的加減碼規則改用隨機分數（{RANDOM_DRAWS} 次），資訊比率贏過幾成的隨機版本。
隨機版本全期的資訊比率 5%／50%／95% 分位：{q.loc[0.05, 'IR']:.2f}／{q.loc[0.5, 'IR']:.2f}／{q.loc[0.95, 'IR']:.2f}
（超額年化 {_pct(q.loc[0.05, 'ΔCAGR'], 1, True)}／{_pct(q.loc[0.5, 'ΔCAGR'], 1, True)}／{_pct(q.loc[0.95, 'ΔCAGR'], 1, True)}）。

{rp.md_table(act)}

![加減碼的累積超額](figures/enhanced_active.png)

## 敏感度（只報告）

{rp.md_table(sens)}

## 欄位說明

- **年換手（買進）**：每年買進金額占淨值的比例；賣出大致相同。
- **超額年化**：年化報酬差（增強 − B）。**追蹤誤差**：日超額報酬的年化標準差。**資訊比率**：年化超額平均 ÷ 追蹤誤差。
- **超額的 beta**：日超額報酬對 B 的迴歸係數；負值代表加減碼偏向低波動股，在多頭裡會少賺。
  **扣 beta 後**：日超額報酬減掉 beta × B 之後的年化平均（算術平均，和上面的年化報酬差口徑不同）。
- **NW t**：日超額報酬的 Newey-West t（落後 {config.NW_LAG_DAILY} 日）。2026 年只到 9 月。
"""
    REPORT.write_text(text, encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    if "--report-only" not in sys.argv:
        run()
    write_report()
