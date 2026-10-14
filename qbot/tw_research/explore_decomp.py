"""事後探索（描述性，不參與判定）：模型 ETF 輸 0050 的差距從哪裡來。

把「模型 ETF − 0050」的年化報酬差拆成四段（逐段相減，加總等於總差）：
    市值代理誤差  CW50 − 0050          市值前 50 加權（代理）和真正的 0050 差多少
    台積電        CW50 不含台積電 − CW50
    等權起點      同股票池等權 − CW50 不含台積電
    選股          模型 ETF − 同股票池等權（同換倉日、同成本；存活者偏差兩邊都有）
CW 指數每天用前一日收盤市值加權、不扣成本；股票池與前 50 名每月更新（月底分數日的 top200，下一個交易日生效）。
另外算模型分數在市值前 50 名與其餘股票裡的每月 rank IC（分數日 → 下次分數日的持有報酬，和 ETF 同樣的成交日）。

市值代理同 explore_enhanced.load_caps。看過 ETF 模擬的結果之後才做，只用來解釋那份結果。

執行：python qbot/tw_research/explore_decomp.py（需要 etf_sim 的預測快取；--report-only 只重寫報告）
輸出：outputs/tw_research/enhanced/decomp_*.csv、outputs/tw_research/DECOMP_REPORT.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_research import analysis, config, etf_sim, explore_enhanced  # noqa: E402

OUT = explore_enhanced.OUT
REPORT = config.OUTPUT_DIR.joinpath("DECOMP_REPORT.md")
TSMC = explore_enhanced.TSMC
PERIODS = explore_enhanced.PERIODS
FULL = explore_enhanced.FULL
STEPS = ["代理誤差 CW50−0050", "台積電 CW50x−CW50", "等權起點 EW−CW50x", "選股 模型−EW", "合計 模型−0050"]
SIZE_BUCKETS = ("市值前 50", "市值 51 名以後", "全部")


def cagr(r: pd.Series) -> float:
    return analysis.summarize(np.log1p(r.dropna()))["CAGR"]


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
    cap = explore_enhanced.load_caps(symbols, close.index)
    bench = data.load_t1_bars()["close"].pct_change().loc[etf_sim.START:etf_sim.END]
    buy, sell = config.buy_cost_rate(), config.sell_cost_rate()
    trade_days = close.loc[etf_sim.START:etf_sim.END].index
    scores = {k: p.unstack(level="instrument").reindex(trade_days) for k, p in preds.items()}

    msd = etf_sim.score_dates(trade_days, "M")
    pool_at = {s: list(scores["M1"].loc[s].dropna().index) for s in msd}
    eff = pd.Series(msd, index=[etf_sim.next_trade_day(ok, s) or trade_days[-1] for s in msd]).sort_index()
    prev = close.index.get_indexer(trade_days) - 1

    def members(d, top=None):
        s = eff.loc[:d].iloc[-1] if (eff.index <= d).any() else msd[0]
        m = pool_at[s]
        return list(cap.loc[s].reindex(m).dropna().nlargest(top).index) if top else m

    def cw_index(top=None, drop=()):
        out = pd.Series(np.nan, index=trade_days)
        for i, d in enumerate(trade_days):
            m = [x for x in members(d, top) if x not in drop]
            w = cap.iloc[prev[i]].reindex(m)
            r = ret.loc[d].reindex(m)
            good = w.notna() & r.notna() & (w > 0)
            out.iloc[i] = float((w[good] * r[good]).sum() / w[good].sum()) if good.any() else 0.0
        return out

    idx = {"CW50": cw_index(50), "CW50 不含台積電": cw_index(50, (TSMC,)),
           "CW200": cw_index(), "CW200 不含台積電": cw_index(drop=(TSMC,))}
    tsmc_w = pd.Series({s: float(cap.at[s, TSMC] / cap.loc[s].reindex(pool_at[s]).dropna().nlargest(50).sum())
                        for s in msd})
    ladder = {"0050（含息）": bench, **idx}

    dec, sel = [], []
    for fname, f in etf_sim.FREQS.items():
        sd = etf_sim.score_dates(trade_days, f)
        ew_picks = {}
        for s in sd:
            t = etf_sim.next_trade_day(ok, s)
            if t is None:
                break
            ew_picks[t] = [x for x in scores["M1"].loc[s].dropna().index if ok.at[t, x]]
        ew = etf_sim.simulate(ew_picks, ret, buy, sell)
        ladder[f"股票池等權（{fname}換倉）"] = ew
        for n in etf_sim.TOP_NS:
            for key, sc in scores.items():
                r = etf_sim.simulate(etf_sim.model_picks(sc, ok, sd, n), ret, buy, sell)
                for p, (a, b) in PERIODS.items():
                    c = {k: cagr(v.loc[a:b]) for k, v in {"0050": bench, "CW50": idx["CW50"],
                                                          "CW50x": idx["CW50 不含台積電"], "EW": ew, "M": r}.items()}
                    dec.append({"模型": etf_sim.MODEL_NAMES[key], "檔數": n, "換倉": fname, "期間": p,
                                STEPS[0]: c["CW50"] - c["0050"], STEPS[1]: c["CW50x"] - c["CW50"],
                                STEPS[2]: c["EW"] - c["CW50x"], STEPS[3]: c["M"] - c["EW"], STEPS[4]: c["M"] - c["0050"]})
                    x = (r - ew.reindex(r.index)).loc[a:b].dropna()
                    _, _, t = analysis.newey_west_t(x.to_numpy(), config.NW_LAG_DAILY)
                    yr = pd.DataFrame({"m": np.log1p(r.loc[a:b]), "b": np.log1p(ew.reindex(r.index).loc[a:b])}
                                      ).groupby(r.loc[a:b].index.year).sum()
                    sel.append({"模型": etf_sim.MODEL_NAMES[key], "檔數": n, "換倉": fname, "期間": p,
                                "選股 ΔCAGR": c["M"] - c["EW"],
                                "IR": float(x.mean() / x.std() * np.sqrt(analysis.TRADING_DAYS)), "NW t": t,
                                "贏等權年數": int((yr["m"] > yr["b"]).sum()), "年數": int(len(yr))})
        print(f"{fname} done", flush=True)

    ic_rows = []
    days = [etf_sim.next_trade_day(ok, s) for s in msd]
    for key, sc in scores.items():
        for i in range(len(msd) - 1):
            s, t0, t1 = msd[i], days[i], days[i + 1]
            if t0 is None or t1 is None:
                continue
            m = pool_at[s]
            df = pd.DataFrame({"score": sc.loc[s].reindex(m), "fwd": (close.loc[t1] / close.loc[t0] - 1).reindex(m),
                               "caprank": cap.loc[s].reindex(m).rank(ascending=False)}).dropna()
            for bucket, mask in zip(SIZE_BUCKETS, [df["caprank"] <= 50, df["caprank"] > 50, df["caprank"] > 0]):
                d = df[mask]
                ic_rows.append({"模型": etf_sim.MODEL_NAMES[key], "組": bucket, "月": s, "n": len(d),
                                "IC": d["score"].corr(d["fwd"], method="spearman"),
                                "前10減等權": (d.nlargest(10, "score")["fwd"].mean() - d["fwd"].mean())
                                if bucket == SIZE_BUCKETS[0] else np.nan})

    lad = []
    for name, r in ladder.items():
        row = {"組合": name}
        for p, (a, b) in PERIODS.items():
            s = analysis.summarize(np.log1p(r.loc[a:b].dropna()))
            row.update({f"{p} CAGR": s["CAGR"], f"{p} Sharpe": s["Sharpe"], f"{p} MaxDD": s["MaxDD"]})
        lad.append(row)
    pd.DataFrame(lad).to_csv(OUT.joinpath("decomp_ladder.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(dec).to_csv(OUT.joinpath("decomp_steps.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(sel).to_csv(OUT.joinpath("decomp_selection.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(ic_rows).to_csv(OUT.joinpath("decomp_ic_by_size.csv"), index=False, encoding="utf-8-sig")
    tsmc_w.rename("台積電占 CW50 權重").to_csv(OUT.joinpath("decomp_tsmc_weight.csv"), encoding="utf-8-sig")
    corr = float(pd.concat([idx["CW50"], bench], axis=1).dropna().corr().iloc[0, 1])
    pd.Series({"CW50 與 0050 日報酬相關": corr}).to_csv(OUT.joinpath("decomp_proxy_check.csv"), encoding="utf-8-sig")
    print("done", flush=True)


def _p(x, sign=False):
    return f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%"


def write_report():
    from tw_research import report as rp

    lad = pd.read_csv(OUT.joinpath("decomp_ladder.csv")).set_index("組合")
    dec = pd.read_csv(OUT.joinpath("decomp_steps.csv"))
    sel = pd.read_csv(OUT.joinpath("decomp_selection.csv"))
    ic = pd.read_csv(OUT.joinpath("decomp_ic_by_size.csv"), parse_dates=["月"])
    tw = pd.read_csv(OUT.joinpath("decomp_tsmc_weight.csv"), index_col=0, parse_dates=True).iloc[:, 0]
    corr = float(pd.read_csv(OUT.joinpath("decomp_proxy_check.csv"), index_col=0).iloc[0, 0])

    lad_t = pd.DataFrame({f"{p} {k}": lad[f"{p} {k}"].map(_p if k != "Sharpe" else (lambda v: f"{v:.2f}"))
                          for p in PERIODS for k in ("CAGR", "Sharpe")})
    lad_t.columns = [c.replace("CAGR", "年化") for c in lad_t.columns]

    dec_sections, sel_rows = [], []
    for p in PERIODS:
        d = dec[dec["期間"] == p]
        rep = d[(d["檔數"] == etf_sim.REPRESENTATIVE[0]) & (d["換倉"] == etf_sim.REPRESENTATIVE[1])].set_index("模型")
        rep = rep[STEPS].map(lambda v: _p(v, True))
        rep.loc["36 組中位數"] = [_p(d[c].median(), True) for c in STEPS]
        dec_sections.append(f"### {p}\n\n{rp.md_table(rep)}\n")
        s = sel[sel["期間"] == p]
        sel_rows.append({"期間": p, "選股為正的組數": f"{int((s['選股 ΔCAGR'] > 0).sum())}/36",
                         "中位數": _p(s["選股 ΔCAGR"].median(), True), "NW t > 2 的組數": int((s["NW t"] > 2).sum()),
                         "資訊比率中位數": f"{s['IR'].median():.2f}"})
    full_sel = sel[sel["期間"] == FULL].copy()
    full_sel["格"] = full_sel["選股 ΔCAGR"].map(lambda v: _p(v, True)) + "（t " + full_sel["NW t"].map(
        lambda v: f"{v:.1f}") + "）"
    grid = full_sel.pivot_table(index=["模型", "檔數"], columns="換倉", values="格", aggfunc="first", sort=False)
    grid = grid[list(etf_sim.FREQS)]
    grid.index = [f"{m}｜前 {n}" for m, n in grid.index]
    grid.index.name = "模型｜檔數"

    ic_out = []
    for (m, g), d in ic.groupby(["模型", "組"], sort=False):
        row = {"模型": m, "組": g}
        for p, (a, b) in PERIODS.items():
            x = d.set_index("月").loc[a:b]
            mu, _, t = analysis.newey_west_t(x["IC"].to_numpy(), 3)
            row[p] = f"{mu:.3f}（t {t:.1f}）"
        if g == SIZE_BUCKETS[0]:
            mu2, _, t2 = analysis.newey_west_t(d["前10減等權"].to_numpy(), 3)
            row["前 10 名減 50 檔等權（年化、未扣成本）"] = f"{mu2 * 12 * 100:+.1f}%（t {t2:.1f}）"
        ic_out.append(row)
    ic_t = pd.DataFrame(ic_out).fillna("").set_index("模型")

    text = f"""# 模型 ETF 為什麼輸 0050：差距拆解（事後探索）

由 `qbot/tw_research/explore_decomp.py` 產生。**描述性，不參與判定**：看過 ETF 模擬（`ETF_SIM_REPORT.md`）之後才做，
只用來解釋那份結果。

**做法**：把「模型 ETF − 0050」的年化報酬差逐段拆開（加總等於總差）：
市值代理誤差（CW50 − 0050）、台積電（CW50 不含台積電 − CW50）、等權起點（同股票池等權 − CW50 不含台積電）、
選股（模型 ETF − 同股票池等權，同換倉日、同成本）。CW 指數每天用前一日市值加權、不扣成本，前 50 名每月更新。
市值 = 未還原收盤價 × 2026-10 的發行股數（近似）；CW50 和 0050 的日報酬相關 {corr:.3f}。
台積電占 CW50 的權重：2013 年初 {tw.iloc[0]:.0%}、2020 年底 {tw.loc[:'2020-12-31'].iloc[-1]:.0%}、最後 {tw.iloc[-1]:.0%}。

## 各組合的報酬

{rp.md_table(lad_t)}

## 拆解（代表設定：前 {etf_sim.REPRESENTATIVE[0]} 名、{etf_sim.REPRESENTATIVE[1]}換倉；最後一列是 36 組的中位數）

{chr(10).join(dec_sections)}
## 選股：模型 ETF 減同股票池等權

{rp.md_table(pd.DataFrame(sel_rows).set_index("期間"))}

全期各組（年化報酬差、日超額的 Newey-West t）：

{rp.md_table(grid)}

## 大型股之間也有預測力嗎：每月 rank IC（分數日 → 下次分數日的報酬）

{rp.md_table(ic_t)}
"""
    REPORT.write_text(text, encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    if "--report-only" not in sys.argv:
        run()
    write_report()
