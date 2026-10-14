"""事後探索（描述性，不參與判定）：讓增強型指數的加減碼不偏向低波動股（beta 中性）。

第五輪（wf_label60.py，60 日標籤）的加減碼四個都判 (?)，描述數字顯示加減碼一致偏向低 beta（約 −0.055），
扣掉樣本內 beta 後超額 +1.0%～+1.6%。這裡檢查「事前就把加減碼的 beta 調成 0」實際做得到多少。
這個想法是看過第五輪結果才提出的，而 2026-09-30 之後還沒有新資料，所以只能用已看過的 2013–2026/09，結果不是證據。

設計在執行前固定，不依結果調整：
- 分數：第五輪的 60 日標籤預測（快取），四個模型；其餘規則同第五輪主設定（每季、±1.5%、台積電 ≤25% 的市值前 50）。
- beta：每個分數日，個股對 0050 含息日報酬的 OLS beta，取分數日（含）以前 250 個交易日，有效觀測少於 120 記為 1，
  截斷在 [0, 2]（同第三輪 wf_hedge.stock_betas）。
- 中性化：加減碼後的權重 w、基準權重 b，對台積電以外的 49 檔加上 δ_i = k × b_i ×（β_i − β̄），β̄ 是 b 加權平均，
  k 取使 Σ (w_i + δ_i − b_i) β_i = 0；再把負權重設為 0、等比例縮放回 1 − 台積電權重。台積電權重不變。
- 隨機對照：隨機分數＋同樣的中性化，300 次（seed 同 explore_enhanced）。

執行：python qbot/tw_research/explore_beta_neutral.py（需要 wf_label60 與 etf_sim 的預測快取；--report-only 只重寫報告）
輸出：outputs/tw_research/beta_neutral/、outputs/tw_research/BETA_NEUTRAL_REPORT.md
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_research import analysis, config, etf_sim, explore_enhanced as xe, wf_label60  # noqa: E402

OUT = config.OUTPUT_DIR.joinpath("beta_neutral")
REPORT = config.OUTPUT_DIR.joinpath("BETA_NEUTRAL_REPORT.md")
BETA_HISTORY_START = "2011-06-01"  # 第一個分數日（2013-01-02）之前要有 250 個交易日


def active_beta(w: pd.Series, bw: pd.Series, beta: pd.Series) -> float:
    others = bw.index.drop(xe.TSMC, errors="ignore")
    return float(((w.reindex(others).fillna(0.0) - bw.reindex(others)) * beta.reindex(others).fillna(1.0)).sum())


def neutralize(w: pd.Series, bw: pd.Series, beta: pd.Series) -> pd.Series:
    """把台積電以外 49 檔的加減碼對 beta 的曝險調成 0（依基準權重比例調整，台積電不動）。"""
    others = bw.index.drop(xe.TSMC, errors="ignore")
    b = bw.reindex(others)
    be = beta.reindex(others).fillna(1.0)
    dev = be - (b * be).sum() / b.sum()
    v = float((b * dev * be).sum())
    k = -active_beta(w, bw, be) / v if v > 0 else 0.0
    new = (w.reindex(others).fillna(0.0) + k * b * dev).clip(lower=0.0)
    new = new / new.sum() * b.sum()
    return pd.concat([w.reindex([xe.TSMC]).dropna(), new])


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    from qlib.data import D

    from tw_research import data, wf_hedge

    ctx = xe.load_context()
    preds, _ = wf_label60.predictions()
    l60 = {k: p.unstack(level="instrument").reindex(ctx.trade_days) for k, p in preds.items()}

    close = D.features(list(ctx.close.columns), ["$close"], start_time=BETA_HISTORY_START, end_time=etf_sim.END)
    stock_ret = etf_sim.daily_returns(close["$close"].unstack(level="instrument").sort_index())
    bench_ret = data.load_t1_bars()["close"].pct_change()
    betas = {s: wf_hedge.stock_betas(stock_ret, bench_ret, s, caps.index)
             for s, _, caps in xe.rebalance_plan(ctx, xe.FREQ)}
    print("betas done", flush=True)

    def post(s, bw, w):
        return neutralize(w, bw, betas[s])

    series, turnover, ex_ante = {}, {}, {}

    def go(name, targets):
        series[name], turnover[name] = xe.simulate_weights(targets, ctx.ret, ctx.ok, ctx.buy, ctx.sell)

    go("B", xe.build_targets(ctx, xe.FREQ, xe.TILT))
    plan = xe.rebalance_plan(ctx, xe.FREQ)
    for k in etf_sim.MODEL_NAMES:
        fn = xe.table_scores(l60[k])
        raw = xe.build_targets(ctx, xe.FREQ, xe.TILT, fn)
        neu = xe.build_targets(ctx, xe.FREQ, xe.TILT, fn, post=post)
        go(f"原始|{k}", raw)
        go(f"中性|{k}", neu)
        ex_ante[k] = pd.DataFrame({
            "原始": [active_beta(raw[t], xe.bench_weights(caps), betas[s]) for s, t, caps in plan],
            "中性": [active_beta(neu[t], xe.bench_weights(caps), betas[s]) for s, t, caps in plan],
        }, index=[s for s, _, _ in plan])
        print(f"{k} done", flush=True)
    rand = xe.random_tilts(ctx, post=post)
    print("random done", flush=True)
    with open(OUT.joinpath("beta_neutral.pkl"), "wb") as f:
        pickle.dump({"series": series, "turnover": turnover, "ex_ante": ex_ante, "random": rand}, f)
    print("done", flush=True)


def _pct(x, d=2):
    return "—" if not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def write_report():
    from tw_research import report as rp

    with open(OUT.joinpath("beta_neutral.pkl"), "rb") as f:
        res = pickle.load(f)
    series, turnover, ex_ante, rand = res["series"], res["turnover"], res["ex_ante"], res["random"]
    b = series["B"]
    names = etf_sim.MODEL_NAMES
    years = (pd.Timestamp(etf_sim.END) - pd.Timestamp(etf_sim.START)).days / 365.25
    rand_e = np.array([wf_label60.effect((r - b).dropna())["E"] for r in rand])

    rows, yearly = [], {}
    for k, v in names.items():
        neu, raw = wf_label60.effect((series[f"中性|{k}"] - b).dropna()), wf_label60.effect((series[f"原始|{k}"] - b).dropna())
        x = xe.active_stats(series[f"中性|{k}"], b)
        rows.append({"模型": v, "中性 E（年化）": _pct(neu["E"]), "t": f"{neu['t']:.2f}",
                     "2013–2020／2021–2026": f"{_pct(neu['halves'][0])}／{_pct(neu['halves'][1])}",
                     "E>0 年數": f"{neu['n_positive_years']}/{len(neu['years'])}",
                     "剔除最好 1% 日": _pct(neu["concentration_E"]),
                     "事後 beta": f"{x['β']:+.3f}", "第五輪（未中性）E": _pct(raw["E"]),
                     "年換手 中性／未中性": f"{turnover[f'中性|{k}'].sum() / years:.0%}／{turnover[f'原始|{k}'].sum() / years:.0%}",
                     "隨機百分位": f"{np.mean(rand_e < neu['E']):.0%}"})
        yearly[v] = neu["years"]
    main = pd.DataFrame(rows).set_index("模型")
    yearly = pd.DataFrame(yearly).map(lambda x: _pct(x, 1))
    yearly.index.name = "年"
    ea = pd.DataFrame({v: {"事前 beta 平均（未中性）": f"{ex_ante[k]['原始'].mean():+.3f}",
                           "事前 beta 平均（中性後）": f"{ex_ante[k]['中性'].mean():+.3f}",
                           "中性後最大絕對值": f"{ex_ante[k]['中性'].abs().max():.3f}"}
                       for k, v in names.items()}).T
    ea.index.name = "模型"
    q = np.quantile(rand_e, [0.05, 0.5, 0.95])
    text = f"""# beta 中性的增強型指數加減碼（事後探索）

由 `qbot/tw_research/explore_beta_neutral.py` 產生。**描述性，不參與判定，不是證據**：這個想法是看過第五輪
（`LABEL60_REPORT.md`）的結果才提出的，而且只能用已看過的 2013–2026/09；要當證據必須另立預先承諾、用 2026-09-30 之後的資料。

**設計**（執行前固定）：第五輪的 60 日標籤分數與主設定（每季、±1.5%、台積電 ≤25% 的市值前 50）。
加減碼之後，把台積電以外 49 檔對 0050 的事前 beta（過去 250 日）曝險調成 0：依基準權重比例，加碼高 beta、減碼低 beta，
再把負權重設為 0、縮放回原本的總權重。台積電權重不變。

## 結果（相對基準 B；E 是日超額報酬算術平均 × 252）

{rp.md_table(main)}

隨機分數＋同樣的中性化（{xe.RANDOM_DRAWS} 次）的 E：5%／50%／95% 分位 {_pct(q[0])}／{_pct(q[1])}／{_pct(q[2])}。

事前 beta 曝險（每季調整日，加減碼後對 0050 的 beta 減基準）：

{rp.md_table(ea)}

逐年 E（中性後）：

{rp.md_table(yearly)}
"""
    REPORT.write_text(text, encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    if "--report-only" not in sys.argv:
        run()
    write_report()
