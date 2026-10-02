"""讀 run_strategies.py / run_factors.py 的結果，畫圖並寫 outputs/tw_research/REPORT.md。

報告開頭原文貼上 config.PRE_REGISTRATION；表格與判定全部由程式產生。
執行：python qbot/tw_research/report.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_research import analysis, config  # noqa: E402
from tw_research.strategies import NOT_PORTED, STRATEGY_BY_CODE  # noqa: E402

OUT = config.OUTPUT_DIR
FIG = OUT.joinpath("figures")

# 參考調色盤（dataviz skill references/palette.md，淺色模式）
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BENCH = "#898781"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft JhengHei", "Microsoft YaHei", "Segoe UI", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})


def pct(x, digits=2):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def num(x, digits=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.to_numpy()]) + " |")
    return "\n".join(lines)


def load(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- 圖
def effect_plot(names, effect, lo, hi, materiality, verdicts, xlabel, path, fmt=lambda v: f"{v:.2f}"):
    """每項一列：效果量點 + 95% 區間，0 與材料性門檻為參考線，右側標判定。"""
    n = len(names)
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * n + 1.2))
    y = np.arange(n)[::-1]
    ax.hlines(y, lo, hi, color=SERIES[0], linewidth=2)
    ax.plot(effect, y, "o", color=SERIES[0], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.axvline(materiality, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(materiality, n - 0.35, f" 門檻 {fmt(materiality)}", color=INK2, va="bottom", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.tick_params(axis="y", colors=INK)
    xmax = np.nanmax(hi)
    for yi, v in zip(y, verdicts):
        # 正黑體沒有 U+2212（−），圖上改用連字號
        ax.text(1.01, yi, v.replace("−", "-"), transform=ax.get_yaxis_transform(), va="center",
                color=INK, fontsize=9)
    ax.set_xlabel(xlabel.replace("−", "-"))
    ax.set_ylim(-0.7, n - 0.3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def small_multiples(curves: pd.DataFrame, bench: pd.Series, titles, path, bench_label, ncols=4):
    codes = [c for c in curves.columns]
    nrows = int(np.ceil(len(codes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.2 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    b = np.exp(bench.fillna(0).cumsum())
    for ax, code in zip(axes, codes):
        s = np.exp(curves[code].fillna(0).cumsum())
        ax.plot(b.index, b.to_numpy(), color=BENCH, linewidth=1.2)
        ax.plot(s.index, s.to_numpy(), color=SERIES[0], linewidth=1.6)
        ax.set_yscale("log")
        ax.set_title(titles.get(code, code), fontsize=9, color=INK, loc="left")
    for ax in axes[len(codes):]:
        ax.set_visible(False)
    axes[0].xaxis.set_major_locator(mdates.YearLocator(4))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].yaxis.set_minor_locator(FixedLocator([]))
    handles = [plt.Line2D([], [], color=SERIES[0], linewidth=1.6), plt.Line2D([], [], color=BENCH, linewidth=1.2)]
    fig.legend(handles, ["策略（扣成本）", bench_label], loc="upper right", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def line_chart(series: dict, path, ylabel, title=None):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (name, s) in enumerate(series.items()):
        color = BENCH if name.startswith("基準") else SERIES[i % len(SERIES)]
        ax.plot(s.index, s.to_numpy(), color=color, linewidth=1.8 if not name.startswith("基準") else 1.2)
        ax.annotate(name, (s.index[-1], s.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", color=INK)
    ax.legend(list(series), frameon=False, loc="lower left", fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- A
def section_t1(d) -> str:
    t1 = d["t1"]
    start, end = d["window"]
    rows = pd.DataFrame({
        "策略": t1["strategy"],
        "淨 Sharpe": t1["full_rule_sharpe"].map(num),
        "對照中位數": t1["full_null_median"].map(num),
        "ΔSharpe": t1["full_delta_sharpe"].map(num),
        "SD₀": t1["full_sd0"].map(num),
        "MDE": t1["full_mde"].map(num),
        "p": t1["full_p"].map(num),
        "Holm p": t1["holm_p"].map(num),
        "軌 A Δ": t1["A_delta_sharpe"].map(num),
        "軌 B Δ": t1["B_delta_sharpe"].map(num),
        "正年度": t1["full_positive_year_share"].map(lambda v: f"{v:.0%}"),
        "集中度": t1["concentration_ok"].map(lambda v: "通過" if v else "未通過"),
        "上界": t1["full_upper95"].map(num),
        "判定": t1["verdict"],
    })
    rows.index.name = "代號"
    desc = pd.DataFrame({
        "策略": t1["strategy"],
        "CAGR": t1["full_CAGR"].map(pct),
        "Sharpe": t1["full_Sharpe"].map(num),
        "最大回撤": t1["full_MaxDD"].map(pct),
        "曝險": t1["full_Exposure"].map(lambda v: f"{v:.0%}"),
        "來回/年": t1["full_RoundTripsPerYear"].map(lambda v: f"{v:.1f}"),
        "交易數": t1["n_trades"],
        "backtrader CAGR": t1["bt_CAGR"].map(pct),
    })
    desc.index.name = "代號"
    bh = d["bh"]
    names = [f"{c} {s}" for c, s in zip(t1.index, t1["strategy"])]
    effect_plot(names, t1["full_delta_sharpe"].to_numpy(),
                (t1["full_delta_sharpe"] - 1.645 * t1["full_sd0"]).to_numpy(),
                t1["full_upper95"].to_numpy(), config.T1_MATERIALITY, t1["verdict"].tolist(),
                "ΔSharpe（規則 - 隨機擇時中位數）；線段 = ±1.645×SD0，右端是判定用的 95% 上界",
                FIG.joinpath("t1_delta_sharpe.png"))
    curves = d["curves"]
    small_multiples(curves.drop(columns=["買進持有 0050"]), curves["買進持有 0050"],
                    dict(zip(t1.index, names)), FIG.joinpath("t1_equity.png"), "買進持有 0050（含息）")
    counts = t1["verdict"].value_counts().to_dict()
    return f"""## A-T1：0050 擇時

評估期 {start.date()} ～ {end.date()}（0050 證交所資料 {bh['first_bar']} 起，前 {config.WARMUP_DAYS} 根暖身）。
資料中超過漲跌幅的跳動：{bh['bad_days']} 天。

**判定：{', '.join(f'{k} × {v}' for k, v in sorted(counts.items()))}。**

買進持有 0050（含息、扣一次買進成本）：CAGR {pct(bh['CAGR'])}、Sharpe {num(bh['Sharpe'])}、最大回撤 {pct(bh['MaxDD'])}。

![T1 ΔSharpe](figures/t1_delta_sharpe.png)

{md_table(rows)}

描述統計（開盤到開盤、扣成本；最後一欄是 backtrader 淨值算的 CAGR，用來交叉核對報酬口徑）：

{md_table(desc)}

![T1 淨值](figures/t1_equity.png)
"""


def section_t2(d) -> str:
    t2 = d["t2"]
    rows = pd.DataFrame({
        "策略": t2["strategy"],
        "交易數": t2["full_n_trades"],
        "剔除（跳動）": t2["excluded_bad_day"],
        "平均持有天數": t2["full_mean_hold_days"].map(lambda v: f"{v:.1f}"),
        "每筆淨報酬": t2["full_mean_net"].map(pct),
        "E（月平均超額）": t2["full_E"].map(pct),
        "t": t2["full_t"].map(lambda v: f"{v:.2f}"),
        "Holm p": t2["holm_p"].map(num),
        "軌 A E": t2["A_E"].map(pct),
        "軌 B E": t2["B_E"].map(pct),
        "正年度": t2["full_positive_year_share"].map(lambda v: f"{v:.0%}"),
        "集中度": t2["concentration_ok"].map(lambda v: "通過" if v else "未通過"),
        "上界": t2["full_upper95"].map(pct),
        "判定": t2["verdict"],
    })
    rows.index.name = "代號"
    names = [f"{c} {s}" for c, s in zip(t2.index, t2["strategy"])]
    effect_plot(names, t2["full_E"].to_numpy() * 100,
                (t2["full_E"] - 1.645 * t2["full_se"]).to_numpy() * 100,
                t2["full_upper95"].to_numpy() * 100, config.T2_MATERIALITY * 100, t2["verdict"].tolist(),
                "每筆超額報酬 E（%，規則 - 同股隨機進場）；線段 = ±1.645×SE，右端是 95% 上界",
                FIG.joinpath("t2_excess.png"), fmt=lambda v: f"{v:.2f}%")
    port = d["portfolio"]
    port = port.loc[:config.DATA_END]
    lr = np.log1p(port.fillna(0.0))
    titles = dict(zip(t2.index, names))
    small_multiples(lr[[c for c in t2.index if c in lr]], lr["bh"], titles,
                    FIG.joinpath("t2_portfolio.png"), "股票池等權買進持有")
    desc = {}
    for c in ["bh"] + list(t2.index):
        if c in lr:
            desc[c] = analysis.summarize(lr[c].loc["2006-01-01":])
    desc = pd.DataFrame(desc).T
    desc_tab = pd.DataFrame({"CAGR": desc["CAGR"].map(pct), "Sharpe": desc["Sharpe"].map(num),
                             "最大回撤": desc["MaxDD"].map(pct)})
    desc_tab.index = ["股票池等權" if i == "bh" else f"{i} {STRATEGY_BY_CODE[i].__name__}" for i in desc_tab.index]
    desc_tab.index.name = "組合"
    counts = t2["verdict"].value_counts().to_dict()
    errors = d.get("errors", [])
    err_text = (f"執行失敗的（股票, 策略）組合：{len(errors)} 個"
                + ("：" + "、".join(f"{s} {c}" for s, c, _ in errors[:20]) if errors else "。"))
    return f"""## A-T2：個股波段（每季流動性前 {config.T2_TOP_N} 名）

**判定：{', '.join(f'{k} × {v}' for k, v in sorted(counts.items()))}。**

{err_text} 持有期間碰到不可能跳動而剔除的交易數列在「剔除（跳動）」欄。

![T2 超額](figures/t2_excess.png)

{md_table(rows)}

描述（不判定）：把每檔股票在成分期間的策略日報酬等權平均成一個組合（空手部位報酬為 0），2006 年起：

{md_table(desc_tab)}

![T2 組合](figures/t2_portfolio.png)
"""


# ---------------------------------------------------------------- B
def section_b1(d) -> str:
    j = d["judged"]
    cand = j[j["candidate"]].copy()
    cand = cand.reindex(cand["test_signed_t"].sort_values(ascending=False).index)
    tab = pd.DataFrame({
        "方向": cand["sign"].map(lambda s: "+" if s > 0 else "−"),
        "探索 IC": cand["disc_ic_w"].map(lambda v: f"{v:+.3f}"),
        "探索 t": cand["disc_t"].map(lambda v: f"{v:+.1f}"),
        "測試 IC": cand["test_ic_w"].map(lambda v: f"{v:+.3f}"),
        "測試 t（方向化）": cand["test_signed_t"].map(lambda v: f"{v:+.2f}"),
        "Holm p": cand["holm_p"].map(num),
        "狀態": cand["status"],
    })
    tab.index.name = "因子"
    counts = j.loc[j["candidate"], "status"].value_counts().to_dict()

    # 圖：探索期 vs 測試期的方向化加權 IC，每個候選一點
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    status_color = {"通過": SERIES[0], "未通過": SERIES[1], "方向翻轉": SERIES[2]}
    for status, color in status_color.items():
        sub = cand[cand["status"] == status]
        ax.scatter((sub["sign"] * sub["disc_ic_w"]), (sub["sign"] * sub["test_ic_w"]), s=36, color=color,
                   edgecolor=SURFACE, linewidth=1.5, label=f"{status}（{len(sub)}）", zorder=3)
    lim = max(0.01, float(np.nanmax(np.abs(cand["disc_ic_w"]))) * 1.1,
              float(np.nanmax(np.abs(cand["test_ic_w"]))) * 1.1)
    ax.plot([0, lim], [0, lim], color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
    ax.axhline(config.B1_MATERIALITY, color=INK2, linewidth=1, linestyle=(0, (2, 2)))
    ax.text(lim, config.B1_MATERIALITY, f"門檻 {config.B1_MATERIALITY}", color=INK2, fontsize=8,
            ha="right", va="bottom")
    ax.text(lim, lim, "兩期相同", color=MUTED, fontsize=8, ha="right", va="top")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("探索期 2008–2020：方向化加權 IC")
    ax.set_ylabel("測試期 2021–2026/09：方向化加權 IC")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG.joinpath("b1_ic_scatter.png"), dpi=150)
    plt.close(fig)

    q_text = ""
    if d.get("quantiles"):
        q = pd.DataFrame(d["quantiles"]).T * 1e4
        q.columns = [f"Q{int(c)}" for c in q.columns]
        q["Q5−Q1"] = q.iloc[:, -1] - q.iloc[:, 0]
        q = q.round(2)
        q["平均數與 IC 同向"] = np.where(q["Q5−Q1"] > 0, "是", "**否**")
        q_text = ("通過因子在測試期的五分位平均日報酬（相對當日平均，bp；alphalens `mean_return_by_quantile`，"
                  "已依方向化，Q5 為 IC 預期最好的一組）。rank IC 量的是名次（接近中位數），等權持有賺的是平均數；"
                  "兩者方向不同時，表示右尾（漲停）集中在 IC 預期較差的那組（cross-sectional-selection §1）：\n\n"
                  + md_table(q))
    return f"""## B1：Alpha158 單因子篩選

158 個因子中探索期 |t(z)| ≥ {config.B1_DISCOVERY_T} 的候選 {int(j['candidate'].sum())} 個；
測試期結果：{', '.join(f'{k} {v}' for k, v in counts.items())}。
label 超過漲跌幅的不可能報酬設為缺值：{d.get('bad_labels', 0)} 筆（全部 {d.get('n_labels', 0)} 筆）。

![B1 IC](figures/b1_ic_scatter.png)

{md_table(tab)}

{q_text}
"""


def section_b2(d) -> str:
    t = d["table"]
    tab = pd.DataFrame({
        "測試期 IC（加權）": t["ic_ic_w"].map(lambda v: f"{v:+.4f}"),
        "ICIR": t["ic_icir"].map(num),
        "IC t": t["ic_t"].map(lambda v: f"{v:.2f}"),
        "IC Holm p": t["ic_holm_p"].map(num),
        "年化毛超額": t["ann_gross_excess_vs_ew"].map(pct),
        "年化成本": t["ann_cost"].map(pct),
        "年化淨超額": t["ann_excess"].map(pct),
        "t": t["t"].map(lambda v: f"{v:.2f}"),
        "前半": t["half1"].map(pct),
        "後半": t["half2"].map(pct),
        "上界": t["upper95"].map(pct),
        "判定": t["verdict"],
    })
    tab.index = tab.index.rename("模型")
    sens = pd.DataFrame({
        "主設定": t["main_ann_excess_vs_ew"].map(pct),
        "手續費 0": t["fee0_ann_excess_vs_ew"].map(pct),
        "手續費不打折": t["fee_full_ann_excess_vs_ew"].map(pct),
        "n_drop=1": t["ndrop1_ann_excess_vs_ew"].map(pct),
        "n_drop=2": t["ndrop2_ann_excess_vs_ew"].map(pct),
        "每日換手": t["daily_turnover"].map(lambda v: f"{v:.1%}"),
    })
    sens.index = sens.index.rename("模型（年化淨超額 vs 等權）")
    diag = ""
    if "ic_diff_vs_linear" in t:
        dd = t.dropna(subset=["ic_diff_vs_linear"])
        diag = "\n".join(f"- {m}：每日 IC 差 {r.ic_diff_vs_linear:+.4f}（NW t {r.ic_diff_vs_linear_t:.2f}）"
                         for m, r in dd.iterrows())

    ew = d["ew"]
    series = {}
    decomp = {}
    first = next(iter(d["backtests"].values()))["main"]
    bench = first["bench"]
    decomp["0050（含息）"] = analysis.summarize(np.log1p(bench))
    decomp["top200 等權"] = analysis.summarize(np.log1p(ew.reindex(bench.index).fillna(0)))
    for name, bts in d["backtests"].items():
        b = bts["main"]
        ex = (b["net"] - ew.reindex(b.index)).fillna(0)
        series[name] = ex.cumsum()
        decomp[f"{name}（top50，淨）"] = analysis.summarize(np.log1p(b["net"]))
    line_chart(series, FIG.joinpath("b2_excess.png"), "累積淨超額報酬（相對 top200 等權，算術加總）")
    dec = pd.DataFrame(decomp).T
    dec_tab = pd.DataFrame({"CAGR": dec["CAGR"].map(pct), "Sharpe": dec["Sharpe"].map(num),
                            "最大回撤": dec["MaxDD"].map(pct)})
    dec_tab.index.name = "組合（測試期）"
    counts = t["verdict"].value_counts().to_dict()
    return f"""## B2：qlib 多因子模型（TopkDropout topk {config.B2_TOPK}、n_drop {config.B2_N_DROP}）

**判定：{', '.join(f'{k} × {v}' for k, v in sorted(counts.items()))}。**

{md_table(tab)}

![B2 超額](figures/b2_excess.png)

分解（cross-sectional-selection §0）：

{md_table(dec_tab)}

對線性基準 M1 的每日 rank IC 差（不改判定）：

{diag}

敏感度（只報告，不改判定）：

{md_table(sens)}
"""


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    parts = ["# 台股：Qbot 策略回測與因子挖掘\n",
             "由 `qbot/tw_research/report.py` 產生；數字以程式輸出為準。方法、移植差異與解讀見 "
             "`docs/03-智能策略/台股策略回測與因子挖掘.md`。\n",
             "<details><summary><b>預先承諾（原文，寫於任何結果計算之前）</b></summary>\n\n"
             + config.PRE_REGISTRATION + "\n</details>\n"]
    t1 = load(OUT.joinpath("strategies", "t1.pkl"))
    if t1:
        parts.append(section_t1(t1))
    t2 = load(OUT.joinpath("strategies", "t2.pkl"))
    if t2:
        parts.append(section_t2(t2))
    b1 = load(OUT.joinpath("factors", "b1.pkl"))
    if b1:
        parts.append(section_b1(b1))
    b2 = load(OUT.joinpath("factors", "b2.pkl"))
    if b2:
        parts.append(section_b2(b2))
    not_ported = pd.DataFrame({"原因": NOT_PORTED})
    not_ported.index.name = "沒有移植的檔案"
    parts.append("## 沒有移植的策略\n\n" + md_table(not_ported) + "\n")
    OUT.joinpath("REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT.joinpath('REPORT.md')}")


if __name__ == "__main__":
    main()
