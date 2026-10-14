"""事後探索（不參與判定；2021–2026/09，已看過的測試期）：台指期避險後，各模型 × 換手 × 成本情境的年化報酬與逐年正負。

回答「降低成本、改用台指期避險後，有沒有哪個設定每年都贏大盤」。
避險比率用全期迴歸 beta（樣本內，偏樂觀）；個股期貨情境 = 毛報酬 − 換手 × 每邊成本（未計轉倉與標的限制）。

執行：python qbot/tw_research/explore_q3.py
輸出：outputs/tw_research/explore_b2/q3_tx_hedged.csv
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_research import analysis, config, taifex  # noqa: E402

SCENARIO_COSTS = {"散戶股票成本": None, "個股期貨 0.05%": 0.0005, "個股期貨 0.15%": 0.0015, "零成本": 0.0}


def main():
    tx = taifex.load_tx_returns(range(2020, 2027))
    rows = []
    for name in ["M1_Linear_A158", "M2_LGB_A158", "M3_XGB_A158", "M4_LGB_A360"]:
        with open(config.OUTPUT_DIR.joinpath("factors", f"b2_{name}.pkl"), "rb") as f:
            r = pickle.load(f)
        for tag, nd in [("ndrop1", 1), ("ndrop2", 2), ("main", 5)]:
            b = r["backtests"][tag]
            x = tx.reindex(b.index)
            ok = x.notna()
            beta = float(np.cov(b["gross"][ok], x[ok], ddof=1)[0, 1] / x[ok].var())
            for sname, c in SCENARIO_COSTS.items():
                y = b["net"] if c is None else b["gross"] - b["turnover"] * c
                h = (y - beta * x - beta * config.D_HEDGE_COST_ANNUAL / analysis.TRADING_DAYS).dropna()
                mu, _, t = analysis.newey_west_t(h.to_numpy(), config.NW_LAG_DAILY)
                yearly = h.groupby(h.index.year).mean() * analysis.TRADING_DAYS
                rows.append({"model": name, "n_drop": nd, "scenario": sname, "beta_tx": beta,
                             "E": mu * analysis.TRADING_DAYS, "t": t, "pos_years": int((yearly > 0).sum()),
                             "n_years": len(yearly), "worst_year": float(yearly.min()),
                             "yearly": " ".join(f"{yr % 100:02d}:{v:+.0%}" for yr, v in yearly.items())})
    df = pd.DataFrame(rows)
    out = config.OUTPUT_DIR.joinpath("explore_b2", "q3_tx_hedged.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"wrote {out}; cells with every year positive: {int((df['pos_years'] == df['n_years']).sum())}")


if __name__ == "__main__":
    main()
