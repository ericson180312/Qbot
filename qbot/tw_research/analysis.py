"""績效、隨機對照、Newey-West、Holm 與預先承諾的機械式判定。

報酬口徑（檢定一律用這個，backtrader 淨值只做描述）：
    h_d = 1 表示從 d 日開盤持有到 d+1 日開盤（等於 d 日收盤時持有）。
    日對數報酬 = h_d * log(open_{d+1} / open_d) + 進場日 log(1 - 買進成本) + 出場日 log(1 - 賣出成本)
規則與它的隨機對照用同一個式子算，成本相同，差別只剩進出場時機。
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import config

TRADING_DAYS = 252


# ---------------------------------------------------------------- 基本績效
def open_to_open_log_returns(bars: pd.DataFrame) -> pd.Series:
    """d 日開盤到下一個交易日開盤的對數報酬；最後一天為 NaN。"""
    o = bars["open"]
    return np.log(o.shift(-1) / o)


def position_log_returns(held: pd.Series, oo: pd.Series, buy_cost: float, sell_cost: float) -> pd.Series:
    h = held.reindex(oo.index).fillna(False).astype(float)
    prev = h.shift(1).fillna(0.0)
    entries = ((h == 1) & (prev == 0)).astype(float)
    exits = ((h == 0) & (prev == 1)).astype(float)
    lr = h * oo.fillna(0.0) + entries * np.log1p(-buy_cost) + exits * np.log1p(-sell_cost)
    return lr


def sharpe(log_returns: pd.Series) -> float:
    r = np.expm1(np.asarray(log_returns, dtype=float))
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def max_drawdown(log_returns: pd.Series) -> float:
    curve = np.exp(np.cumsum(np.asarray(log_returns, dtype=float)))
    peak = np.maximum.accumulate(np.concatenate([[1.0], curve]))[1:]
    return float((curve / peak - 1).min()) if len(curve) else 0.0


def summarize(log_returns: pd.Series, held: Optional[pd.Series] = None) -> Dict[str, float]:
    lr = log_returns.dropna()
    years = len(lr) / TRADING_DAYS
    out = {
        "CAGR": float(np.exp(lr.sum() / years) - 1) if years > 0 else np.nan,
        "Sharpe": sharpe(lr),
        "MaxDD": max_drawdown(lr),
        "Vol": float(np.expm1(lr).std(ddof=1) * np.sqrt(TRADING_DAYS)),
    }
    if held is not None:
        h = held.reindex(lr.index).fillna(False).astype(int)
        out["Exposure"] = float(h.mean())
        out["RoundTripsPerYear"] = float(((h.diff() == 1).sum()) / years) if years > 0 else np.nan
    return out


# ---------------------------------------------------------------- 統計工具
def newey_west_t(x: Sequence[float], lag: int) -> Tuple[float, float, float]:
    """回傳 (平均, NW 標準誤, t)。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float(np.mean(x)) if n else np.nan, np.nan, np.nan
    mu = x.mean()
    e = x - mu
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    se = np.sqrt(max(s, 0.0) / n)
    return float(mu), float(se), float(mu / se) if se > 0 else np.nan


def one_sided_p(t: float, n: int) -> float:
    if not np.isfinite(t):
        return np.nan
    return float(stats.t.sf(t, df=max(n - 1, 1)))


def holm(pvalues: Dict[str, float], alpha: float = config.FWER) -> Dict[str, Tuple[float, bool]]:
    """Holm 逐步校正。回傳 {名稱: (校正後 p, 是否拒絕)}；NaN 視為 1。"""
    items = sorted(pvalues.items(), key=lambda kv: (np.nan_to_num(kv[1], nan=1.0)))
    m = len(items)
    out = {}
    running = 0.0
    still_rejecting = True
    for i, (name, p) in enumerate(items):
        p = 1.0 if not np.isfinite(p) else p
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        reject = still_rejecting and p <= alpha / (m - i)
        still_rejecting = reject
        out[name] = (adj, reject)
    return out


# ---------------------------------------------------------------- T1：重排持有段
def runs(h: np.ndarray):
    """把 0/1 序列切成 (值, 長度) 的段落。"""
    if len(h) == 0:
        return []
    change = np.flatnonzero(np.diff(h)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(h)]])
    return [(int(h[s]), int(e - s)) for s, e in zip(starts, ends)]


def shuffle_segments(h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """持有段與空手段各自隨機重排，保留交易次數、曝險與持有天數分布，只拿掉時機。"""
    segs = runs(h)
    if len(segs) <= 1:
        return h.copy()
    first = segs[0][0]
    ones = [n for v, n in segs if v == 1]
    zeros = [n for v, n in segs if v == 0]
    rng.shuffle(ones)
    rng.shuffle(zeros)
    out = np.empty(len(h), dtype=np.int8)
    pos = 0
    oi = zi = 0
    v = first
    for _ in range(len(segs)):
        if v == 1:
            n = ones[oi]
            oi += 1
        else:
            n = zeros[zi]
            zi += 1
        out[pos:pos + n] = v
        pos += n
        v = 1 - v
    return out


def t1_test(held: pd.Series, bars: pd.DataFrame, start: str, end: str, buy_cost: float, sell_cost: float,
            draws: int = config.T1_NULL_DRAWS, seed: int = config.NULL_SEED) -> Dict[str, object]:
    """單一序列（0050）的擇時檢定：規則 vs 重排持有段的隨機擇時。"""
    oo = open_to_open_log_returns(bars).loc[start:end].dropna()
    h = held.reindex(oo.index).fillna(False).astype(np.int8).to_numpy()
    rule_lr = position_log_returns(pd.Series(h.astype(bool), index=oo.index), oo, buy_cost, sell_cost)
    rule_sharpe = sharpe(rule_lr)

    rng = np.random.default_rng(seed)
    years = oo.index.year.to_numpy()
    uniq_years = np.unique(years)
    null_sharpes = np.empty(draws)
    null_year = np.empty((draws, len(uniq_years)))
    oo_v = oo.to_numpy()
    lb, ls = np.log1p(-buy_cost), np.log1p(-sell_cost)
    for k in range(draws):
        hs = shuffle_segments(h, rng).astype(float)
        prev = np.concatenate([[0.0], hs[:-1]])
        lr = hs * oo_v + (hs > prev) * lb + (hs < prev) * ls
        null_sharpes[k] = sharpe(lr)
        null_year[k] = [lr[years == y].sum() for y in uniq_years]

    null_median = float(np.median(null_sharpes))
    sd0 = float(null_sharpes.std(ddof=1))
    rule_year = np.array([rule_lr.to_numpy()[years == y].sum() for y in uniq_years])
    year_excess = rule_year - np.median(null_year, axis=0)
    traded_years = np.array([h[years == y].any() for y in uniq_years])
    return {
        "rule_sharpe": rule_sharpe,
        "null_median": null_median,
        "delta_sharpe": rule_sharpe - null_median,
        "sd0": sd0,
        "mde": 2 * sd0,
        "p": float((1 + (null_sharpes >= rule_sharpe).sum()) / (draws + 1)),
        "upper95": rule_sharpe - null_median + 1.645 * sd0,
        "positive_year_share": float((year_excess[traded_years] > 0).mean()) if traded_years.any() else np.nan,
        "year_excess": pd.Series(year_excess, index=uniq_years),
        "rule_log_returns": rule_lr,
        "exposure": float(h.mean()),
    }


def trade_net_log_returns(trades: pd.DataFrame, bars: pd.DataFrame, buy_cost: float, sell_cost: float) -> np.ndarray:
    o = bars["open"].to_numpy()
    closed = trades.dropna(subset=["exit_date"])
    return (np.log(o[closed["exit_i"].to_numpy()] / o[closed["entry_i"].to_numpy()])
            + np.log1p(-buy_cost) + np.log1p(-sell_cost))


# ---------------------------------------------------------------- T2：同股隨機進場日
def t2_trade_null(trades: pd.DataFrame, bars: pd.DataFrame, buy_cost: float, sell_cost: float,
                  rng: np.random.Generator, draws: int = config.T2_NULL_DRAWS,
                  radius: int = config.T2_NULL_RADIUS) -> pd.DataFrame:
    """每筆已平倉交易：實際淨報酬，以及同一檔股票、同樣持有天數、前後 radius 日內隨機進場的平均淨報酬。

    隨機進場日排除開盤漲停（實際交易在那種日子買不到）。報酬為簡單報酬。
    持有期間 [進場日, 出場日] 碰到無法解釋的跳動（data.bad_days）的交易標記 spans_bad_day 並剔除，
    隨機對照也只從不碰到跳動的進場日抽。
    """
    closed = trades.dropna(subset=["exit_date"]).copy()
    if closed.empty:
        return closed.assign(net=[], null_net=[], excess=[], spans_bad_day=[])
    o = bars["open"].to_numpy()
    n = len(o)
    from .data import bad_days, open_limit_flags

    blocked = open_limit_flags(bars)["limit_up_open"].to_numpy()
    bad_cum = np.concatenate([[0], np.cumsum(bad_days(bars).to_numpy().astype(int))])

    def spans_bad(start, stop):  # [start, stop] 內有沒有跳動
        return bad_cum[stop + 1] - bad_cum[start] > 0

    cost = np.log1p(-buy_cost) + np.log1p(-sell_cost)
    e = closed["entry_i"].to_numpy()
    x = closed["exit_i"].to_numpy()
    hold = x - e
    actual = np.expm1(np.log(o[x] / o[e]) + cost)
    bad = spans_bad(e, x)
    null_mean = np.full(len(e), np.nan)
    for j in range(len(e)):
        if bad[j]:
            continue
        lo = max(0, e[j] - radius)
        hi = min(n - 1 - hold[j], e[j] + radius)
        if hi < lo:
            continue
        cand = np.arange(lo, hi + 1)
        cand = cand[~blocked[cand] & ~spans_bad(cand, cand + hold[j])]
        if len(cand) == 0:
            continue
        pick = rng.choice(cand, size=draws, replace=True)
        null_mean[j] = np.expm1(np.log(o[pick + hold[j]] / o[pick]) + cost).mean()
    closed["net"] = actual
    closed["null_net"] = null_mean
    closed["excess"] = actual - null_mean
    closed["spans_bad_day"] = bad
    return closed


def t2_test(trade_table: pd.DataFrame, start: str, end: str, lag: int = config.T2_NW_LAG) -> Dict[str, object]:
    """以進場月份取當月交易超額的平均，形成月序列做 NW t 檢定。E = 月序列平均。"""
    t = trade_table.dropna(subset=["excess"])
    t = t[(t["entry_date"] >= pd.Timestamp(start)) & (t["entry_date"] <= pd.Timestamp(end))]
    if t.empty:
        return {"n_trades": 0, "E": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "upper95": np.nan}
    monthly = t.groupby(t["entry_date"].dt.to_period("M"))["excess"].mean()
    mu, se, tstat = newey_west_t(monthly.to_numpy(), lag)
    yearly = t.groupby(t["entry_date"].dt.year)["excess"].mean()
    return {
        "n_trades": int(len(t)),
        "n_months": int(len(monthly)),
        "E": mu,
        "se": se,
        "t": tstat,
        "p": one_sided_p(tstat, len(monthly)),
        "upper95": mu + 1.645 * se if np.isfinite(se) else np.nan,
        "positive_year_share": float((yearly > 0).mean()),
        "pooled_mean_excess": float(t["excess"].mean()),
        "mean_net": float(t["net"].mean()),
        "mean_hold_days": float((t["exit_i"] - t["entry_i"]).mean()),
    }


def drop_top(trade_table: pd.DataFrame, share: float) -> pd.DataFrame:
    t = trade_table.dropna(subset=["excess"])
    k = int(np.ceil(len(t) * share))
    if k == 0:
        return t
    return t.drop(t["net"].nlargest(k).index)


# ---------------------------------------------------------------- 判定
def verdict(holm_reject: bool, E: float, upper95: float, track_a: float, track_b: float,
            positive_year_share: float, concentration_ok: bool, materiality: float) -> str:
    """預先承諾的三分類：(+) 支持、(−) 無效果、(?) 檢定力不足、無法回答。"""
    if (holm_reject and np.isfinite(E) and E >= materiality and track_a > 0 and track_b > 0
            and positive_year_share >= config.POSITIVE_YEAR_SHARE and concentration_ok):
        return "(+)"
    if np.isfinite(upper95) and upper95 < materiality:
        return "(−)"
    return "(?)"
