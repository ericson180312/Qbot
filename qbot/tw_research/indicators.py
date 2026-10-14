"""pandas 版指標：給原本用 pandas 寫成、或在 backtrader 逐根計算太慢的策略。

每個函式只用 t 日（含）以前的資料，tests/test_tw_research.py 以「截斷資料後前段結果不變」檢查。
"""

from collections import deque

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def rsi_qbot(close: pd.Series, period: int = 14) -> pd.Series:
    """pytrader/strategies/StochRSI.py 與 rsi_departure_strategy.py 的 calcRSI（Wilder 平滑）。

    第一列的漲跌記為 0、第 period 列用前 period 列的簡單平均起算，之後 Wilder 遞迴；與原始碼逐列相同。
    """
    diff = close.diff()
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    n = len(close)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    if n >= period:
        avg_gain[period - 1] = gain[:period].mean()
        avg_loss[period - 1] = loss[:period].mean()
        for i in range(period, n):
            avg_gain[i] = ((period - 1) * avg_gain[i - 1] + gain[i]) / period
            avg_loss[i] = ((period - 1) * avg_loss[i - 1] + loss[i]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    rsi[: period - 1] = np.nan
    return pd.Series(rsi, index=close.index)


def stoch_rsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> pd.Series:
    """StochRSI.py 的 calcStochOscillator：RSI 在過去 stoch_period 日區間內的位置（0–100）。"""
    rsi = rsi_qbot(close, rsi_period)
    low = rsi.rolling(stoch_period).min()
    high = rsi.rolling(stoch_period).max()
    return 100 * (rsi - low) / (high - low)


def stoch_rsi_reversion_target(close: pd.Series, rsi_period=14, stoch_period=14,
                               short_level=80, buy_level=20) -> pd.Series:
    """StochRSIReversionStrategy(shorts=False)：低於 buy_level 持有，高於 short_level 空手，其餘延續。"""
    srsi = stoch_rsi(close, rsi_period, stoch_period)
    pos = pd.Series(np.nan, index=close.index)
    pos[srsi < buy_level] = 1.0
    pos[srsi > short_level] = 0.0
    return pos.ffill().fillna(0.0)


def ssa_last(close: pd.Series, window: int = 15) -> pd.Series:
    """qbot/strategies/ssa_strategy_bt.py 的 ssa_index_ind，向量化版本。

    原始碼每根 K 棒取最近 2*window 個收盤價，嵌入成 (window+1) x window 的軌跡矩陣，
    只留最大奇異值做一階重建，再取重建序列的最後一點。對角平均到最後一點時只有
    一個元素（矩陣右下角），所以值等於 s1 * U[-1, 0] * V[0, -1]。
    tests 用原始碼的 recreate_array 逐點比對。
    """
    t = 2 * window
    m = window
    values = close.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < t:
        return pd.Series(out, index=close.index)
    windows = np.lib.stride_tricks.sliding_window_view(values, t)  # (N, t)
    n = t - m + 1
    traj = np.lib.stride_tricks.sliding_window_view(windows, m, axis=1)[:, :n, :]  # (N, n, m)
    u, s, vt = np.linalg.svd(traj, full_matrices=False)
    out[t - 1:] = s[:, 0] * u[:, n - 1, 0] * vt[:, 0, m - 1]
    return pd.Series(out, index=close.index)


# ------------------------------------------------------------- RSI 背離（rsi_departure_strategy.py）
def _consecutive(idx: np.ndarray, vals: np.ndarray, K: int, keep_if) -> list:
    """原始碼 getHigherHighs / getLowerLows… 的共同骨架：連續 K 個極值都滿足 keep_if 才記一組。"""
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, ix in enumerate(idx):
        if i == 0:
            ex_deque.append(ix)
            continue
        if not keep_if(vals[i], vals[i - 1]):
            ex_deque.clear()
        ex_deque.append(ix)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())
    return extrema


def _confirm(extrema: list, order: int, n: int) -> np.ndarray:
    """確認日 = 這組最後一個極值 + order（argrelextrema 要再等 order 根才知道它是極值）。"""
    idx = np.array([e[-1] + order for e in extrema], dtype=int)
    return idx[idx < n]


def peak_states(values: np.ndarray, order: int = 5, K: int = 2):
    """回傳 (highs, lows)：最近一次確認的型態，highs 1=更高的高點、-1=更低的高點；
    lows 1=更低的低點、-1=更高的低點；尚無確認時為 0。

    原始碼 getPeaks 把 lows 欄誤寫成 highs 欄的 ffill（lows 欄從未被使用），這裡改正，
    見 docs/03-智能策略/台股策略回測與因子挖掘.md 的移植差異表。
    """
    n = len(values)
    vals = np.asarray(values, dtype=float)
    # 和 NaN 比較一律為 False，所以暖身期的 NaN 旁邊不會被當成極值，與原始碼相同
    hi = argrelextrema(vals, np.greater, order=order)[0]
    lo = argrelextrema(vals, np.less, order=order)[0]
    hh = _confirm(_consecutive(hi, vals[hi], K, lambda cur, prev: cur >= prev), order, n)
    lh = _confirm(_consecutive(hi, vals[hi], K, lambda cur, prev: cur <= prev), order, n)
    ll = _confirm(_consecutive(lo, vals[lo], K, lambda cur, prev: cur <= prev), order, n)
    hl = _confirm(_consecutive(lo, vals[lo], K, lambda cur, prev: cur >= prev), order, n)

    highs = np.full(n, np.nan)
    highs[hh] = 1
    highs[lh] = -1
    lows = np.full(n, np.nan)
    lows[ll] = 1
    lows[hl] = -1
    highs = pd.Series(highs).ffill().fillna(0).to_numpy()
    lows = pd.Series(lows).ffill().fillna(0).to_numpy()
    return highs, lows


def rsi_divergence_target(close: pd.Series, period: int = 14, order: int = 5, K: int = 2) -> pd.Series:
    """RSIDivergenceStrategy 的做多部分（台股散戶融券受限，去掉放空）。

    進場：價格的低點型態為 -1（更高的低點）且 RSI 的低點型態為 1（更低的低點），且 RSI < 50；
    持有：RSI < 50 且 RSI < 進場時的 RSI，否則出場。條件照原始碼（原始碼的說明文字與條件方向相反）。
    """
    rsi = rsi_qbot(close, period)
    _, close_lows = peak_states(close.to_numpy(dtype=float), order, K)
    rsi_vals = rsi.to_numpy()
    _, rsi_lows = peak_states(rsi_vals, order, K)
    pos = np.zeros(len(close))
    entry_rsi = np.nan
    for i in range(len(close)):
        r = rsi_vals[i]
        if np.isnan(r):
            continue
        prev = pos[i - 1] if i > 0 else 0
        if prev == 0:
            if close_lows[i] == -1 and rsi_lows[i] == 1 and r < 50:
                pos[i] = 1
                entry_rsi = r
        elif prev == 1:
            if r < 50 and r < entry_rsi:
                pos[i] = 1
    return pd.Series(pos, index=close.index)
