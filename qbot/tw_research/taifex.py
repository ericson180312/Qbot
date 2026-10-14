"""台指期（TX）近月連續日報酬，用來代表賣台指期避險的損益。

資料來源：`python_台指最佳槓桿/data/fut_TX_YYYY.parquet`（FinMind TaiwanFuturesDaily，逐日逐合約），
第一次使用時複製到 BASE_DIR/taifex，之後不再依賴那個專案。

只用一般交易時段（trading_session == "position"）、6 碼月合約（排除價差合約）、收盤價 > 0。
轉倉：計算 t 日報酬（t−1 收盤 → t 收盤）時，持有「最後交易日在 t 之後」的最近月合約，
所以到期日當天已經換到次月，不會用到到期日的收盤價（那不是最終結算價）。
報酬是期貨價格變動，已內含正逆價差與預期股利；空單的損益就是 −報酬。
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

TX_SOURCE_DIR = Path(r"C:\Users\User\Desktop\python_台指最佳槓桿\data")
CACHE_DIR = config.BASE_DIR.joinpath("taifex")


def load_tx_daily(years=range(2011, 2022)) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for y in years:
        cached = CACHE_DIR.joinpath(f"fut_TX_{y}.parquet")
        if not cached.exists():
            shutil.copy(TX_SOURCE_DIR.joinpath(f"fut_TX_{y}.parquet"), cached)
        frames.append(pd.read_parquet(cached))
    df = pd.concat(frames, ignore_index=True)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["contract_date"] = df["contract_date"].astype(str).str.strip()
    df = df[(df["trading_session"] == "position") & df["contract_date"].str.fullmatch(r"\d{6}")]
    df = df[df["close"] > 0]
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "contract_date", "close", "volume"]].drop_duplicates(["date", "contract_date"])


def front_returns(df: pd.DataFrame) -> pd.Series:
    """近月連續日報酬（見模組說明的轉倉規則）。"""
    close = df.pivot(index="date", columns="contract_date", values="close").sort_index()
    close = close[sorted(close.columns)]
    last_day = close.apply(lambda s: s.last_valid_index())
    dates = close.index
    out = pd.Series(np.nan, index=dates)
    for i in range(1, len(dates)):
        t, prev = dates[i], dates[i - 1]
        for c in close.columns:  # 依到期月份由近到遠
            # 最後交易日在資料結束前的才是真的到期；資料最後一天還在交易的合約不算
            if last_day[c] <= t and last_day[c] < dates[-1]:
                continue
            p0, p1 = close.at[prev, c], close.at[t, c]
            if np.isfinite(p0) and np.isfinite(p1):
                out.iloc[i] = p1 / p0 - 1
                break
    return out.dropna()


def load_tx_returns(years=range(2011, 2022)) -> pd.Series:
    return front_returns(load_tx_daily(years)).rename("tx")
