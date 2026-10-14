"""證交所官方 API：0050 的原始日線與除權息參考價，自己建含息還原價。

為什麼不用 Yahoo：Yahoo 的 0050 在 2014-01-02 有一個 −75% 的跳動（2025 年 1 拆 4 的分割只調整到
2014 年以後）、2013 年以前沒有任何配息紀錄、2014 年的配息用未分割的單位（算成 9.5%），
2009 年第一季還有 ±12% 的日報酬（當時漲跌幅上限 7%）。T1 只有這一條序列，所以改用官方資料。

    STOCK_DAY  每月一次請求，官方只提供 2010-01-04 以後
    TWT49U     除權息計算結果表：除權息前收盤價、除權息參考價
還原因子 = 參考價 / 前收盤價（除權息日當天生效）；股票分割用 SPLITS 手動列出並以價格比驗證。
"""

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

from . import config

STOCK_DAY_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
EX_RIGHT_URL = "https://www.twse.com.tw/rwd/zh/exRight/TWT49U"
HEADERS = {"User-Agent": "Mozilla/5.0"}
FIRST_MONTH = "2010-01"  # STOCK_DAY 不提供更早的資料
CACHE_DIR = config.BASE_DIR.joinpath("twse")

# 股票分割：(恢復交易日, 1 股變幾股)。0050 於 2025-06-18 以 1 拆 4 恢復交易
SPLITS = {"0050": [("2025-06-18", 4.0)]}


def roc_to_date(text: str) -> pd.Timestamp:
    """'101年10月24日' 或 '101/10/24' → 2012-10-24。"""
    text = text.strip().replace("年", "/").replace("月", "/").replace("日", "")
    y, m, d = (int(x) for x in text.split("/"))
    return pd.Timestamp(year=y + 1911, month=m, day=d)


def to_float(text) -> float:
    text = str(text).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan  # '--' 表示當天無成交


def _get(url: str, params: dict, retries: int = 4, pause: float = 3.0) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            time.sleep(pause * 2 ** (attempt + 1))
    raise RuntimeError(f"TWSE request failed: {url} {params}")


def fetch_daily(stock: str, start_month: str = FIRST_MONTH, end: str = config.DATA_END,
                pause: float = 3.0) -> pd.DataFrame:
    rows: List[list] = []
    for month in pd.period_range(start_month, pd.Timestamp(end).to_period("M"), freq="M"):
        payload = _get(STOCK_DAY_URL, {"date": month.strftime("%Y%m01"), "stockNo": stock, "response": "json"})
        if payload.get("stat") == "OK":
            rows.extend(payload["data"])
        time.sleep(pause)
    df = pd.DataFrame(
        {
            "date": [roc_to_date(r[0]) for r in rows],
            "volume": [to_float(r[1]) for r in rows],
            "open": [to_float(r[3]) for r in rows],
            "high": [to_float(r[4]) for r in rows],
            "low": [to_float(r[5]) for r in rows],
            "close": [to_float(r[6]) for r in rows],
        }
    )
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_ex_rights(stock: str, start_year: int = 2010, end: str = config.DATA_END,
                    pause: float = 3.0) -> pd.DataFrame:
    rows = []
    for year in range(start_year, pd.Timestamp(end).year + 1):
        payload = _get(EX_RIGHT_URL, {"startDate": f"{year}0101", "endDate": f"{year}1231", "response": "json"})
        for r in payload.get("data", []) or []:
            if r[1].strip() == stock:
                rows.append({"date": roc_to_date(r[0]), "prev_close": to_float(r[3]),
                             "reference": to_float(r[4]), "amount": to_float(r[5]), "kind": r[6].strip()})
        time.sleep(pause)
    return pd.DataFrame(rows, columns=["date", "prev_close", "reference", "amount", "kind"])


def adjusted_bars(daily: pd.DataFrame, events: pd.DataFrame, splits=()) -> pd.DataFrame:
    """以最後一天為基準的向後還原價：除權息日之前的價格乘上 參考價 / 前收盤價，分割日之前除以分割比例。"""
    df = daily.dropna(subset=["open", "high", "low", "close"]).set_index("date").sort_index()
    df = df[df["volume"] > 0]
    step = pd.Series(1.0, index=df.index)  # 當天（含）以前要乘的比例，在事件日的前一天記錄
    for ev in events.itertuples():
        if ev.date not in df.index or not (ev.prev_close > 0 and ev.reference > 0):
            continue
        pos = df.index.get_loc(ev.date)
        if pos > 0:
            step.iloc[pos - 1] *= ev.reference / ev.prev_close
    for date, ratio in splits:
        date = pd.Timestamp(date)
        pos = df.index.searchsorted(date)
        prev_close, new_open = df["close"].iloc[pos - 1], df["open"].iloc[pos]
        observed = prev_close / new_open
        if not 0.85 * ratio < observed < 1.15 * ratio:
            raise ValueError(f"split on {date.date()}: price ratio {observed:.2f} != {ratio}")
        step.iloc[pos - 1] /= ratio
    # 由後往前累乘：d 日的因子 = d 日（含）以後記錄的比例乘積 = 所有在 d 日之後生效的事件
    factor = step[::-1].cumprod()[::-1]
    out = df[["open", "high", "low", "close"]].mul(factor, axis=0)
    out["volume"] = df["volume"] / factor
    out["factor"] = factor
    return out


def load_or_fetch(stock: str = "0050", refresh: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = CACHE_DIR.joinpath(f"{stock}_daily.csv")
    events_path = CACHE_DIR.joinpath(f"{stock}_ex_rights.csv")
    if refresh or not daily_path.exists():
        fetch_daily(stock).to_csv(daily_path, index=False)
    if refresh or not events_path.exists():
        fetch_ex_rights(stock).to_csv(events_path, index=False)
    daily = pd.read_csv(daily_path, parse_dates=["date"])
    events = pd.read_csv(events_path, parse_dates=["date"])
    return adjusted_bars(daily, events, SPLITS.get(stock, ()))


def main(stock: Optional[str] = "0050"):
    bars = load_or_fetch(stock, refresh=True)
    print(f"{stock}: {len(bars)} bars {bars.index[0].date()} – {bars.index[-1].date()}")


if __name__ == "__main__":
    main()
