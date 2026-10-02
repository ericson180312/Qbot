"""資料層：讀 tw_collector 產生的日線，做成回測用的還原價 OHLCV。

價格口徑：normalize 輸出的價格是「還原價 / 首日收盤」，factor 也同樣縮放，
所以 價格 / 最後一天的 factor = 以最後一天為基準的向後還原價（等於 Yahoo 的 Adj Close 口徑）。
訊號、報酬、漲跌停判斷都用這個價格；除息日的前收已扣掉股息，等於當天的參考價。
"""

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from . import config

sys.path.insert(0, str(Path(__file__).resolve().parents[1].joinpath("data")))
import tw_collector  # noqa: E402

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def bars_from_normalized(df: pd.DataFrame, end: Optional[str] = config.DATA_END) -> pd.DataFrame:
    """normalize 檔 → 以最後一天為基準的還原價 OHLCV，剔除停牌（價格為 NaN）的日子。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if end is not None:
        df = df.loc[: pd.Timestamp(end)]
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    last_factor = df["factor"].iloc[-1]
    out = df[BAR_COLUMNS].copy()
    for col in ["open", "high", "low", "close"]:
        out[col] = out[col] / last_factor
    out["volume"] = out["volume"] * last_factor
    return out


def load_bars(symbol: str, base_dir: Path = config.BASE_DIR, end: Optional[str] = config.DATA_END) -> pd.DataFrame:
    path = Path(base_dir).joinpath(tw_collector.NORMALIZE_DIR_NAME, f"{symbol}.csv")
    return bars_from_normalized(pd.read_csv(path), end=end)


def open_limit_flags(bars: pd.DataFrame) -> pd.DataFrame:
    """開盤是否在漲停 / 跌停：開盤相對前一個交易日收盤的漲跌幅超過門檻。

    第一天沒有前收，視為不在漲跌停。停牌日已剔除，所以「前一日」是前一個有成交的日子。
    """
    prev_close = bars["close"].shift(1)
    change = bars["open"] / prev_close - 1
    threshold = pd.Series(
        np.where(bars.index >= pd.Timestamp(config.LIMIT_CHANGE_DATE),
                 config.LIMIT_THRESHOLD_NEW, config.LIMIT_THRESHOLD_OLD),
        index=bars.index,
    )
    return pd.DataFrame(
        {
            "limit_up_open": (change >= threshold).fillna(False),
            "limit_down_open": (change <= -threshold).fillna(False),
        },
        index=bars.index,
    )


def limit_for_dates(index: pd.DatetimeIndex) -> np.ndarray:
    return np.where(index >= pd.Timestamp(config.LIMIT_CHANGE_DATE), config.LIMIT_NEW, config.LIMIT_OLD)


def bad_days(bars: pd.DataFrame) -> pd.Series:
    """收盤或開盤相對前一日收盤的漲跌超過制度上限 + 0.5pp：不可能是正常交易，視為資料問題。"""
    prev = bars["close"].shift(1)
    cap = limit_for_dates(bars.index) + config.JUMP_TOLERANCE
    close_jump = (bars["close"] / prev - 1).abs() > cap
    open_jump = (bars["open"] / prev - 1).abs() > cap
    return (close_jump | open_jump).fillna(False)


def load_t1_bars() -> pd.DataFrame:
    """0050：證交所官方日線 + 除權息參考價自建的含息還原價（見 twse.py 為何不用 Yahoo）。"""
    from . import twse

    bars = twse.load_or_fetch("0050")
    return bars.loc[: pd.Timestamp(config.DATA_END), BAR_COLUMNS]


def load_stock_list(base_dir: Path = config.BASE_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(base_dir).joinpath(tw_collector.STOCK_LIST_FILE), dtype=str)


def universe_spans(
    top_n: int = config.T2_TOP_N,
    window: int = config.T2_WINDOW,
    freq: str = config.T2_FREQ,
    base_dir: Path = config.BASE_DIR,
) -> pd.DataFrame:
    """動態股票池的入選區間（symbol / start / end），沿用 tw_collector 的無前視選股。"""
    base_dir = Path(base_dir)
    calendar = tw_collector.load_calendar(base_dir)
    calendar = calendar[calendar <= pd.Timestamp(config.DATA_END)]
    values = tw_collector._load_trading_values(base_dir, calendar)
    return tw_collector.compute_liquidity_universe(values, top_n=top_n, window=window, freq=freq)


def membership(spans: pd.DataFrame, symbol: str, index: pd.DatetimeIndex) -> pd.Series:
    """symbol 在每個交易日是否屬於股票池。"""
    flag = pd.Series(False, index=index)
    for row in spans[spans["symbol"] == symbol].itertuples():
        flag.loc[pd.Timestamp(row.start): pd.Timestamp(row.end)] = True
    return flag


def load_many(symbols: Iterable[str], base_dir: Path = config.BASE_DIR) -> Dict[str, pd.DataFrame]:
    out = {}
    for symbol in symbols:
        try:
            bars = load_bars(symbol, base_dir)
        except FileNotFoundError:
            continue
        if not bars.empty:
            out[symbol] = bars
    return out
