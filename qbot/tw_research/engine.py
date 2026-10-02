"""backtrader 回測引擎：單一標的、單一策略，輸出每日持有旗標、成交與淨值。"""

from dataclasses import dataclass
from typing import Optional, Type

import backtrader as bt
import pandas as pd

from . import broker, config, data
from .strategies import TWStrategy

EXTRA_LINES = ("in_universe", "limit_up_open", "limit_down_open", "ssa", "target")


class TWPandasData(bt.feeds.PandasData):
    lines = EXTRA_LINES
    params = tuple((name, -1) for name in EXTRA_LINES) + (("openinterest", None),)


@dataclass
class RunResult:
    symbol: str
    code: str
    held: pd.Series  # 收盤時是否持有（= 從當天開盤持有到隔天開盤）
    values: pd.Series  # 每日收盤淨值
    fills: pd.DataFrame  # date / side / size / price / comm
    blocked_buys: int
    delayed_sells: int


def build_feed_frame(bars: pd.DataFrame, strategy_cls: Type[TWStrategy],
                     in_universe: Optional[pd.Series] = None) -> pd.DataFrame:
    frame = bars[data.BAR_COLUMNS].copy()
    flags = data.open_limit_flags(bars)
    frame["limit_up_open"] = flags["limit_up_open"].astype(float)
    frame["limit_down_open"] = flags["limit_down_open"].astype(float)
    if in_universe is None:
        frame["in_universe"] = 1.0
    else:
        frame["in_universe"] = in_universe.reindex(frame.index).fillna(False).astype(float)
    frame["ssa"] = float("nan")
    frame["target"] = float("nan")
    for name, series in strategy_cls.prepare(bars).items():
        frame[name] = series.reindex(frame.index)
    return frame


def run(bars: pd.DataFrame, strategy_cls: Type[TWStrategy], symbol: str,
        in_universe: Optional[pd.Series] = None, cash: float = config.INITIAL_CASH,
        fee_multiplier: float = 1.0, **strategy_kwargs) -> RunResult:
    frame = build_feed_frame(bars, strategy_cls, in_universe)
    cerebro = bt.Cerebro(cheat_on_open=True, stdstats=False)
    cerebro.adddata(TWPandasData(dataname=frame), name=symbol)
    comm = broker.commission_for(symbol, fee_multiplier)
    cerebro.broker.addcommissioninfo(comm)
    cerebro.broker.setcash(cash)
    cerebro.addstrategy(strategy_cls, buy_cost_rate=comm.p.fee_rate, **strategy_kwargs)
    strat = cerebro.run()[0]

    held = pd.Series(dict(strat.held), dtype=bool)
    held.index = pd.to_datetime(held.index)
    values = pd.Series(dict(strat.values), dtype=float)
    values.index = pd.to_datetime(values.index)
    fills = pd.DataFrame(strat.fills, columns=["date", "side", "size", "price", "comm"])
    fills["date"] = pd.to_datetime(fills["date"])
    return RunResult(symbol, strategy_cls.code, held, values, fills,
                     strat.blocked_buys, strat.delayed_sells)


def trades_from_held(held: pd.Series) -> pd.DataFrame:
    """由每日持有旗標還原每筆交易：entry = 開始持有的那天開盤，exit = 結束持有的那天開盤。

    最後一天仍持有的交易 exit 為 NaT（未平倉）。
    """
    h = held.astype(int).to_numpy()
    idx = held.index
    rows = []
    start = None
    for i, v in enumerate(h):
        if v and start is None:
            start = i
        elif not v and start is not None:
            rows.append((idx[start], idx[i], start, i))
            start = None
    if start is not None:
        rows.append((idx[start], pd.NaT, start, -1))
    return pd.DataFrame(rows, columns=["entry_date", "exit_date", "entry_i", "exit_i"])
