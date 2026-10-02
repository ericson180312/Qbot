"""移植到台股的 Qbot 擇時策略。

每個策略只保留原始碼的「進出場規則」，下單細節統一由 TWStrategy 處理：
    - t 日收盤判斷訊號，t+1 日開盤以市價成交（原始碼在 next() 下市價單，backtrader 也是隔日開盤成交）
    - 全額買進（整股、可零股），空手時現金報酬為 0
    - 開盤漲停買不到：該次訊號作廢，隔天由規則重新判斷
    - 開盤跌停賣不掉：順延到下一個非跌停的開盤
    - 只做多（台股散戶放空要融券，原始碼裡的放空分支拿掉）
    - 只有當天在股票池內才能進場；出場不受股票池限制

和原始碼的差異逐條列在 docs/03-智能策略/台股策略回測與因子挖掘.md。
"""

import math
from typing import Dict

import backtrader as bt
import numpy as np
import pandas as pd

from . import indicators


class TWStrategy(bt.Strategy):
    """共同的下單、記錄邏輯。子類別實作 buy_signal() / sell_signal()（在 t 日收盤呼叫）。"""

    code = ""
    source = ""
    params = (("buy_cost_rate", 0.0),)

    @classmethod
    def prepare(cls, bars: pd.DataFrame) -> Dict[str, pd.Series]:
        """需要預先用 pandas 算好的欄位（例如 ssa、target）。"""
        return {}

    def __init__(self):
        self.pending = None
        self.signal_bar = None
        self.just_filled = False
        self.entry_close = None
        self.blocked_buys = 0
        self.delayed_sells = 0
        self.held = []
        self.values = []
        self.fills = []

    # ------------------------------------------------------------ 規則介面
    def buy_signal(self) -> bool:
        raise NotImplementedError

    def sell_signal(self) -> bool:
        raise NotImplementedError

    # ------------------------------------------------------------ 每日流程
    def _record(self):
        dt = self.data.datetime.date(0)
        self.held.append((dt, self.position.size > 0))
        self.values.append((dt, self.broker.getvalue()))

    def prenext(self):
        self._record()

    def next(self):
        if self.just_filled:  # 今天開盤剛買進：記下進場日收盤（教學程式的 bar_executed_close）
            self.entry_close = self.data.close[0]
            self.just_filled = False
        self._record()
        if self.pending == "sell":
            return
        if self.position:
            if self.sell_signal():
                self.pending = "sell"
        elif self.data.in_universe[0] > 0 and self.buy_signal():
            self.pending = "buy"
            self.signal_bar = len(self)

    def next_open(self):
        if self.pending == "buy":
            self.pending = None
            if self.data.limit_up_open[0] > 0:
                self.blocked_buys += 1
                return
            price = self.data.open[0]
            comminfo = self.broker.getcommissioninfo(self.data)
            cash = self.broker.getcash() - comminfo.p.min_fee
            size = math.floor(cash / (price * (1 + self.p.buy_cost_rate) * (1 + 1e-9)))
            if size > 0:
                self.buy(size=size)
                self.just_filled = True
        elif self.pending == "sell":
            if self.data.limit_down_open[0] > 0:
                self.delayed_sells += 1
                return
            self.pending = None
            self.close()

    def notify_order(self, order):
        if order.status == order.Completed:
            self.fills.append(
                (
                    self.data.datetime.date(0),
                    "buy" if order.isbuy() else "sell",
                    order.executed.size,
                    order.executed.price,
                    order.executed.comm,
                )
            )
        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            raise RuntimeError(f"order {order.getstatusname()} on {self.data.datetime.date(0)}")


# ================================================================ 均線類
class BiggerThanSma(TWStrategy):
    code = "S01"
    source = "qbot/strategies/bigger_than_ema_bt.py"
    params = (("maperiod", 15),)

    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SimpleMovingAverage(self.data, period=self.p.maperiod)

    def buy_signal(self):
        return self.data.close[0] > self.sma[0]

    def sell_signal(self):
        return self.data.close[0] < self.sma[0]


class SmaCross(TWStrategy):
    code = "S02"
    source = "qbot/strategies/sma_cross_strategy_bt.py"
    params = (("pfast", 10), ("pslow", 30))

    def __init__(self):
        super().__init__()
        sma1 = bt.indicators.SMA(self.data, period=self.p.pfast)
        sma2 = bt.indicators.SMA(self.data, period=self.p.pslow)
        self.crossover = bt.indicators.CrossOver(sma1, sma2)

    def buy_signal(self):
        return self.crossover[0] > 0

    def sell_signal(self):
        return self.crossover[0] < 0


class KlinesSma5(TWStrategy):
    code = "S03"
    source = "qbot/strategies/klines_bt.py"
    params = (("maperiod", 5),)

    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SimpleMovingAverage(self.data, period=self.p.maperiod)

    def buy_signal(self):
        return self.data.close[0] > self.sma[0]

    def sell_signal(self):
        return self.data.close[0] < self.sma[0]


class BollReversion(TWStrategy):
    code = "S04"
    source = "qbot/strategies/boll_strategy_bt.py"
    params = (("nk", 13),)

    def __init__(self):
        super().__init__()
        boll = bt.indicators.BollingerBands(self.data, period=self.p.nk)
        self.top = boll.top
        self.bot = boll.bot

    def buy_signal(self):
        return self.data.close[0] < self.bot[0]

    def sell_signal(self):
        return self.data.close[0] > self.top[0]


class RsiSma(TWStrategy):
    code = "S05"
    source = "qbot/strategies/multi_strategy_bt.py"
    params = (("maperiod", 15),)

    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SimpleMovingAverage(self.data, period=self.p.maperiod)
        self.rsi = bt.indicators.RelativeStrengthIndex(self.data)

    def buy_signal(self):
        return self.rsi[0] < 50 and self.rsi[-1] >= 50 and self.data.close[0] > self.sma[0]

    def sell_signal(self):
        return self.rsi[0] > 50 and self.rsi[-1] <= 50 and self.data.close[0] < self.sma[0]


class Ssa(TWStrategy):
    code = "S06"
    source = "qbot/strategies/ssa_strategy_bt.py"
    ssa_window = 15

    @classmethod
    def prepare(cls, bars):
        return {"ssa": indicators.ssa_last(bars["close"], cls.ssa_window)}

    def buy_signal(self):
        return not math.isnan(self.data.ssa[0]) and self.data.close[0] > self.data.ssa[0]

    def sell_signal(self):
        return not math.isnan(self.data.ssa[0]) and self.data.close[0] < self.data.ssa[0]


class AdxMacd(TWStrategy):
    """EMA13 > EMA55 > EMA89、ADX ≤ 25 且上升、MACD 柱狀體上升時買進，持有 3 天。"""

    code = "S07"
    source = "qbot/strategies/adx_strategy.py"
    params = (("ma1", 13), ("ma2", 55), ("ma3", 89), ("adx", 25), ("hold", 3))

    def __init__(self):
        super().__init__()
        self.ema1 = bt.indicators.EMA(self.data, period=self.p.ma1)
        self.ema2 = bt.indicators.EMA(self.data, period=self.p.ma2)
        self.ema3 = bt.indicators.EMA(self.data, period=self.p.ma3)
        self.hist = bt.indicators.MACDHisto(self.data, period_me1=12, period_me2=26, period_signal=9).histo
        self.adxi = bt.indicators.AverageDirectionalMovementIndex(self.data, period=14)

    def buy_signal(self):
        return (
            self.ema1[0] > self.ema2[0] > self.ema3[0]
            and self.adxi[0] <= self.p.adx
            and self.adxi[0] > self.adxi[-1]
            and self.hist[0] > self.hist[-1]
        )

    def sell_signal(self):
        return len(self) - self.signal_bar >= self.p.hold


# ================================================================ MACD / KDJ / K 線型態
def _kdj(strategy, period=9):
    high_n = bt.indicators.Highest(strategy.data.high, period=period)
    low_n = bt.indicators.Lowest(strategy.data.low, period=period)
    rsv = 100 * bt.DivByZero(strategy.data.close - low_n, high_n - low_n, zero=None)
    k = bt.indicators.EMA(rsv, period=3, plot=False)
    d = bt.indicators.EMA(k, period=3, plot=False)
    return k, d, 3 * k - 2 * d


class MacdTenPct(TWStrategy):
    """MACD 金叉買進；收盤相對進場日收盤漲跌超過 10% 時出場（docs/tutorials_code 02）。"""

    code = "S08"
    source = "docs/tutorials_code/02.easy_macd_strategy/macd.py"

    def __init__(self):
        super().__init__()
        me1 = bt.indicators.EMA(self.data, period=12)
        me2 = bt.indicators.EMA(self.data, period=26)
        self.macd = me1 - me2
        self.signal = bt.indicators.EMA(self.macd, period=5)

    def buy_signal(self):
        return self.macd[-1] - self.signal[-1] < 0 and self.macd[0] - self.signal[0] > 0

    def sell_signal(self):
        change = (self.data.close[0] - self.entry_close) / self.data.close[0]
        return change > 0.1 or change < -0.1


class Kdj(TWStrategy):
    """J 上穿 D 買進、J 下穿 D 賣出（docs/tutorials_code 04 kdj.py）。

    原始碼的賣出條件寫成 `J[-1] > D[-1] or J[0] < D[0]`，進場隔天必定成立，
    每筆交易都只持有一天；這裡改成死叉（and），見文件的移植差異表。
    """

    code = "S09"
    source = "docs/tutorials_code/04.kdj_with_macd/kdj.py"

    def __init__(self):
        super().__init__()
        self.K, self.D, self.J = _kdj(self)

    def buy_signal(self):
        return self.J[-1] - self.D[-1] < 0 and self.J[0] - self.D[0] > 0

    def sell_signal(self):
        return self.J[-1] - self.D[-1] > 0 and self.J[0] - self.D[0] < 0


class KdjMacd(TWStrategy):
    """MACD(12,26,9) 金叉買進、KDJ 死叉賣出（docs/tutorials_code 04 kdj_macd.py，賣出條件同 S09 修正）。"""

    code = "S10"
    source = "docs/tutorials_code/04.kdj_with_macd/kdj_macd.py"

    def __init__(self):
        super().__init__()
        self.K, self.D, self.J = _kdj(self)
        me1 = bt.indicators.EMA(self.data, period=12)
        me2 = bt.indicators.EMA(self.data, period=26)
        self.macd = me1 - me2
        self.signal = bt.indicators.EMA(self.macd, period=9)

    def buy_signal(self):
        return self.macd[-1] - self.signal[-1] < 0 and self.macd[0] - self.signal[0] > 0

    def sell_signal(self):
        return self.J[-1] - self.D[-1] > 0 and self.J[0] - self.D[0] < 0


class Harami(TWStrategy):
    """當日高低點都落在前一日 K 棒實體內（孕線）時買進；漲跌超過 10% 出場（docs/tutorials_code 07）。"""

    code = "S11"
    source = "docs/tutorials_code/07.harami/harami.py"

    def buy_signal(self):
        d = self.data
        if len(self) < 2:  # 第一根沒有前一日 K 棒
            return False
        if d.close[-1] < d.open[-1]:
            return d.high[0] < d.open[-1] and d.low[0] > d.close[-1]
        return d.high[0] < d.close[-1] and d.low[0] > d.open[-1]

    def sell_signal(self):
        change = (self.data.close[0] - self.entry_close) / self.data.close[0]
        return change > 0.1 or change < -0.1


# ================================================================ 振盪指標（pandas 原始碼）
class _TargetStrategy(TWStrategy):
    """原始碼直接產生部位序列（1 持有、0 空手）的策略：照序列進出。"""

    def buy_signal(self):
        return self.data.target[0] > 0.5

    def sell_signal(self):
        return self.data.target[0] < 0.5


class StochRsiReversion(_TargetStrategy):
    code = "S12"
    source = "pytrader/strategies/StochRSI.py"

    @classmethod
    def prepare(cls, bars):
        return {"target": indicators.stoch_rsi_reversion_target(bars["close"])}


class RsiDivergence(_TargetStrategy):
    code = "S13"
    source = "pytrader/strategies/rsi_departure_strategy.py"

    @classmethod
    def prepare(cls, bars):
        return {"target": indicators.rsi_divergence_target(bars["close"])}


class Rsi6Reversal(TWStrategy):
    """pytrader/strategies/CCI.py 的 RSI6 反轉（原為分 K，這裡用日 K）。

    RSI6 最近兩根任一 < 35 且回升 → 買進；任一 > 80 且不再上升 → 賣出；
    其餘情況若收盤跌破 23 日均價且低於成本 → 停損（原始碼註解寫 20 日線，實際取最近 23 根）。
    """

    code = "S14"
    source = "pytrader/strategies/CCI.py"

    def __init__(self):
        super().__init__()
        self.rsi6 = bt.indicators.RelativeStrengthIndex(self.data, period=6, safediv=True)
        self.ma = bt.indicators.SimpleMovingAverage(self.data, period=23)

    def _zones(self):
        last, prev = self.rsi6[0], self.rsi6[-1]
        return last, prev, (last > 80 or prev > 80), (last < 35 or prev < 35)

    def buy_signal(self):
        last, prev, high, low = self._zones()
        return (not high) and low and last > prev

    def sell_signal(self):
        last, prev, high, low = self._zones()
        if high:
            return not last > prev
        if low and last > prev:
            return False
        return self.data.close[0] < self.ma[0] and self.position.price > self.data.close[0]


STRATEGIES = [
    BiggerThanSma, SmaCross, KlinesSma5, BollReversion, RsiSma, Ssa, AdxMacd,
    MacdTenPct, Kdj, KdjMacd, Harami, StochRsiReversion, RsiDivergence, Rsi6Reversal,
]
STRATEGY_BY_CODE = {s.code: s for s in STRATEGIES}

# 沒有移植的檔案與原因（寫進報告）
NOT_PORTED = {
    "qbot/strategies/arbr_strategy.py": "只畫 AR/BR 指標，沒有進出場規則",
    "qbot/strategies/lstm_strategy_bt.py": "前視偏差：scaler 用全期資料 fit，且每天都拿測試集最後一筆的預測下單；"
                                           "ML 選股改由 B 部分的 qlib 流程檢驗",
    "qbot/strategies/rl_strategy_bt.py": "依賴不存在的 rlkit BacktraderEnv，self.model 從未建立，無法執行",
    "pytrader/strategies/lgb_strategy.py": "label 寫成 close.shift(5)/close-1（過去報酬，不是未來報酬），"
                                           "且是 4 檔 A 股指數輪動；ML 選股改由 B 部分檢驗",
    "pytrader/strategies/svm_strategy.py": "同 lgb_strategy.py 的架構，依賴不存在的 model.svm 模組",
    "pytrader/strategies/q-learning.py": "用 A 股 5 分 K 訓練；台股沒有免費的長期分 K",
    "pytrader/strategies/aroon_strategy.py": "在 28 個申萬行業指數間輪動，缺 TemplateStrategy 基底類別；台股沒有對應的免費產業指數日線",
    "pytrader/strategies/multi_factor.py": "聚寬平台腳本，需要市值、ROE 等基本面資料（Yahoo 沒有）",
    "pytrader/strategies/multi_factor_strategy.py": "程式無法執行（PEFetcher 傳入參數順序錯、get_data 未用到 start/end），且需要基本面資料",
    "qbot/strategies/undervalued_stock_picking_strategy.py": "聚寬平台腳本，需要 PE、PB 等基本面資料",
    "pytrader/strategies/中概互联_自动T.py": "單一 A 股 ETF 的盤中做 T",
    "pytrader/strategies/k-line.py / monte_carlo.py / example.py": "畫圖與範例，不是策略",
}
