#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""qbot/tw_research 的測試：只用合成資料，不需要網路或 qlib 資料。

涵蓋：無前視、t+1 開盤成交、可手算的損益、漲跌停、股票池限制、移植與原始碼一致、統計工具。
"""

import sys
import unittest
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("qbot")))

from scipy.signal import argrelextrema  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from tw_research import analysis, broker, config, data, engine, factors, indicators  # noqa: E402
from tw_research import strategies as S  # noqa: E402


def make_bars(n=600, seed=0, start="2016-01-04", drift=0.0, vol=0.015):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    op = close * np.exp(rng.normal(0, 0.004, n))
    hi = np.maximum(op, close) * (1 + rng.uniform(0, 0.01, n))
    lo = np.minimum(op, close) * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": 1e6}, index=idx)


# ---------------------------------------------------------------- 原始碼參考實作（照抄，用來比對）
def original_calc_rsi(close, P=14):
    data = pd.DataFrame({"Close": close.to_numpy()})
    data["diff_close"] = data["Close"] - data["Close"].shift(1)
    data["gain"] = np.where(data["diff_close"] > 0, data["diff_close"], 0)
    data["loss"] = np.where(data["diff_close"] < 0, np.abs(data["diff_close"]), 0)
    data[["init_avg_gain", "init_avg_loss"]] = data[["gain", "loss"]].rolling(P).mean()
    avg_gain = np.zeros(len(data))
    avg_loss = np.zeros(len(data))
    for i, _row in enumerate(data.iterrows()):
        row = _row[1]
        if i < P - 1:
            continue
        elif i == P - 1:
            avg_gain[i] += row["init_avg_gain"]
            avg_loss[i] += row["init_avg_loss"]
        else:
            avg_gain[i] += ((P - 1) * avg_gain[i - 1] + row["gain"]) / P
            avg_loss[i] += ((P - 1) * avg_loss[i - 1] + row["loss"]) / P
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def original_ssa(input_array, t, m):
    temp = [input_array[i:i + m] for i in range(t - m + 1)]
    window_matrix = np.array(temp)
    u, s, v = np.linalg.svd(window_matrix)
    m1, _ = u.shape
    _, n2 = v.shape
    index = s.argmax()
    new_matrix = s.max() * np.dot(u[:, index].reshape((m1, 1)), v[index].reshape((1, n2)))
    ret = []
    n = t - m + 1
    for p in range(1, t + 1):
        if p < m:
            alpha = p
        elif p > t - m + 1:
            alpha = t - p + 1
        else:
            alpha = m
        sigma = 0
        for j in range(1, m + 1):
            i = p - j + 1
            if 0 < i < n + 1:
                sigma += new_matrix[i - 1][j - 1]
        ret.append(sigma / alpha)
    return ret


def original_extrema(data, kind, order=5, K=2):
    """rsi_departure_strategy.py 的 getHigherHighs / getLowerHighs / getLowerLows / getHigherLows + get*Index。"""
    if kind in ("HH", "LH"):
        idx = argrelextrema(data, np.greater, order=order)[0]
    else:
        idx = argrelextrema(data, np.less, order=order)[0]
    vals = data[idx]
    extrema, ex_deque = [], deque(maxlen=K)
    for i, ix in enumerate(idx):
        if i == 0:
            ex_deque.append(ix)
            continue
        if kind == "HH" and vals[i] < vals[i - 1]:
            ex_deque.clear()
        if kind == "LH" and vals[i] > vals[i - 1]:
            ex_deque.clear()
        if kind == "LL" and vals[i] > vals[i - 1]:
            ex_deque.clear()
        if kind == "HL" and vals[i] < vals[i - 1]:
            ex_deque.clear()
        ex_deque.append(ix)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())
    out = np.array([e[-1] + order for e in extrema])
    return out[np.where(out < len(data))]


# ---------------------------------------------------------------- 資料層
class TestData(unittest.TestCase):
    def test_bars_rescaled_to_last_factor(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        raw = np.array([100.0, 101.0, 95.0, 96.0])
        factor = np.array([0.95, 0.95, 1.0, 1.0])  # 第 3 天除息
        first_close = raw[0] * factor[0]
        df = pd.DataFrame({
            "date": dates, "symbol": "TW1",
            "open": raw * factor / first_close, "high": raw * factor / first_close,
            "low": raw * factor / first_close, "close": raw * factor / first_close,
            "volume": 1000 / factor * first_close, "vwap": 0, "factor": factor / first_close, "change": 0,
        })
        bars = data.bars_from_normalized(df, end=None)
        # 最後一天等於原始價、除息前的價格乘上 0.95
        np.testing.assert_allclose(bars["close"].to_numpy(), raw * factor)
        np.testing.assert_allclose(bars["volume"].to_numpy(), 1000 / factor)

    def test_suspended_days_dropped_and_end_cut(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        df = pd.DataFrame({"date": dates, "open": [1, np.nan, 1, 1], "high": 1, "low": 1,
                           "close": [1, np.nan, 1, 1], "volume": 1, "factor": 1.0})
        bars = data.bars_from_normalized(df, end=str(dates[2].date()))
        self.assertEqual(list(bars.index), [dates[0], dates[2]])

    def test_limit_threshold_switches_on_2015_06_01(self):
        idx = pd.DatetimeIndex(["2015-05-27", "2015-05-28", "2015-05-29", "2015-06-01", "2015-06-02"])
        bars = pd.DataFrame({"open": [10, 10.7, 10, 10.7, 9.0], "close": [10, 10, 10, 10, 10],
                             "high": 11, "low": 9, "volume": 1}, index=idx)
        flags = data.open_limit_flags(bars)
        # 05-28 開盤 +7%（舊制門檻 6.5% → 漲停）；06-01 同樣 +7%（新制門檻 9.5% → 不算）；06-02 −10% 跌停
        self.assertEqual(flags["limit_up_open"].tolist(), [False, True, False, False, False])
        self.assertEqual(flags["limit_down_open"].tolist(), [False, False, False, False, True])

    def test_bad_days(self):
        idx = pd.DatetimeIndex(["2015-05-27", "2015-05-28", "2015-05-29", "2016-01-04", "2016-01-05"])
        bars = pd.DataFrame({"open": [10, 10.6, 10, 10, 11.1], "close": [10, 10.6, 10, 11.04, 10],
                             "high": 12, "low": 9, "volume": 1}, index=idx)
        # 05-28 +6%（舊制上限 7% 內）；05-29 −5.7%；2016-01-04 +10.4%（新制 10.5% 內）；01-05 收盤 −9.4%、開盤 +0.5%
        self.assertEqual(data.bad_days(bars).tolist(), [False, False, False, False, False])
        bars.loc["2015-05-29", "close"] = 9.8  # 相對 10.6 為 −7.55%，超過舊制 7% + 0.5pp
        # 連帶 2016-01-04 收盤 11.04 相對 9.8 為 +12.7%，也超過新制 10% + 0.5pp
        self.assertEqual(data.bad_days(bars).tolist(), [False, False, True, True, False])

    def test_twse_adjustment_hand_checked(self):
        from tw_research import twse

        daily = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]),
            "open": [100, 100, 98, 99, 25], "high": [101, 101, 99, 100, 25.5],
            "low": [99, 99, 97, 98, 24.5], "close": [100, 100, 98, 100, 25], "volume": [10, 10, 10, 10, 40],
        })
        events = pd.DataFrame({"date": pd.to_datetime(["2020-01-06"]), "prev_close": [100.0],
                               "reference": [98.0], "amount": [2.0], "kind": ["息"]})
        bars = twse.adjusted_bars(daily, events, splits=[("2020-01-08", 4.0)])
        # 除息日（01-06）之前乘 0.98；分割（01-08）之前再除以 4
        np.testing.assert_allclose(bars["close"].to_numpy(), [24.5, 24.5, 24.5, 25.0, 25.0])
        np.testing.assert_allclose(bars["volume"].to_numpy(), [10 / 0.245, 10 / 0.245, 40, 40, 40])
        with self.assertRaises(ValueError):  # 價格比對不上分割比例時拒絕
            twse.adjusted_bars(daily, events, splits=[("2020-01-08", 2.0)])

    def test_roc_date(self):
        from tw_research import twse

        self.assertEqual(twse.roc_to_date("101年10月24日"), pd.Timestamp("2012-10-24"))
        self.assertEqual(twse.roc_to_date("115/09/30"), pd.Timestamp("2026-09-30"))

    def test_membership(self):
        spans = pd.DataFrame({"symbol": ["TW1", "TW1"], "start": ["2020-01-02", "2020-01-06"],
                              "end": ["2020-01-02", "2020-01-07"]})
        idx = pd.bdate_range("2020-01-01", "2020-01-08")
        m = data.membership(spans, "TW1", idx)
        # 交易日：1/1、1/2、1/3、1/6、1/7、1/8
        self.assertEqual(m.tolist(), [False, True, False, True, True, False])


# ---------------------------------------------------------------- 指標：無前視、與原始碼一致
class TestIndicators(unittest.TestCase):
    def setUp(self):
        self.close = make_bars(500, seed=3)["close"]

    def assert_causal(self, fn):
        full = fn(self.close)
        for cut in (120, 260, 400):
            part = fn(self.close.iloc[:cut])
            np.testing.assert_allclose(part.to_numpy(), full.iloc[:cut].to_numpy(), equal_nan=True,
                                       err_msg=f"{fn.__name__} changes past values when future data is added")

    def test_no_lookahead(self):
        self.assert_causal(lambda c: indicators.ssa_last(c, 15))
        self.assert_causal(indicators.stoch_rsi_reversion_target)
        self.assert_causal(indicators.rsi_divergence_target)
        self.assert_causal(lambda c: indicators.rsi_qbot(c, 14))

    def test_rsi_matches_original(self):
        np.testing.assert_allclose(indicators.rsi_qbot(self.close, 14).to_numpy()[13:],
                                   original_calc_rsi(self.close, 14)[13:])

    def test_ssa_matches_original(self):
        ssa = indicators.ssa_last(self.close, 15)
        v = self.close.to_numpy()
        for end in (29, 30, 100, 499):
            expected = original_ssa(v[end - 29:end + 1], 30, 15)[-1]
            self.assertAlmostEqual(ssa.iloc[end], expected, places=8)
        self.assertTrue(ssa.iloc[:29].isna().all())

    def test_peak_confirmation_matches_original(self):
        values = self.close.to_numpy()
        highs, lows = indicators.peak_states(values, order=5, K=2)
        hh, lh = original_extrema(values, "HH"), original_extrema(values, "LH")
        ll, hl = original_extrema(values, "LL"), original_extrema(values, "HL")
        expected_highs = pd.Series(np.nan, index=range(len(values)))
        expected_highs[hh] = 1
        expected_highs[lh] = -1
        expected_lows = pd.Series(np.nan, index=range(len(values)))
        expected_lows[ll] = 1
        expected_lows[hl] = -1
        np.testing.assert_array_equal(highs, expected_highs.ffill().fillna(0).to_numpy())
        np.testing.assert_array_equal(lows, expected_lows.ffill().fillna(0).to_numpy())

    def test_stoch_rsi_target_rules(self):
        target = indicators.stoch_rsi_reversion_target(self.close)
        srsi = indicators.stoch_rsi(self.close)
        self.assertTrue((target[srsi < 20] == 1).all())
        self.assertTrue((target[srsi > 80] == 0).all())
        between = (srsi >= 20) & (srsi <= 80)
        self.assertTrue((target[between] == target.shift(1).fillna(0)[between]).all())


# ---------------------------------------------------------------- 引擎
class OnDates(S.TWStrategy):
    """測試用：在指定日期收盤發出買 / 賣訊號。"""

    code = "TEST"
    buy_dates = ()
    sell_dates = ()

    def buy_signal(self):
        return self.data.datetime.date(0) in self.buy_dates

    def sell_signal(self):
        return self.data.datetime.date(0) in self.sell_dates


def constant_bars(opens, start="2020-01-02"):
    idx = pd.bdate_range(start, periods=len(opens))
    opens = np.asarray(opens, dtype=float)
    return pd.DataFrame({"open": opens, "high": opens * 1.001, "low": opens * 0.999,
                         "close": opens, "volume": 1e6}, index=idx)


class TestEngine(unittest.TestCase):
    def run_on_dates(self, bars, buy_i, sell_i, symbol="TW1101", universe=None, cash=1_000_000):
        strat = type("T", (OnDates,), {"buy_dates": {bars.index[i].date() for i in buy_i},
                                       "sell_dates": {bars.index[i].date() for i in sell_i}})
        return engine.run(bars, strat, symbol, in_universe=universe, cash=cash)

    def test_hand_checked_round_trip(self):
        bars = constant_bars([100, 100, 100, 110, 110, 110])
        res = self.run_on_dates(bars, buy_i=[1], sell_i=[2])
        # 第 1 天收盤訊號 → 第 2 天開盤 100 元買；第 2 天收盤訊號 → 第 3 天開盤 110 元賣
        self.assertEqual(res.fills["date"].tolist(), [bars.index[2], bars.index[3]])
        fee = 0.001425 * 0.28
        size = 9995  # floor((1,000,000 − 20) / (100 × 1.000399)) = floor(9995.8)
        buy_comm = max(size * 100 * fee, 20)
        sell_comm = max(size * 110 * fee, 20) + size * 110 * 0.003
        expected = 1_000_000 - size * 100 - buy_comm + size * 110 - sell_comm
        self.assertEqual(int(res.fills["size"].iloc[0]), size)
        self.assertAlmostEqual(res.values.iloc[-1], expected, places=4)
        self.assertAlmostEqual(expected, 1_095_814.1690, places=3)
        self.assertEqual(res.held.tolist(), [False, False, True, False, False, False])

    def test_etf_uses_lower_tax(self):
        bars = constant_bars([100, 100, 100, 110, 110, 110])
        res = self.run_on_dates(bars, buy_i=[1], sell_i=[2], symbol="TW0050")
        sell = res.fills.iloc[1]
        value = abs(sell["size"]) * 110
        self.assertAlmostEqual(sell["comm"], max(value * 0.000399, 20) + value * 0.001)

    def test_limit_up_open_blocks_buy(self):
        bars = constant_bars([100, 100, 110, 110, 110, 110])  # 第 2 天開盤 +10% 漲停
        res = self.run_on_dates(bars, buy_i=[1], sell_i=[])
        self.assertTrue(res.fills.empty)
        self.assertEqual(res.blocked_buys, 1)

    def test_limit_down_open_delays_sell(self):
        bars = constant_bars([100, 100, 100, 90, 89, 89, 89])  # 第 3 天開盤 −10% 跌停
        res = self.run_on_dates(bars, buy_i=[1], sell_i=[2])
        self.assertEqual(res.fills["date"].tolist(), [bars.index[2], bars.index[4]])
        self.assertEqual(res.delayed_sells, 1)
        self.assertTrue(res.held.iloc[3])

    def test_no_entry_outside_universe(self):
        bars = constant_bars([100] * 8)
        universe = pd.Series([False, False, False, True, True, True, True, True], index=bars.index)
        res = self.run_on_dates(bars, buy_i=[1, 4], sell_i=[], universe=universe)
        self.assertEqual(res.fills["date"].tolist(), [bars.index[5]])

    def test_trades_from_held(self):
        held = pd.Series([0, 1, 1, 0, 1], index=pd.bdate_range("2020-01-01", periods=5)).astype(bool)
        t = engine.trades_from_held(held)
        self.assertEqual(t["entry_i"].tolist(), [1, 4])
        self.assertEqual(t["exit_i"].tolist(), [3, -1])
        self.assertTrue(pd.isna(t["exit_date"].iloc[1]))

    def test_all_strategies_run_and_fill_at_open(self):
        bars = make_bars(400, seed=7)
        for strat in S.STRATEGIES:
            res = engine.run(bars, strat, "TW2330")
            if not res.fills.empty:
                np.testing.assert_allclose(res.fills["price"].to_numpy(),
                                           bars.loc[res.fills["date"], "open"].to_numpy())
            self.assertEqual(len(res.held), len(bars))


# ---------------------------------------------------------------- 策略邊界
class TestStrategyRules(unittest.TestCase):
    def test_kdj_sells_on_death_cross_not_next_day(self):
        bars = make_bars(800, seed=11)
        res = engine.run(bars, S.Kdj, "TW1")
        trades = engine.trades_from_held(res.held).dropna()
        # 原始碼的 or 條件會讓每筆都只持有 1 天；改成死叉後持有天數應有分布
        self.assertGreater((trades["exit_i"] - trades["entry_i"]).max(), 1)

    def test_flat_prices_do_not_crash_kdj(self):
        # 9 日最高 = 最低時 RSV 分母為 0；原始碼的 zero=None 會讓 backtrader 崩潰
        bars = make_bars(300, seed=1)
        bars.iloc[100:130, :4] = 50.0
        for strat in (S.Kdj, S.KdjMacd):
            res = engine.run(bars, strat, "TW1")
            self.assertEqual(len(res.held), len(bars))

    def test_adx_holds_three_days(self):
        bars = make_bars(1500, seed=5, drift=0.0008)
        res = engine.run(bars, S.AdxMacd, "TW1")
        trades = engine.trades_from_held(res.held).dropna()
        self.assertGreater(len(trades), 0)
        self.assertTrue(((trades["exit_i"] - trades["entry_i"]) == 3).all())

    def test_ten_pct_exit(self):
        # 買進後收盤漲到 112：(112 − 100) / 112 = 10.7% > 10% → 隔日開盤出場
        prices = [100] * 40 + [100, 100, 100, 112, 112, 112]
        bars = constant_bars(prices)
        strat = type("T", (S.MacdTenPct,), {"buy_signal": lambda self: len(self) == 41})
        res = engine.run(bars, strat, "TW1")
        self.assertEqual(res.fills["date"].tolist(), [bars.index[41], bars.index[44]])

    def test_harami_detection(self):
        idx = pd.bdate_range("2020-01-01", periods=6)
        bars = pd.DataFrame({"open": [10, 10, 12, 10.5, 10.5, 10.5], "close": [10, 10, 9, 10.4, 10.5, 10.5],
                             "high": [10, 10, 12.5, 11, 10.6, 10.6], "low": [10, 10, 8.5, 10, 10.4, 10.4],
                             "volume": 1}, index=idx)
        # 第 3 天（index 3）的高低點 11 / 10 都在第 2 天實體 9–12 內 → 收盤訊號，第 4 天開盤買
        res = engine.run(bars, S.Harami, "TW1")
        self.assertEqual(res.fills["date"].tolist()[:1], [idx[4]])


# ---------------------------------------------------------------- 分析與統計
class TestAnalysis(unittest.TestCase):
    def test_position_log_returns_hand_check(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        oo = pd.Series([np.log(1.1), np.log(1.2), np.log(0.9), np.nan], index=idx)
        held = pd.Series([False, True, True, False], index=idx)
        lr = analysis.position_log_returns(held, oo, 0.001, 0.004)
        expected = [0, np.log(1.2) + np.log(0.999), np.log(0.9), np.log(0.996)]
        np.testing.assert_allclose(lr.to_numpy(), expected)

    def test_shuffle_preserves_segments(self):
        rng = np.random.default_rng(0)
        h = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1], dtype=np.int8)
        for _ in range(20):
            s = analysis.shuffle_segments(h, rng)
            self.assertEqual(s.sum(), h.sum())
            self.assertEqual(sorted(analysis.runs(s)), sorted(analysis.runs(h)))
            self.assertEqual(s[0], h[0])

    def test_holm(self):
        out = analysis.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        # 排序 0.01×3=0.03 拒絕；0.03×2=0.06 > 0.05 不拒絕；之後都不拒絕
        self.assertEqual({k: v[1] for k, v in out.items()}, {"a": True, "c": False, "b": False})
        self.assertAlmostEqual(out["a"][0], 0.03)
        self.assertAlmostEqual(out["c"][0], 0.06)
        self.assertAlmostEqual(out["b"][0], 0.06)

    def test_newey_west_lag0_equals_iid_se(self):
        x = np.random.default_rng(1).normal(0.1, 1, 500)
        mu, se, t = analysis.newey_west_t(x, 0)
        self.assertAlmostEqual(se, x.std(ddof=0) / np.sqrt(len(x)))
        self.assertAlmostEqual(t, mu / se)

    def test_t1_null_of_random_rule_is_centered(self):
        bars = make_bars(1200, seed=2)
        rng = np.random.default_rng(9)
        held = pd.Series(np.repeat(rng.integers(0, 2, 60), 20).astype(bool), index=bars.index)
        res = analysis.t1_test(held, bars, bars.index[0], bars.index[-1], 0.0004, 0.0034, draws=300)
        self.assertGreater(res["p"], 0.01)
        self.assertAlmostEqual(res["upper95"], res["delta_sharpe"] + 1.645 * res["sd0"])

    def test_t2_null_zero_excess_on_constant_drift(self):
        idx = pd.bdate_range("2015-01-01", periods=800)
        opens = 100 * np.exp(0.001 * np.arange(800))
        bars = pd.DataFrame({"open": opens, "close": opens, "high": opens, "low": opens, "volume": 1}, index=idx)
        trades = pd.DataFrame({"entry_date": idx[[300, 500]], "exit_date": idx[[310, 530]],
                               "entry_i": [300, 500], "exit_i": [310, 530]})
        table = analysis.t2_trade_null(trades, bars, 0.0004, 0.0034, np.random.default_rng(0))
        np.testing.assert_allclose(table["excess"].to_numpy(), 0, atol=1e-12)
        expected_net = np.exp(0.001 * 10) * (1 - 0.0004) * (1 - 0.0034) - 1
        self.assertAlmostEqual(table["net"].iloc[0], expected_net)

    def test_t2_excludes_trades_across_bad_days(self):
        idx = pd.bdate_range("2017-01-02", periods=400)
        opens = np.full(400, 100.0)
        opens[200:] = 150.0  # 第 200 天跳 +50%：不可能的跳動
        bars = pd.DataFrame({"open": opens, "close": opens, "high": opens, "low": opens, "volume": 1}, index=idx)
        trades = pd.DataFrame({"entry_date": idx[[190, 300]], "exit_date": idx[[210, 310]],
                               "entry_i": [190, 300], "exit_i": [210, 310]})
        table = analysis.t2_trade_null(trades, bars, 0.0, 0.0, np.random.default_rng(0), draws=200)
        self.assertEqual(table["spans_bad_day"].tolist(), [True, False])
        self.assertTrue(np.isnan(table["excess"].iloc[0]))
        # 第二筆的隨機進場日不能跨過第 200 天，所以對照報酬全是 0
        self.assertAlmostEqual(table["null_net"].iloc[1], 0.0)

    def test_verdict_branches(self):
        v = analysis.verdict
        self.assertEqual(v(True, 0.3, 0.5, 0.1, 0.2, 0.7, True, 0.2), "(+)")
        self.assertEqual(v(True, 0.3, 0.5, -0.1, 0.2, 0.7, True, 0.2), "(?)")  # 一軌為負
        self.assertEqual(v(True, 0.3, 0.5, 0.1, 0.2, 0.5, True, 0.2), "(?)")  # 正年度比例不足
        self.assertEqual(v(True, 0.3, 0.5, 0.1, 0.2, 0.7, False, 0.2), "(?)")  # 集中度
        self.assertEqual(v(False, -0.1, 0.15, 0.1, 0.2, 0.7, True, 0.2), "(−)")
        self.assertEqual(v(False, 0.1, 0.4, 0.1, 0.2, 0.7, True, 0.2), "(?)")

    def test_commission(self):
        c = broker.commission_for("TW2330")
        self.assertAlmostEqual(c.getcommission(1000, 10), 20)  # 低於最低手續費
        self.assertAlmostEqual(c.getcommission(-1000, 1000), 1000 * 1000 * (0.000399 + 0.003))
        self.assertAlmostEqual(broker.commission_for("TW2330", 0.0).getcommission(1000, 10), 0)


class TestFactors(unittest.TestCase):
    def test_daily_rank_ic_matches_scipy(self):
        rng = np.random.default_rng(4)
        dates = pd.bdate_range("2021-01-01", periods=5)
        idx = pd.MultiIndex.from_product([dates, [f"TW{i}" for i in range(40)]], names=["datetime", "instrument"])
        x = pd.Series(rng.normal(size=len(idx)), index=idx)
        y = pd.Series(0.3 * x.to_numpy() + rng.normal(size=len(idx)), index=idx)
        x.iloc[3] = np.nan
        y.iloc[50] = np.nan
        x.iloc[45] = x.iloc[44]  # 同分
        ic = factors.daily_rank_ic(x, y)
        for d in dates:
            xd, yd = x.loc[d], y.loc[d]
            ok = xd.notna() & yd.notna()
            self.assertAlmostEqual(ic.loc[d, "ic"], spearmanr(xd[ok], yd[ok])[0])
            self.assertEqual(ic.loc[d, "k"], ok.sum())

    def test_min_cross_section(self):
        idx = pd.MultiIndex.from_product([pd.bdate_range("2021-01-01", periods=2), range(10)])
        s = pd.Series(np.arange(20.0), index=idx)
        self.assertTrue(factors.daily_rank_ic(s, s).empty)

    def test_ic_stats_width_standardized(self):
        ic = pd.DataFrame({"ic": [0.1, 0.1, 0.1, 0.1], "k": [101, 101, 26, 26]})
        st = factors.ic_stats(ic, lag=0)
        self.assertAlmostEqual(st["ic_w"], 0.1)
        self.assertAlmostEqual(st["z_mean"], (2 * 0.1 * 10 + 2 * 0.1 * 5) / 4)

    def test_clean_label(self):
        idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2014-01-02"), "A"), (pd.Timestamp("2014-01-02"), "B"),
                                         (pd.Timestamp("2016-01-04"), "A"), (pd.Timestamp("2016-01-04"), "B")])
        df = pd.DataFrame({"LABEL0": [0.074, 0.08, 0.104, -0.2]}, index=idx)
        out, n = factors.clean_label(df)
        self.assertEqual(n, 2)
        self.assertEqual(out["LABEL0"].isna().tolist(), [False, True, False, True])

    def test_judge_b1(self):
        screen = pd.DataFrame({
            "disc_ic_w": [0.03, -0.03, 0.01, 0.03], "disc_t": [4.0, -5.0, 1.0, 3.5],
            "disc_icir": 0, "test_ic_w": [0.03, -0.01, 0.05, -0.04], "test_t": [4.0, -0.5, 9.0, -4.0],
            "test_icir": 0, "test_days": 1400,
        }, index=["good", "weak", "not_candidate", "flipped"])
        j = factors.judge_b1(screen)
        self.assertEqual(j["status"].to_dict(),
                         {"good": "通過", "weak": "未通過", "not_candidate": "非候選", "flipped": "方向翻轉"})


class TestConfig(unittest.TestCase):
    def test_costs(self):
        self.assertAlmostEqual(config.buy_cost_rate(), 0.000399)
        self.assertAlmostEqual(config.sell_cost_rate("TW2330"), 0.003399)
        self.assertAlmostEqual(config.sell_cost_rate("TW0050"), 0.001399)

    def test_pre_registration_present(self):
        self.assertIn("判準", config.PRE_REGISTRATION)


if __name__ == "__main__":
    unittest.main()


class TestWFHedge(unittest.TestCase):
    """C 部分（wf_hedge.py）：訓練期截止、避險比率排程、beta、避險後報酬、判定。"""

    @classmethod
    def setUpClass(cls):
        from tw_research import wf_hedge

        cls.wf = wf_hedge

    def test_train_end_embargo(self):
        self.assertEqual(self.wf.train_end(2013), "2012-12-20")
        cfg = self.wf.handler_config(2018)["kwargs"]
        self.assertEqual(cfg["fit_end_time"], "2017-12-20")
        self.assertEqual(cfg["end_time"], "2018-12-31")

    def make_returns(self, n=400):
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2012-01-02", periods=n)
        bench = pd.Series(rng.normal(0, 0.01, n), index=idx)
        stocks = pd.DataFrame({"A": 3.0 * bench, "B": 0.5 * bench + rng.normal(0, 1e-9, n),
                               "C": bench.where(idx >= idx[-50])}, index=idx)
        return stocks, bench

    def test_stock_betas_clip_and_min_obs(self):
        stocks, bench = self.make_returns()
        b = self.wf.stock_betas(stocks, bench, stocks.index[-1], ["A", "B", "C", "Z"])
        self.assertAlmostEqual(b["A"], 2.0)  # 3.0 截斷到 2
        self.assertAlmostEqual(b["B"], 0.5, places=5)
        self.assertEqual(b["C"], 1.0)  # 只有 50 筆有效觀測 < 120
        self.assertEqual(b["Z"], 1.0)  # 沒有資料

    def test_hedge_ratio_schedule(self):
        stocks, bench = self.make_returns()
        dates = stocks.index[stocks.index >= "2013-03-01"][:45]  # 3 月、4 月、5 月初
        w = pd.Series({"A": 0.4, "B": 0.6})
        weights = {d: w for d in dates}
        H = self.wf.hedge_ratios(dates, weights, stocks, bench, segment_starts=[dates[0]])
        ups = self.wf.update_dates(dates)
        self.assertEqual(list(ups.month), [3, 4, 5])
        expected = 0.4 * 2.0 + 0.6 * 0.5
        self.assertEqual(H.iloc[0], 0.0)  # 分段第一天還是現金
        self.assertAlmostEqual(H.loc[dates[1]], expected, places=5)
        self.assertAlmostEqual(H.loc[ups[1]], expected, places=5)  # 更新日當天仍用上個月的比率

    def test_hedged_returns_hand_check(self):
        idx = pd.bdate_range("2014-01-01", periods=3)
        net = pd.Series([0.01, -0.02, 0.005], index=idx)
        bench = pd.Series([0.02, -0.01, 0.0], index=idx)
        H = pd.Series([0.0, 0.8, 0.8], index=idx)
        h = self.wf.hedged_returns(net, bench, H, cost_annual=0.0252)
        np.testing.assert_allclose(h.to_numpy(), [0.01, -0.02 + 0.008 - 0.8 * 0.0001, 0.005 - 0.8 * 0.0001])

    def test_verdict_c_branches(self):
        base = {"p": 0.01, "E": 0.05, "n_positive_years": 6, "half1": 0.03, "half2": 0.04,
                "concentration_E": 0.02, "upper95": 0.12, "realized_beta": 0.05}
        self.assertEqual(self.wf.verdict_c(base, 0.03), ("(+)", []))
        self.assertEqual(self.wf.verdict_c(base, -0.01)[1], ["優勢可能來自大小型股的相對表現，而非選股"])
        self.assertEqual(self.wf.verdict_c(dict(base, n_positive_years=4), 0.03)[0], "(?)")
        self.assertEqual(self.wf.verdict_c(dict(base, half1=-0.01), 0.03)[0], "(?)")
        self.assertEqual(self.wf.verdict_c(dict(base, p=0.2, E=-0.1, upper95=0.01), 0.03)[0], "(−)")
        self.assertEqual(self.wf.verdict_c(dict(base, realized_beta=0.3), 0.03)[1], ["避險失效"])

    def test_pre_registration_c_present(self):
        self.assertIn("判準", config.PRE_REGISTRATION_C)
        self.assertIn("walk-forward", config.PRE_REGISTRATION_C)
