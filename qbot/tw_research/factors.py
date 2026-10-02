"""B 部分：qlib Alpha158 單因子篩選（alphalens 式 IC 分析）與多因子模型。

B1 單因子：158 個 Alpha158 因子，每日橫斷面 Spearman rank IC，寬度標準化 z = IC·sqrt(K−1)。
   探索期挑候選、測試期只看一次，Holm 校正。
B2 多因子：repo 內 benchmark 的 Linear / LightGBM / XGBoost（Alpha158）與 LightGBM（Alpha360），
   用 qlib 的 TopkDropoutStrategy 回測，淨超額報酬對同一股票池的等權組合。
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import analysis, config

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def init_qlib(kernels: int = 4):
    import qlib

    qlib.init(
        provider_uri=str(config.QLIB_DIR),
        region="tw",
        kernels=kernels,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": config.OUTPUT_DIR.joinpath("mlruns").as_uri(),
                "default_exp_name": "tw_research",
            },
        },
    )


# ---------------------------------------------------------------- IC 計算
def daily_rank_ic(x: pd.Series, y: pd.Series, min_n: int = config.MIN_CROSS_SECTION) -> pd.DataFrame:
    """每日橫斷面 Spearman 相關（平均名次處理同分）。index 第一層為 datetime。回傳 ic、k 兩欄。"""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    g = d.groupby(level=0)
    k = g["x"].transform("size").astype(float)
    mid = (k + 1) / 2
    rx = g["x"].rank() - mid
    ry = g["y"].rank() - mid
    parts = pd.DataFrame({"xy": rx * ry, "xx": rx * rx, "yy": ry * ry}).groupby(d.index.get_level_values(0))
    s = parts.sum()
    out = pd.DataFrame(
        {"ic": s["xy"] / np.sqrt(s["xx"] * s["yy"]), "k": g.size().astype(float)}
    )
    out = out[out["k"] >= min_n]
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def ic_stats(ic: pd.DataFrame, lag: int = config.NW_LAG_DAILY) -> Dict[str, float]:
    """寬度標準化：z = IC·sqrt(K−1)；效果量用 (K−1) 加權的平均 IC。"""
    if ic.empty:
        return {"n_days": 0, "ic_mean": np.nan, "ic_w": np.nan, "z_mean": np.nan, "t": np.nan, "icir": np.nan}
    z = ic["ic"] * np.sqrt(ic["k"] - 1)
    mu, se, t = analysis.newey_west_t(z.to_numpy(), lag)
    w = ic["k"] - 1
    return {
        "n_days": int(len(ic)),
        "ic_mean": float(ic["ic"].mean()),
        "ic_w": float((ic["ic"] * w).sum() / w.sum()),
        "z_mean": mu,
        "t": t,
        "icir": float(ic["ic"].mean() / ic["ic"].std(ddof=1)) if len(ic) > 1 else np.nan,
    }


def segment(df: pd.DataFrame, period: Tuple[str, str]) -> pd.DataFrame:
    dates = df.index.get_level_values(0)
    return df[(dates >= pd.Timestamp(period[0])) & (dates <= pd.Timestamp(period[1]))]


# ---------------------------------------------------------------- B1
def load_alpha158_raw(start: str = config.DISCOVERY[0], end: str = config.DATA_END) -> pd.DataFrame:
    """Alpha158 原始特徵 + label（不做任何前處理；Spearman 對單調轉換不變）。"""
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP

    h = Alpha158(
        instruments=config.UNIVERSE,
        start_time=start,
        end_time=end,
        fit_start_time=config.TRAIN[0],
        fit_end_time=config.TRAIN[1],
        infer_processors=[],
        learn_processors=[],
    )
    df = h.fetch(col_set="__all", data_key=DataHandlerLP.DK_R)
    df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]
    return df.astype("float32")


def clean_label(df: pd.DataFrame, label_col: str = "LABEL0") -> Tuple[pd.DataFrame, int]:
    """label 超過制度漲跌幅 + 0.5pp 的不可能報酬（資料錯誤、未還原的減資）設為缺值。"""
    dates = df.index.get_level_values(0)
    cap = np.where(dates >= pd.Timestamp(config.LIMIT_CHANGE_DATE), config.LIMIT_NEW, config.LIMIT_OLD)
    bad = df[label_col].abs().to_numpy() > cap + config.JUMP_TOLERANCE
    out = df.copy()
    out.loc[bad, label_col] = np.nan
    return out, int(bad.sum())


def single_factor_screen(df: pd.DataFrame, label_col: str = "LABEL0") -> pd.DataFrame:
    """每個因子：探索期統計、候選判定、測試期統計（Holm 在 judge_b1 裡做）。"""
    rows = []
    y = df[label_col]
    for name in [c for c in df.columns if c != label_col]:
        ic = daily_rank_ic(df[name], y)
        disc = ic_stats(segment(ic, config.DISCOVERY))
        test = ic_stats(segment(ic, config.TEST))
        rows.append(
            {
                "factor": name,
                "disc_ic_w": disc["ic_w"],
                "disc_t": disc["t"],
                "disc_icir": disc["icir"],
                "test_ic_w": test["ic_w"],
                "test_t": test["t"],
                "test_icir": test["icir"],
                "test_days": test["n_days"],
            }
        )
    return pd.DataFrame(rows).set_index("factor")


def judge_b1(screen: pd.DataFrame) -> pd.DataFrame:
    s = screen.copy()
    s["candidate"] = s["disc_t"].abs() >= config.B1_DISCOVERY_T
    s["sign"] = np.sign(s["disc_t"])
    s["test_signed_t"] = s["sign"] * s["test_t"]
    s["test_p"] = [
        analysis.one_sided_p(t, n) if c else np.nan
        for t, n, c in zip(s["test_signed_t"], s["test_days"], s["candidate"])
    ]
    cand = s[s["candidate"]]
    adj = analysis.holm(cand["test_p"].to_dict())
    s["holm_p"] = pd.Series({k: v[0] for k, v in adj.items()})
    s["holm_reject"] = pd.Series({k: v[1] for k, v in adj.items()}, dtype=bool).reindex(s.index, fill_value=False)
    s["material"] = s["sign"] * s["test_ic_w"] >= config.B1_MATERIALITY
    s["passed"] = s["candidate"] & s["holm_reject"] & s["material"]
    flipped_p = [
        analysis.one_sided_p(-t, n) if c else np.nan
        for t, n, c in zip(s["test_signed_t"], s["test_days"], s["candidate"])
    ]
    s["flipped"] = s["candidate"] & (pd.Series(flipped_p, index=s.index) < 0.025)
    s["status"] = np.where(
        ~s["candidate"], "非候選",
        np.where(s["passed"], "通過", np.where(s["flipped"], "方向翻轉", "未通過")),
    )
    return s


# ---------------------------------------------------------------- B2
def _alpha158(processors: str) -> dict:
    kwargs = {
        "start_time": config.TRAIN[0],
        "end_time": config.DATA_END,
        "fit_start_time": config.TRAIN[0],
        "fit_end_time": config.TRAIN[1],
        "instruments": config.UNIVERSE,
    }
    if processors == "linear":
        kwargs["infer_processors"] = [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ]
        kwargs["learn_processors"] = [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ]
    return {"class": "Alpha158", "module_path": "qlib.contrib.data.handler", "kwargs": kwargs}


def _alpha360() -> dict:
    return {
        "class": "Alpha360",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": config.TRAIN[0],
            "end_time": config.DATA_END,
            "fit_start_time": config.TRAIN[0],
            "fit_end_time": config.TRAIN[1],
            "instruments": config.UNIVERSE,
            "infer_processors": [],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ],
        },
    }


_LGB_158 = dict(loss="mse", colsample_bytree=0.8879, learning_rate=0.2, subsample=0.8789,
                lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=12)
_LGB_360 = dict(_LGB_158, learning_rate=0.0421)

# 參數照抄 pytrader/strategies/benchmarks 底下各 yaml（只把 num_threads 改成本機核心數、加上 seed）
MODELS = {
    "M1_Linear_A158": ({"class": "LinearModel", "module_path": "qlib.contrib.model.linear",
                        "kwargs": {"estimator": "ols"}}, lambda: _alpha158("linear")),
    "M2_LGB_A158": ({"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
                     "kwargs": dict(_LGB_158, seed=config.MODEL_SEED)}, lambda: _alpha158("default")),
    "M3_XGB_A158": ({"class": "XGBModel", "module_path": "qlib.contrib.model.xgboost",
                     "kwargs": dict(eval_metric="rmse", colsample_bytree=0.8879, eta=0.0421, max_depth=8,
                                    n_estimators=647, subsample=0.8789, nthread=12, seed=config.MODEL_SEED)},
                    lambda: _alpha158("default")),
    "M4_LGB_A360": ({"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
                     "kwargs": dict(_LGB_360, seed=config.MODEL_SEED)}, _alpha360),
}


def make_dataset(handler: dict):
    from qlib.utils import init_instance_by_config

    return init_instance_by_config(
        {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler,
                "segments": {"train": config.TRAIN, "valid": config.VALID, "test": config.TEST},
            },
        }
    )


def fit_predict(name: str) -> Tuple[pd.Series, pd.Series]:
    """訓練並回傳測試期的 (預測分數, 原始 label)。"""
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    model_cfg, handler_fn = MODELS[name]
    dataset = make_dataset(handler_fn())
    model = init_instance_by_config(model_cfg)
    with R.start(experiment_name="tw_research", recorder_name=name):
        model.fit(dataset)
        pred = model.predict(dataset, segment="test")
    label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
    label = label.iloc[:, 0] if isinstance(label, pd.DataFrame) else label
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    return pred.rename("score"), label.rename("label")


def topk_backtest(pred: pd.Series, fee_multiplier: float = 1.0, n_drop: int = config.B2_N_DROP,
                  start: str = config.TEST[0], end: str = config.BACKTEST_END) -> pd.DataFrame:
    """qlib TopkDropoutStrategy 回測，回傳每日 gross / cost / net / bench / turnover。"""
    from qlib.backtest import backtest

    fee = config.FEE_RATE * config.FEE_DISCOUNT * fee_multiplier
    strategy = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": pred.to_frame("score"), "topk": config.B2_TOPK, "n_drop": n_drop},
    }
    executor = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    portfolio, _ = backtest(
        start_time=start,
        end_time=end,
        strategy=strategy,
        executor=executor,
        benchmark=config.BENCHMARK,
        account=config.B2_ACCOUNT,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": config.LIMIT_THRESHOLD_NEW,
            "deal_price": "close",
            "open_cost": fee,
            "close_cost": fee + config.SELL_TAX_STOCK,
            "min_cost": config.MIN_FEE * (fee_multiplier > 0),
        },
    )
    report = portfolio["1day"][0]
    out = pd.DataFrame(
        {
            "gross": report["return"],
            "cost": report["cost"],
            "bench": report["bench"],
            "turnover": report["turnover"],
        }
    )
    out["net"] = out["gross"] - out["cost"]
    return out


def universe_ew_returns(start: str, end: str) -> pd.Series:
    """同一股票池（top200，依當日成分）等權、每日再平衡的收盤對收盤報酬。"""
    from qlib.data import D

    inst = D.instruments(config.UNIVERSE)
    df = D.features(inst, ["$close / Ref($close, 1) - 1"], start_time=start, end_time=end)
    df.columns = ["ret"]
    return df["ret"].groupby(level="datetime").mean().rename("universe_ew")


def judge_b2(ic_p_holm: Tuple[float, bool], excess: pd.Series, lag: int = config.NW_LAG_DAILY) -> Dict[str, object]:
    """訊號顯著（Holm）+ 年化淨超額 ≥ 門檻且顯著 + 測試期前後兩半都為正 → (+)；上界 < 門檻 → (−)。"""
    mu, se, t = analysis.newey_west_t(excess.to_numpy(), lag)
    ann, ann_se = mu * analysis.TRADING_DAYS, se * analysis.TRADING_DAYS
    p = analysis.one_sided_p(t, len(excess))
    halves = [float(excess.loc[a:b].mean() * analysis.TRADING_DAYS) for a, b in config.TEST_HALVES]
    upper = ann + 1.645 * ann_se
    if ic_p_holm[1] and ann >= config.B2_MATERIALITY and p < 0.05 and all(h > 0 for h in halves):
        v = "(+)"
    elif upper < config.B2_MATERIALITY:
        v = "(−)"
    else:
        v = "(?)"
    return {"ann_excess": ann, "ann_se": ann_se, "t": t, "p": p, "upper95": upper,
            "half1": halves[0], "half2": halves[1], "verdict": v}
