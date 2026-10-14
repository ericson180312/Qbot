"""台股回測與因子挖掘：把 Qbot 的擇時策略與 qlib 因子流程搬到台股。

模組分層：
    config      成本、期間、門檻與預先承諾的判準（唯一的數值來源）
    data        讀 tw_collector 產生的日線、漲跌停判斷、動態股票池
    indicators  pandas 版技術指標（給沒有 backtrader 指標可用的策略）
    broker      台股手續費 / 證交稅
    strategies  移植自 qbot/strategies、pytrader/strategies、docs/tutorials_code 的擇時策略
    engine      backtrader 回測：t 日收盤出訊號、t+1 日開盤成交
    analysis    績效、隨機對照、Holm、Newey-West
    factors     qlib Alpha158 單因子篩選與多因子模型
"""
