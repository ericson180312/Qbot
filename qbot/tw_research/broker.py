"""台股現股的手續費與證交稅（backtrader CommissionInfo）。"""

import backtrader as bt

from . import config


class TWCommission(bt.CommInfoBase):
    """手續費 = max(成交金額 × 費率 × 折數, 最低手續費)，賣出另加證交稅。"""

    params = (
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
        ("fee_rate", config.FEE_RATE * config.FEE_DISCOUNT),
        ("min_fee", config.MIN_FEE),
        ("sell_tax", config.SELL_TAX_STOCK),
    )

    def _getcommission(self, size, price, pseudoexec):
        value = abs(size) * price
        if value == 0:
            return 0.0
        fee = max(value * self.p.fee_rate, self.p.min_fee)
        tax = value * self.p.sell_tax if size < 0 else 0.0
        return fee + tax


def commission_for(symbol: str, fee_multiplier: float = 1.0) -> TWCommission:
    tax = config.SELL_TAX_ETF if symbol in config.ETF_SYMBOLS else config.SELL_TAX_STOCK
    return TWCommission(
        fee_rate=config.FEE_RATE * config.FEE_DISCOUNT * fee_multiplier,
        min_fee=config.MIN_FEE * (fee_multiplier > 0),
        sell_tax=tax,
    )
