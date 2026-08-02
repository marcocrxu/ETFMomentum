"""下单执行 + 净值管理"""
from datetime import date
from typing import List
from .types import Action, Trade


class OrderExecutor:
    def __init__(self, commission: float = 0.0001):
        self.commission = commission
        self.nav = 1.0
        self._trades: List[Trade] = []
        self._last_buy_nav = 1.0
        self._position = None
        self._entry_date = None
        self._entry_nav = 1.0

    def execute(self, action: Action, price_map: dict, trade_date: date) -> float:
        """
        执行操作，更新净值。
        price_map: {name: close_price}
        返回新净值
        """
        if action.type == "CASH" and self._position is not None:
            self.nav *= (1 - self.commission)
            ret = (self.nav / self._entry_nav - 1) * 100
            self._trades.append(Trade(
                entry_date=self._entry_date, exit_date=trade_date,
                asset=self._position,
                days=(trade_date - self._entry_date).days,
                ret_pct=round(ret, 2), reason=action.reason,
            ))
            self._position = None

        elif action.type == "BUY" and action.target:
            if self._position is not None and self._position != action.target:
                self.nav *= (1 - self.commission)
                ret = (self.nav / self._entry_nav - 1) * 100
                self._trades.append(Trade(
                    entry_date=self._entry_date, exit_date=trade_date,
                    asset=self._position,
                    days=(trade_date - self._entry_date).days,
                    ret_pct=round(ret, 2), reason=action.reason,
                ))
            self.nav *= (1 - self.commission)
            self._position = action.target
            self._entry_date = trade_date
            self._entry_nav = self.nav

        return self.nav

    def close_position(self, trade_date: date, price_map: dict):
        """期末平仓"""
        if self._position is not None:
            self.nav *= (1 - self.commission)
            ret = (self.nav / self._entry_nav - 1) * 100
            self._trades.append(Trade(
                entry_date=self._entry_date, exit_date=trade_date,
                asset=self._position,
                days=(trade_date - self._entry_date).days,
                ret_pct=round(ret, 2), reason="期末平仓",
            ))
            self._position = None

    @property
    def trades(self) -> List[Trade]:
        return self._trades

    @property
    def position(self) -> str:
        return self._position
