"""入场 / 切换 / 风险规则"""
import numpy as np
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Dict, List

from core.types import Action, MomentumData
from core.state import StateManager


# ──────────────────────────────────────────
@dataclass
class EntryRules:
    sh_cash: bool = True          # 上证最高 → 空仓
    cash_confirm: int = 2         # 空仓确认天数
    cash_skip_gt: float = 3.0     # 空仓>X%即买

    def enter_from_cash(self, leader: str, leader_mom: float,
                        leader_price: float,
                        state: StateManager, today: date) -> Optional[Action]:
        if leader_mom >= self.cash_skip_gt:
            return Action("BUY", leader,
                          ">%.0f%%即买(%.2f%%)" % (self.cash_skip_gt, leader_mom),
                          {"date": today, "price": leader_price})
        if state.is_confirmed(self.cash_confirm):
            return Action("BUY", leader,
                          "确认买入(%.2f%%)" % leader_mom,
                          {"date": today, "price": leader_price})
        return None


# ──────────────────────────────────────────
@dataclass
class SwitchRules:
    min_hold: int = 10
    min_gap: float = 1.0
    stop_loss: float = -5.0
    exception_gap: float = 3.0
    confirm_days: int = 2

    def check_switch(self, leader: str, leader_mom: float,
                     leader_price: float,
                     state: StateManager, pnl: float, gap: float,
                     today: date) -> Optional[Action]:
        hold = state.hold_days(today)
        confirmed = state.is_confirmed(self.confirm_days)

        if pnl <= self.stop_loss:
            return Action("BUY", leader,
                          "止损: 亏%.1f%%" % pnl,
                          {"date": today, "price": leader_price})
        if gap >= self.exception_gap and confirmed:
            return Action("BUY", leader,
                          "动量差%.1f%%>%.0f%%" % (gap, self.exception_gap),
                          {"date": today, "price": leader_price})
        if hold >= self.min_hold and gap >= self.min_gap and confirmed:
            return Action("BUY", leader,
                          "持有%d天+差%.1f%%" % (hold, gap),
                          {"date": today, "price": leader_price})
        return None


# ──────────────────────────────────────────
@dataclass
class RiskRules:
    # 加速止盈
    accel_exit: bool = False
    accel_window: int = 10
    accel_thresholds: Dict[str, float] = field(default_factory=dict)
    # {name: 10日动量阈值%}
    # 如果没提供，用均值+3σ 自动算

    # 恐慌抄底
    crash_buy: bool = False
    crash_window: int = 10
    crash_threshold: float = -2.0     # z-score 阈值 (默认-2σ)
    crash_ma200: bool = True
    crash_lock_days: int = 15         # 抄底锁仓天数

    # 内部状态
    _crash_entry_idx: int = -1        # 上次抄底入场日
    _short_mom_arrays: dict = field(default_factory=dict)  # 短动量数据(外部注入)
    _ma200_arrays: dict = field(default_factory=dict)
    _crash_mom_arrays: dict = field(default_factory=dict)   # 用于抄底检测的动量

    def set_data(self, short_mom: dict, ma200: dict, crash_mom: dict = None):
        """注入短动量和MA200数据 {name: np.array}"""
        self._short_mom_arrays = short_mom
        self._ma200_arrays = ma200
        self._crash_mom_arrays = crash_mom or short_mom

    def check_accel(self, pos: str, mom_data: MomentumData,
                    state: StateManager, today_idx: int) -> Optional[Action]:
        """加速止盈: 持仓的短动量超过阈值 → 卖出"""
        if not self.accel_exit or pos not in self._short_mom_arrays:
            return None

        short_mom = self._short_mom_arrays[pos]
        if today_idx >= len(short_mom) or np.isnan(short_mom[today_idx]):
            return None

        threshold = self.accel_thresholds.get(pos)
        if threshold is None:
            # 自动计算: 均值+3σ
            valid = short_mom[~np.isnan(short_mom)]
            if len(valid) < 252:
                return None
            threshold = valid.mean() + 3 * valid.std()

        if short_mom[today_idx] > threshold:
            return Action("CASH", None,
                          "加速止盈: 10日动量%.1f%%>%.0f%%" % (short_mom[today_idx], threshold))
        return None

    def check_crash(self, mom_data: MomentumData,
                    today_idx: int, total_days: int) -> Optional[Action]:
        """恐慌抄底: 上证或创业短动量 < -2σ 且 < MA200 → 买入创业板"""
        if not self.crash_buy or not self._crash_mom_arrays:
            return None
        if self._crash_entry_idx >= 0 and today_idx - self._crash_entry_idx < self.crash_lock_days:
            return None

        for name in ["上证指数", "创业"]:
            if name not in self._crash_mom_arrays:
                continue
            mom_arr = self._crash_mom_arrays[name]
            if today_idx >= len(mom_arr) or np.isnan(mom_arr[today_idx]):
                continue

            valid = mom_arr[:today_idx][~np.isnan(mom_arr[:today_idx])]
            if len(valid) < 252:
                continue
            threshold = valid.mean() + self.crash_threshold * valid.std()

            if mom_arr[today_idx] < threshold:
                # MA200检查: 价格 < 200MA
                if self.crash_ma200 and name in self._ma200_arrays:
                    ma200_arr = self._ma200_arrays[name]
                    if today_idx < len(ma200_arr):
                        price_now = mom_data.prices.get(name, 0)
                        if not np.isnan(ma200_arr[today_idx]) and price_now >= ma200_arr[today_idx]:
                            continue  # 未跌破MA200, 不触发

                self._crash_entry_idx = today_idx
                return Action("BUY", "创业",
                              "恐慌抄底: %s 10日动量%.1f%%" % (name, mom_arr[today_idx]),
                              {"date": None, "price": 0.0})
        return None
