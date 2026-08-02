"""策略引擎 — 组合 EntryRules + SwitchRules + RiskRules（带自动校准）"""
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import date
import numpy as np

from core.types import Action, MomentumData
from core.state import StateManager
from strategy.momentum import MomentumMethod, PctMomentum
from strategy.rules import EntryRules, SwitchRules, RiskRules


@dataclass
class Strategy:
    name: str
    universe: List[str]
    momentum: MomentumMethod = None
    entry: EntryRules = None
    switch: SwitchRules = None
    risk: RiskRules = None
    commission: float = 0.0001

    # 自动校准参数
    auto_calibrate: bool = True
    accel_sigma: float = 3.0       # 加速止盈 σ 倍数
    crash_sigma: float = -2.0      # 恐慌抄底 σ 倍数
    crash_lock_days: int = 15       # 抄底锁仓天数

    def __post_init__(self):
        if self.momentum is None:
            self.momentum = PctMomentum(20)
        if self.entry is None:
            self.entry = EntryRules()
        if self.switch is None:
            self.switch = SwitchRules()
        if self.risk is None:
            self.risk = RiskRules()
        self.state = StateManager()

    def set_data(self, prices: dict, short_mom: dict, ma200: dict,
                 crash_mom: dict = None):
        """
        prices: {name: np.array} — ETF收盘价
        short_mom: {name: np.array} — 短动量(10日)
        ma200: {name: np.array} — 200日均线
        crash_mom: {name: np.array} — 抄底检测用动量
        """
        self._prices = prices
        self._short_mom = short_mom
        self._ma200 = ma200

        # 自动校准加速止盈阈值
        if self.auto_calibrate and self.risk.accel_exit:
            self.risk.accel_thresholds = {}
            for name in self.universe:
                if name in short_mom:
                    valid = short_mom[name][~np.isnan(short_mom[name])]
                    if len(valid) >= 252:
                        self.risk.accel_thresholds[name] = round(
                            valid.mean() + self.accel_sigma * valid.std())

        # 注入数据
        crash_data = crash_mom or {"上证指数": short_mom.get("上证指数", np.array([]))}
        self.risk.set_data(short_mom, ma200, crash_data)
        self.risk.crash_lock_days = self.crash_lock_days

    def signal(self, mom_data: MomentumData, today: date, today_idx: int = -1) -> Action:
        sh_mom = mom_data.values.get("上证指数", 0)
        etf_moms = {k: v for k, v in mom_data.values.items() if k != "上证指数"}
        prices = mom_data.prices

        if not etf_moms:
            return Action("HOLD", None, "无有效ETF数据")

        leader = max(etf_moms, key=etf_moms.get)
        leader_mom = etf_moms[leader]
        leader_price = prices.get(leader, 0)
        self.state.track_leader(leader)

        # 规则1: 上证空仓
        if self.entry.sh_cash and sh_mom > max(etf_moms.values()):
            return Action("CASH", None, "上证动量最高(%.2f%%)" % sh_mom)

        # 风险: 恐慌抄底（最高优先级，独立于if/elif链）
        if self.risk.crash_buy:
            crash_action = self.risk.check_crash(mom_data, today_idx, 9999)
            if crash_action:
                return crash_action

        # 规则2: 空仓入场
        if self.state.position is None:
            act = self.entry.enter_from_cash(
                leader, leader_mom, leader_price, self.state, today)
            if act:
                return act
            return Action("CASH", None,
                          "待确认 %s(%d/%d天)" % (leader, self.state.leader_streak,
                                                   self.entry.cash_confirm))

        # 规则3: 持仓切换
        if self.state.position != leader:
            current_price = prices.get(self.state.position, 0)
            pnl = self.state.pnl(current_price)
            gap = leader_mom - etf_moms.get(self.state.position, 0)

            act = self.switch.check_switch(
                leader, leader_mom, leader_price,
                self.state, pnl, gap, today)
            if act:
                return act

        # 风险: 加速止盈（独立判断，不管是否换冠军）
        if self.risk.accel_exit and self.state.position:
            accel_action = self.risk.check_accel(
                self.state.position, mom_data, self.state, today_idx)
            if accel_action:
                return accel_action

        pos_mom = etf_moms.get(self.state.position, 0)
        return Action("HOLD", self.state.position,
                      "%s仍为冠军(%.2f%%)" % (self.state.position, pos_mom))
