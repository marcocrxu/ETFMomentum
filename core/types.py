"""核心类型定义"""
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List, Dict


class ActionType:
    BUY = "BUY"
    SELL = "SELL"
    CASH = "CASH"
    HOLD = "HOLD"


@dataclass
class Action:
    type: str              # BUY / SELL / CASH / HOLD
    target: Optional[str]  # ETF名称, None=空仓
    reason: str
    meta: dict = field(default_factory=dict)


@dataclass
class Trade:
    entry_date: date
    exit_date: date
    asset: str
    days: int
    ret_pct: float
    reason: str


@dataclass
class BacktestResult:
    total_ret_pct: float
    ann_ret_pct: float
    max_dd_pct: float
    calmar: float
    trades: List[Trade]
    yearly: Dict[int, float]
    nav_series: List[tuple]
    holdings: List[tuple]
    recovery_days: Optional[int] = None


@dataclass
class MomentumData:
    """某个时间点所有标的的动量快照"""
    date: date
    values: Dict[str, float]   # {name: momentum%}
    prices: Dict[str, float]   # {name: price}
