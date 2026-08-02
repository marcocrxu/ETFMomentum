"""持仓状态管理"""
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class StateManager:
    position: Optional[str] = None
    entry_date: Optional[date] = None
    entry_price: Optional[float] = None
    pending_leader: Optional[str] = None
    leader_streak: int = 0

    def update(self, action):
        """根据操作更新状态"""
        if action.type == "CASH":
            self.position = None
            self.entry_date = None
            self.entry_price = 0.0
            self.pending_leader = None
            self.leader_streak = 0
        elif action.type == "BUY":
            self.position = action.target
            self.entry_date = action.meta.get("date", date.today())
            self.entry_price = action.meta.get("price") or 0.0
            self.pending_leader = None
            self.leader_streak = 0

    def track_leader(self, leader_name: str):
        """追踪新冠军连胜"""
        if self.pending_leader == leader_name:
            self.leader_streak += 1
        else:
            self.pending_leader = leader_name
            self.leader_streak = 1

    def is_confirmed(self, days: int = 2) -> bool:
        return self.leader_streak >= days

    def hold_days(self, today: date) -> int:
        if self.entry_date is None:
            return 0
        return (today - self.entry_date).days

    def pnl(self, current_price: float) -> float:
        """当前盈亏 (%)"""
        if self.entry_price is None or self.entry_price <= 0 or current_price is None or current_price <= 0:
            return 0.0
        return (current_price / self.entry_price - 1) * 100

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "entry_price": self.entry_price,
            "pending_leader": self.pending_leader,
            "leader_streak": self.leader_streak,
        }

    @classmethod
    def from_dict(cls, d: dict):
        entry_date = None
        if d.get("entry_date"):
            entry_date = date.fromisoformat(d["entry_date"])
        return cls(
            position=d.get("position"),
            entry_date=entry_date,
            entry_price=d.get("entry_price", 0.0),
            pending_leader=d.get("pending_leader"),
            leader_streak=d.get("leader_streak", 0),
        )
