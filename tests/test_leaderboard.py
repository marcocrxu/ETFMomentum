"""打榜模块单测 — 网格结构与排序逻辑, 无网络"""
import pytest

from web.leaderboard import (ETF_GRID, YANYAN_GRID, METRICS,
                             run_leaderboard, _row)


def test_grid_sizes_bounded():
    assert len(ETF_GRID) == 16
    assert len(YANYAN_GRID) == 54
    for label, p in ETF_GRID + YANYAN_GRID:
        assert label and p


def test_row_shape():
    r = _row("etf", "x", {"total_ret": 1.0, "mdd": -2.0, "calmar": 0.5,
                          "ann_ret": 3.0, "sharpe": 0.4, "trade_count": 9})
    assert r["strategy"] == "etf" and r["params"] == "x"
    assert r["trade_count"] == 9


def test_metric_in_metrics():
    for m in METRICS:
        assert m in ("calmar", "total_ret", "ann_ret", "sharpe", "mdd")


def test_check_token(monkeypatch):
    import web.leaderboard as lb
    monkeypatch.setattr(lb, "get_leaderboard_token", lambda: "abc123")
    ok, err = lb.check_token("abc123")
    assert ok and err == ""
    ok2, err2 = lb.check_token("wrong")
    assert not ok2 and "token" in err2


def test_submit_queue_cap(monkeypatch, tmp_path):
    import web.leaderboard as lb
    monkeypatch.setattr(lb, "QUEUE_FILE", str(tmp_path / "q.json"))
    monkeypatch.setattr(lb, "_ensure_worker", lambda: None)   # 不真正启动 dsh
    for i in range(3):
        item, err = lb.submit_custom("测试策略描述第%d个, 长度足够" % i)
        assert item is not None and err is None
    item, err = lb.submit_custom("第四个测试策略描述, 足够长")
    assert item is None and "队列已满" in err


def test_submit_rejects_short_desc(monkeypatch, tmp_path):
    import web.leaderboard as lb
    monkeypatch.setattr(lb, "QUEUE_FILE", str(tmp_path / "q.json"))
    monkeypatch.setattr(lb, "_ensure_worker", lambda: None)
    item, err = lb.submit_custom("太短")
    assert item is None and "太短" in err
