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
