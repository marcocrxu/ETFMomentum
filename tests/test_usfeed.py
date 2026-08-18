"""usfeed 单元测试 — 全部用已提交的 data/qqq_usdcny.csv, 无需网络"""
import numpy as np
import pandas as pd
import pytest

from data.usfeed import load_qqq_cny, align_to_cn, CSV


def test_csv_committed_and_valid():
    import os
    assert os.path.exists(CSV), "data/qqq_usdcny.csv 未提交"
    df = load_qqq_cny()
    assert len(df) > 2500, f"数据量异常: {len(df)}"
    assert df["qqq_cny"].min() > 0, "qqq_cny 应为正数"
    assert df["date"].min() <= pd.Timestamp("2014-01-01"), "数据起点不足以支撑 2015 长窗预热"


def test_align_no_lookahead():
    """对齐语义: CN 日期 D 取美国日期 ≤ D-1 的最后收盘"""
    df = load_qqq_cny()
    s = df.set_index("date")["qqq_cny"]
    cn = [pd.Timestamp("2026-08-14"), pd.Timestamp("2026-08-17"), pd.Timestamp("2026-08-18")]
    out = align_to_cn(cn, df)
    for d, v in zip(cn, out):
        lim = d - pd.Timedelta(days=1)
        eligible = s[s.index <= lim]
        assert not eligible.empty
        expect = eligible.iloc[-1]
        assert v == pytest.approx(float(expect)), \
            f"{d.date()} 应对齐到 {eligible.index[-1].date()} 的收盘, 实际 {v}"


def test_align_monotonic_gap_free():
    """对齐序列与 CN 交易日历同长, 缺失仅在最前段"""
    df = load_qqq_cny()
    cn = pd.date_range("2025-01-01", "2026-08-17", freq="B")
    out = align_to_cn(cn, df)
    assert len(out) == len(cn)
    assert np.isnan(out).sum() == 0, "2025 年起不应有缺失"
