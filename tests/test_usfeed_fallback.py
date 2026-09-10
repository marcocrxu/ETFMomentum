"""QQQ-CNY 兜底数据源测试 (2026-09 Yahoo 被墙后的真实事故对应)

覆盖:
1. Yahoo 失败时自动切 新浪/东财 + 中国货币网 兜底
2. 中间价汇率按重叠期价差重定基, 接缝处无假跳变
3. 只追加新日期, 历史行不变; 原子写回
4. 新鲜度检查 qqq_cny_freshness
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import usfeed


@pytest.fixture
def tmp_csv(tmp_path, monkeypatch):
    """造一份 5 行的旧 CSV (Yahoo 现货口径)"""
    csv = tmp_path / "qqq_usdcny.csv"
    old = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-01", "2026-09-02"]),
        "QQQ": [700.0, 710.0],
        "USDCNY=X": [6.72, 6.72],
    })
    old["date"] = old["date"].dt.date
    old.to_csv(csv, index=False)
    monkeypatch.setattr(usfeed, "CSV", str(csv))
    return str(csv)


def _new_qqq():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-04"]),
        "QQQ": [710.0, 720.0, 730.0],
    })
    return df


def _new_fx_parity():
    # 中间价口径: 比 Yahoo 现货(6.72) 高 ~0.9%
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]),
        "USDCNY=X": [6.78, 6.78, 6.79, 6.80],
    })
    return df


def test_fallback_appends_and_rebases(tmp_csv, monkeypatch):
    monkeypatch.setattr(usfeed, "_fetch", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("yahoo down")))
    monkeypatch.setattr(usfeed, "_fetch_qqq_sina", _new_qqq)
    monkeypatch.setattr(usfeed, "_fetch_usdcny_cm", lambda beg: _new_fx_parity())

    df = usfeed.refresh_qqq_cny()

    # 追加了 2 行新数据
    assert len(df) == 4
    assert str(df["date"].iloc[-1]) == "2026-09-04"

    # 历史行不变
    assert df["QQQ"].iloc[0] == 700.0
    assert abs(df["USDCNY=X"].iloc[0] - 6.72) < 1e-9

    # 重定基: 新汇率应缩放到旧现货尺度 (比值≈6.72/6.78), 接缝处无跳变
    ratio = 6.72 / 6.78
    assert abs(df["USDCNY=X"].iloc[2] - 6.79 * ratio) < 1e-6
    seam_jump = abs(df["USDCNY=X"].iloc[2] / df["USDCNY=X"].iloc[1] - 1)
    assert seam_jump < 0.002, f"接缝跳变 {seam_jump:.4%} 过大, 会污染动量"

    # qqq_cny 列齐全 (align_to_cn 直接可用)
    assert "qqq_cny" in df.columns
    assert abs(df["qqq_cny"].iloc[-1] - 730.0 * 6.80 * ratio) < 1e-4


def test_fallback_raises_when_no_new_data(tmp_csv, monkeypatch):
    monkeypatch.setattr(usfeed, "_fetch", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("yahoo down")))
    stale = pd.DataFrame({"date": pd.to_datetime(["2026-09-01"]), "QQQ": [700.0]})
    monkeypatch.setattr(usfeed, "_fetch_qqq_sina", lambda: stale)
    monkeypatch.setattr(usfeed, "_fetch_usdcny_cm",
                        lambda beg: pd.DataFrame({"date": pd.to_datetime(["2026-09-01"]),
                                                  "USDCNY=X": [6.78]}))
    with pytest.raises(RuntimeError, match="无新数据"):
        usfeed.refresh_qqq_cny()


def test_freshness(tmp_csv):
    last, expect, fresh = usfeed.qqq_cny_freshness(day=pd.Timestamp("2026-09-03"))
    # CSV 最新 09-02; 09-03 决策期望美国 09-02 数据 → 新鲜
    assert str(last.date()) == "2026-09-02"
    assert str(expect.date()) == "2026-09-02"
    assert fresh

    _, expect2, fresh2 = usfeed.qqq_cny_freshness(day=pd.Timestamp("2026-09-04"))
    assert str(expect2.date()) == "2026-09-03"
    assert not fresh2
