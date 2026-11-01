"""MA200 空洞防御回归测试 (2026-10-15 事故)

事故: 数据缓存混入 000001.SH 一条周六脏行(2026-09-12) → 全量日历多出一天,
其余标的当天全 NaN → 创业板指 rolling(200) 的 MA200 从该日起全部变 NaN →
`cyb_idx_p[tf] < ma200_cyb[tf]` 与 NaN 比较恒为 False → 恐慌抄底被静默否决,
10-14 该出的抄底信号被吞掉 (空洞在触发日之前, 此后 200 个交易日的 MA200 全废)。
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import build_inputs, gbm_prices
from strategy.signal import SignalConfig, compute_signals, _ma_safe, _rolling_mean_valid

TRIGGER_DAY = 301   # 无空洞时恐慌抄底的触发日 (由合成数据决定)


def _panic_inputs(hole_at=None):
    """纳指慢牛(会被持有) + 创业板指暴跌(应触发恐慌抄底); hole_at 处插入数据空洞"""
    n = 400
    prices = {
        "纳指": gbm_prices(n, 100, 0.0008, 0.008, seed=11),
        "黄金": gbm_prices(n, 100, 0.0002, 0.006, seed=12),
        "创业": np.concatenate([np.full(300, 100.0),
                                100 * np.linspace(1, 0.70, n - 300)]),
    }
    cyb = np.full(n, 3000.0)
    cyb[300:] = 3000 * np.linspace(1, 0.75, n - 300)   # 急跌 → mom10 跌破 μ-2.5σ
    if hole_at is not None:
        cyb = cyb.copy()
        cyb[hole_at] = np.nan
    return build_inputs(prices, {"cyb": cyb, "sh": np.full(n, 3000.0)})


def _cfg():
    return SignalConfig(accel_enabled=False, sh_cash=False)


def test_rolling_mean_valid_matches_pandas_without_holes():
    """无空洞时必须与 pandas rolling 逐位相同(含前导 NaN)"""
    a = gbm_prices(500, 100, 0.0005, 0.01, seed=7)
    got = _rolling_mean_valid(a, 200)
    ref = np.asarray(pd.Series(a).rolling(200).mean().values)
    np.testing.assert_array_equal(got, ref)


def test_ma_safe_backfills_hole_and_warns():
    """有空洞时回填为'最后200个有效报价'均值并告警"""
    a = gbm_prices(500, 100, 0.0005, 0.01, seed=7)
    a = a.copy()
    a[300] = np.nan
    with pytest.warns(RuntimeWarning, match="MA200 有 .* NaN 空洞"):
        ma = _ma_safe(a, 200, None, "创业板指")
    assert np.isfinite(ma[499]), "空洞之后 MA200 不应是 NaN"
    valid = a[~np.isnan(a)][-200:]
    assert ma[499] == pytest.approx(valid.mean(), rel=1e-12)
    assert np.isnan(ma[100]), "历史不足 200 天处仍应保持 NaN"


def test_panic_buy_triggers_without_hole():
    """对照: 无空洞时恐慌抄底正常触发"""
    actions, _, _ = compute_signals(_cfg(), **_panic_inputs())
    hits = [a[0] for a in actions if a[3] == "恐慌抄底"]
    assert hits == [TRIGGER_DAY], f"无空洞应触发恐慌抄底: {hits}"


def test_panic_buy_not_silently_killed_by_ma_hole():
    """回归: 触发日之前的指数空洞曾让 MA200 全 NaN, 恐慌抄底被静默吞掉

    修复前此用例得到空列表(整个决策区一次都不触发)。"""
    inp = _panic_inputs(hole_at=TRIGGER_DAY - 11)   # 空洞早于触发日, 此后 MA200 全 NaN
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actions, _, _ = compute_signals(_cfg(), **inp)
    hits = [a[0] for a in actions if a[3] == "恐慌抄底"]
    assert hits == [TRIGGER_DAY], f"MA200 空洞吞掉了恐慌抄底: {hits}"
    assert any("MA200" in str(x.message) for x in w), "应输出数据告警"
