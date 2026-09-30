"""N-24/N-14 退化检验 (合成数据, 无网络) —— 这是 2026-09-30 那次 bug 的永久防线。

背景: `calc_mom_mix` / `calc_wls_score` 最初只吃 `price_mat`, 丢掉了生产的
**纳指 QQQ-CNY 混合口径** (`mom20_mat[纳指] = 0.3×513100 + 0.7×QQQ-CNY`)。
后果: 连权重退化成 `20:1.00` 都不等于基线 —— 一整轮 1771 格横扫因此作废。
本文件把"退化必须等价"钉成测试, 使这类错误在**跑批量之前**就暴露。

覆盖:
  1) `calc_mom_mix` 传 blend 时, 20:1.00 必须等于 `mom20_mat`(含混合);
  2) **不传 blend 时 20:1.00 ≠ 含混合的 mom20** —— 显式记录这个陷阱本身;
  3) 混合权重 0 / 1 的两个边界;
  4) 权重和不为 1 必须报错; 空 spec 必须报错;
  5) 单窗口纯量: `5:1.00` 必须等于 `calc_mom(pm,5)`;
  6) `calc_wls_score` 的 blend 边界: w=0 时等于不混合, w=1 时等于另一序列单独打分。
"""
import numpy as np
import pytest

from strategy.signal import calc_mom, calc_mom_mix, calc_wls_score

N = 120


def _pm(seed=3):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.013, (3, N)), axis=1))


PM = _pm()
QQQ = 100.0 * np.exp(np.cumsum(np.random.default_rng(9).normal(0.0008, 0.010, N)))
ROW, W = 1, 0.7


def _blend(mat, row, other, w):
    """按生产口径把 mat[row] 的动量与 other 的动量混合 (参考实现)"""
    out = mat.copy()
    m_other = calc_mom(other[None, :], 20)[0]
    out[row] = (1 - w) * out[row] + w * m_other
    return out


class TestMomMixDegenerate:
    def test_blend_20_only_equals_plain_mom20(self):
        """★ 核心防线: 传 blend 时 20:1.00 必须逐位等于(含混合的)mom20"""
        got = calc_mom_mix(PM, "5:0.00,10:0.00,15:0.00,20:1.00", blend=(ROW, QQQ, W))
        want = _blend(calc_mom(PM, 20), ROW, QQQ, W)
        np.testing.assert_allclose(got, want, atol=1e-12, equal_nan=True)

    def test_without_blend_is_different_trap(self):
        """不传 blend 会丢掉 QQQ 混合 —— 显式钉住这个陷阱 (不是"正确", 是要能被发现)"""
        got = calc_mom_mix(PM, "5:0.00,10:0.00,15:0.00,20:1.00")
        want = _blend(calc_mom(PM, 20), ROW, QQQ, W)
        assert not np.allclose(got, want, atol=1e-9, equal_nan=True), \
            "不传 blend 竟然等于混合口径? 说明混合没生效, 检查 _blend 参考实现"

    def test_blend_weight_zero_equals_no_blend(self):
        got = calc_mom_mix(PM, "5:0.20,10:0.20,15:0.20,20:0.40", blend=(ROW, QQQ, 0.0))
        want = calc_mom_mix(PM, "5:0.20,10:0.20,15:0.20,20:0.40")
        np.testing.assert_allclose(got, want, atol=1e-12, equal_nan=True)

    def test_blend_weight_one_uses_other_series(self):
        got = calc_mom_mix(PM, "20:1.00", blend=(ROW, QQQ, 1.0))
        ref = np.tile(calc_mom(QQQ[None, :], 20)[0], (3, 1))
        np.testing.assert_allclose(got[ROW], ref[ROW], atol=1e-12, equal_nan=True)


class TestMomMixBasics:
    def test_single_window_equals_calc_mom(self):
        for w in (5, 10, 15, 20):
            got = calc_mom_mix(PM, "%d:1.00" % w)
            np.testing.assert_allclose(got, calc_mom(PM, w), atol=1e-12, equal_nan=True)

    def test_weighted_sum_is_linear(self):
        got = calc_mom_mix(PM, "5:0.25,10:0.25,15:0.25,20:0.25")
        want = 0.25 * sum(calc_mom(PM, w) for w in (5, 10, 15, 20))
        np.testing.assert_allclose(got, want, atol=1e-12, equal_nan=True)

    def test_bad_weight_sum_raises(self):
        with pytest.raises(ValueError):
            calc_mom_mix(PM, "5:0.50,20:0.40")
        with pytest.raises(ValueError):
            calc_mom_mix(PM, "")

    def test_nan_when_nonzero_component_missing(self):
        """非零权重分量预热不足 → 该日 NaN (mom20 需 20 天, mom5 只需 5 天)"""
        got = calc_mom_mix(PM, "5:0.50,20:0.50")
        assert np.all(np.isnan(got[:, :19])), "20 日分量未就绪时不该有值"
        assert np.isfinite(got[0, 20])


class TestWlsBlend:
    def test_blend_zero_equals_plain(self):
        a = calc_wls_score(PM, 25, 1, 250.0, blend=(ROW, QQQ, 0.0))
        b = calc_wls_score(PM, 25, 1, 250.0)
        np.testing.assert_allclose(a, b, atol=1e-12, equal_nan=True)

    def test_blend_one_uses_other_series_score(self):
        a = calc_wls_score(PM, 25, 1, 250.0, blend=(ROW, QQQ, 1.0))
        b = calc_wls_score(QQQ[None, :], 25, 1, 250.0)[0]
        np.testing.assert_allclose(a[ROW], b, atol=1e-12, equal_nan=True)
