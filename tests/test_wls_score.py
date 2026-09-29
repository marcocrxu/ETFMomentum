"""N-14 WLS 排名分回归测试 (合成数据, 无网络)

`calc_wls_score` 复刻同行 SCORE_WLS: score = (时间加权回归斜率 / 加权均价 × ann) / vol20,
smooth>1 时对位移 k=1..smooth 的同样回归取均值。本测试只钉住可解析验证的性质。
"""
import numpy as np
import pytest

from strategy.signal import SignalConfig, calc_wls_score, calc_vol20


def _linear(n, a=10.0, b=0.05):
    """严格线性价格: p[t] = a + b·t (回归无残差, 斜率恒为 b)"""
    return a + b * np.arange(n, dtype=float)


def test_default_off():
    cfg = SignalConfig()
    assert cfg.score_wls is False
    assert cfg.wls_window == 25 and cfg.wls_smooth == 1 and cfg.wls_ann == 250.0


def test_closed_form_linear_series():
    """线性序列上 WLS 斜率 = b (精确), 故 raw = b/wmy×ann, score = raw/vol。"""
    n, a, b, w = 60, 10.0, 0.05, 20
    pm = _linear(n, a, b)[None, :]
    vol = np.full((1, n), 0.02)                      # 固定分母, 隔离分子
    sc = calc_wls_score(pm, window=w, smooth=1, ann=250.0, vol_mat=vol)
    xs = np.arange(w, dtype=float)
    wts = xs + 1.0
    wsum = wts.sum()
    wmx = float((wts * xs).sum() / wsum)
    t = n - 1
    seg = pm[0, t - w + 1:t + 1]
    wmy = float((wts * seg).sum() / wsum)
    expect = (b / wmy) * 250.0 / 0.02
    assert np.isclose(sc[0, t], expect, rtol=1e-10)


def test_sign_and_monotonicity_in_slope():
    """斜率为正 → 分数为正; 斜率越大分数越大; 下降趋势为负。"""
    n, w = 60, 20
    vol = np.full((1, n), 0.02)
    up1 = calc_wls_score(_linear(n, 10, 0.02)[None, :], w, 1, 250.0, vol)[0, -1]
    up2 = calc_wls_score(_linear(n, 10, 0.05)[None, :], w, 1, 250.0, vol)[0, -1]
    dn = calc_wls_score(_linear(n, 10, -0.02)[None, :], w, 1, 250.0, vol)[0, -1]
    assert up1 > 0 and dn < 0
    assert up2 > up1


def test_window_param_effective():
    """window 参数必须真的改变窗口长度 (对照最后 window 根手工复算)。"""
    n, a, b = 80, 8.0, 0.03
    pm = _linear(n, a, b)[None, :]
    vol = np.full((1, n), 0.02)
    s20 = calc_wls_score(pm, 20, 1, 250.0, vol)[0, -1]
    s40 = calc_wls_score(pm, 40, 1, 250.0, vol)[0, -1]
    # 线性序列上分子只依赖窗口内的加权均价, 两个窗口必然不同
    assert not np.isclose(s20, s40)


def test_smooth_averages_shifted_windows():
    """smooth=2 的分数应等于位移 0 与位移 1 两个窗口分数的均值。"""
    n, w, a, b = 80, 20, 9.0, 0.04
    pm = _linear(n, a, b)[None, :]
    vol = np.full((1, n), 0.02)
    s1 = calc_wls_score(pm, w, 1, 250.0, vol)          # 位移 0
    s2 = calc_wls_score(pm, w, 2, 250.0, vol)          # mean(位移0, 位移1)
    t = n - 1
    # 单独算位移 1 (即 t-1 结束的窗口)
    s1b = calc_wls_score(pm[:, :t], w, 1, 250.0, vol[:, :t])[0, t - 1]
    assert np.isclose(s2[0, t], (s1[0, t] + s1b) / 2.0, rtol=1e-10)


def test_nan_and_zero_vol_are_nan():
    """数据不足 / vol=0 的位置必须是 NaN, 不能参与排名。"""
    n, w = 40, 20
    pm = _linear(n, 10, 0.05)[None, :]
    vol = np.full((1, n), 0.02)
    vol[0, -1] = 0.0
    sc = calc_wls_score(pm, w, 1, 250.0, vol)
    assert np.all(np.isnan(sc[0, :w - 1])), "预热段应为 NaN"
    assert np.isnan(sc[0, -1]), "vol=0 应为 NaN"
    assert np.isfinite(sc[0, w])


def test_smoothing_insufficient_history_is_nan():
    """smooth>1 时, 位移窗口需要额外历史, 起点应比 smooth=1 更晚有值。"""
    n, w = 40, 20
    pm = _linear(n, 10, 0.05)[None, :]
    vol = np.full((1, n), 0.02)
    s1 = calc_wls_score(pm, w, 1, 250.0, vol)[0]
    s3 = calc_wls_score(pm, w, 3, 250.0, vol)[0]
    assert not np.isnan(s1[w - 1])
    assert np.isnan(s3[w - 1])
    assert not np.isnan(s3[w + 1])
