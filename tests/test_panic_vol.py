"""N-7 波动缩放急跌阈值 (panic_vol_on) 回归测试 (合成数据, 无网络)

同行 V12-C 的口径: thr = clip(K × VOL, 2%, 10%), VOL 取该标的**上一实际报价日**
的 VOL20 / VOL60 / 二者较大值。本测试覆盖可合成验证的部分:
  1) 开关默认关 (铁律: 新开关默认关, 基线不变);
  2) 未启用时不受 panic_vol_* 参数影响 (防误接线);
  3) 打开后确实改变决策 (防惰性开关);
  4) 阈值方向: K 越大 → 越难触发 → 急跌次数单调不增;
  5) prior 口径必须只用到"上一实际报价日" (无前视/无当日)。
"""
import numpy as np
import pytest

from strategy.signal import SignalConfig, calc_vol20, compute_signals
from conftest import build_inputs, gbm_prices


def _crash_market(seed=21, n=400):
    """平静市场 + 创业在第 300 日单日 -9% (必然触发各种量级的急跌阈值)。"""
    prices = {
        "纳指": gbm_prices(n, 100, 0.0008, 0.010, seed=1),
        "黄金": gbm_prices(n, 100, 0.0004, 0.006, seed=2),
        "创业": gbm_prices(n, 100, 0.0009, 0.014, seed=seed),
    }
    prices["创业"][300] = prices["创业"][299] * 0.91
    index = {
        "cyb": gbm_prices(n, 3000, 0.0003, 0.010, seed=4),
        "sh": gbm_prices(n, 3000, 0.0002, 0.008, seed=5),
    }
    return build_inputs(prices, index)


def test_default_off():
    cfg = SignalConfig()
    assert cfg.panic_vol_on is False
    assert cfg.panic_drop == 0.0
    assert (cfg.panic_vol_lo, cfg.panic_vol_hi) == (0.02, 0.10)
    assert cfg.panic_vol_src == "current20"


def test_calc_vol20_window_param():
    """window 参数必须真的按该窗口计算 (对照独立 numpy 复算, ddof=1)。"""
    px = gbm_prices(120, 100, 0.0005, 0.01, seed=7)
    pm = px[None, :]
    rets = np.diff(px) / px[:-1]
    v20, v60 = calc_vol20(pm, 20), calc_vol20(pm, 60)
    assert np.all(np.isnan(v20[0, :20])), "VOL20 前 20 日应无值"
    assert np.all(np.isnan(v60[0, :60])), "VOL60 前 60 日应无值"
    assert np.isclose(v20[0, 20], rets[:20].std(ddof=1), atol=1e-12)
    assert np.isclose(v60[0, 60], rets[:60].std(ddof=1), atol=1e-12)
    assert not np.isclose(v20[0, 60], v60[0, 60]), "window 参数未生效"


class TestPanicVolSwitch:
    def test_on_changes_decisions(self):
        """开关打开必须真的改变决策 (防惰性开关)。"""
        inp = _crash_market()
        a0, _, log0 = compute_signals(SignalConfig(panic_drop=1.0), **inp)
        a1, _, log1 = compute_signals(
            SignalConfig(panic_vol_on=True, panic_vol_k=1.4,
                         panic_vol_src="prior_max20_60"), **inp)
        assert len(log0) == len(log1)
        assert [x[3] for x in a0] != [x[3] for x in a1] or not np.allclose(log0, log1), \
            "panic_vol_on 未产生任何差异, 参数未接通"

    def test_higher_k_never_more_triggers(self):
        """K 越大阈值越宽 → 触发越少 → 急跌次数单调不增。"""
        inp = _crash_market()
        counts = []
        for k in (0.5, 1.0, 2.0, 4.0):
            cfg = SignalConfig(panic_vol_on=True, panic_vol_k=k,
                               panic_vol_lo=0.0, panic_vol_hi=99.0)
            acts, _, _ = compute_signals(cfg, **inp)
            counts.append(sum(1 for a in acts if a[3] == "单日急跌"))
        assert counts == sorted(counts, reverse=True), \
            "急跌触发次数未随 K 单调不增: %s" % counts

    def test_prior_src_has_no_lookahead(self):
        """prior 口径用"上一实际报价日": 把触发当日价格改成不可能触发, 前一日决策不受影响;
        反之把触发日改为暴涨, prior 阈值本身不应随之变化 (阈值取自前一日)。"""
        inp = _crash_market()
        cfg = SignalConfig(panic_vol_on=True, panic_vol_k=1.4,
                           panic_vol_src="prior_max20_60")
        acts, _, _ = compute_signals(cfg, **inp)
        hits = [a[0] for a in acts if a[3] == "单日急跌"]
        assert hits, "合成急跌未触发, 测试无效"
        t = hits[0]
        # 复制输入, 把触发日之后一天的价格翻倍 —— 不得改变 t 日已经做出的决策
        inp2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in inp.items()}
        inp2["price_mat"][:, t + 1] *= 2.0
        acts2, _, _ = compute_signals(cfg, **inp2)
        hits2 = [a[0] for a in acts2 if a[3] == "单日急跌"]
        assert t in hits2, "未来价格变化影响了过去决策 (前视)"

    def test_disabled_ignores_vol_params(self):
        """未启用时, panic_vol_* 取任何值都不得改变结果。"""
        inp = _crash_market()
        c1 = SignalConfig(panic_drop=2.0, panic_vol_on=False)
        c2 = SignalConfig(panic_drop=2.0, panic_vol_on=False, panic_vol_k=9.9,
                          panic_vol_src="prior_max20_60", panic_vol_lo=0.0)
        a1, _, log1 = compute_signals(c1, **inp)
        a2, _, log2 = compute_signals(c2, **inp)
        np.testing.assert_allclose(log1, log2, atol=0)
        assert [x[3] for x in a1] == [x[3] for x in a2]
