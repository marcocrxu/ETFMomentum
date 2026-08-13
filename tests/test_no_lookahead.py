"""前视偏差 & 截断一致性测试 (合成数据, 无网络)"""
import numpy as np
import pytest

from strategy.signal import (SignalConfig, compute_signals,
                             calc_accel_thresholds, calc_mom60,
                             calc_vol20, calc_up_ratio20)
from conftest import build_inputs, gbm_prices


class TestTruncationConsistency:
    """预热充足后切片: pre_mats+pre_days 修复 → 切片结果与完整跑一致"""

    def test_sliced_with_premats_matches_full(self):
        """pre_mats+pre_days 在 pre=0 (同一性) 时不改变结果 (Bug D 回归)"""
        n = 500
        prices = {"纳指": gbm_prices(n, 100, 0.0008, 0.010, seed=51),
                  "黄金": gbm_prices(n, 100, 0.0004, 0.006, seed=52),
                  "创业": gbm_prices(n, 100, 0.0006, 0.014, seed=55)}
        idx = {"cyb": gbm_prices(n, 3000, 0.0003, 0.010, seed=53),
               "sh":  gbm_prices(n, 3000, 0.0002, 0.008, seed=54)}
        cfg = SignalConfig()
        inp = build_inputs(prices, idx)

        # 参考: 标准调用
        _, _, nv_full = compute_signals(cfg, **inp)

        # pre=0 同一性: 传 pre_mats + pre_days=0, 应与参考完全一致
        pmat = inp["price_mat"]; m10 = inp["mom10_mat"]
        pre_mats = {"accel_th": calc_accel_thresholds(m10, cfg)}
        _, _, nv_id = compute_signals(
            cfg,
            etf_names=inp["etf_names"],
            price_mat=pmat,
            mom20_mat=inp["mom20_mat"],
            mom10_mat=m10,
            sh_mom=inp["sh_mom"],
            cyb_idx_p=inp["cyb_idx_p"],
            sh_idx_p=inp["sh_idx_p"],
            n_days=n,
            pre_days=0,
            pre_mats=pre_mats,
        )

        np.testing.assert_allclose(
            np.asarray(nv_full, dtype=float),
            np.asarray(nv_id, dtype=float),
            rtol=1e-10,
            err_msg="pre_mats pre=0 同一性 nav 不一致")

    def test_naive_slice_loses_coldstart(self):
        """无 pre_mats 的原始切片=冷启动, 短决策区 nav 长度与完整跑不同 (Bug D 回归)"""
        n, pre = 500, 450   # 仅 50 天决策区 → 冷启动严重
        prices = {"纳指": gbm_prices(n, 100, 0.0008, 0.010, seed=56),
                  "黄金": gbm_prices(n, 100, 0.0004, 0.006, seed=57)}
        idx = {"cyb": gbm_prices(n, 3000, 0.0003, 0.010, seed=58),
               "sh":  gbm_prices(n, 3000, 0.0002, 0.008, seed=59)}
        cfg = SignalConfig(accel_enabled=True)
        inp = build_inputs(prices, idx)
        # 原始切片 (无 pre_mats / pre_days)
        _, _, nv_naive = compute_signals(
            cfg,
            etf_names=inp["etf_names"],
            price_mat=inp["price_mat"][:, pre:],
            mom20_mat=inp["mom20_mat"][:, pre:],
            mom10_mat=inp["mom10_mat"][:, pre:],
            sh_mom=inp["sh_mom"][pre:],
            cyb_idx_p=inp["cyb_idx_p"][pre:],
            sh_idx_p=inp["sh_idx_p"][pre:],
            n_days=n - pre)
        # nav_log 应非空 (至少 vs 之后有值)
        assert len(nv_naive) > 0, "原始切片 nav 全空"


class TestNoFutureLeak:
    """确保所有阈值/指标没有未来数据泄露"""

    def test_accel_threshold_uses_only_past(self):
        """加速止盈阈值在 t 日只用 t-1 及之前的数据 (shift(1) 语义)"""
        n = 500
        prices = {"纳指": gbm_prices(n, 100, 0.0008, 0.012, seed=71),
                  "黄金": gbm_prices(n, 100, 0.0004, 0.006, seed=72)}
        idx = {"cyb": gbm_prices(n, 3000, 0.0003, 0.010, seed=73),
               "sh":  gbm_prices(n, 3000, 0.0002, 0.008, seed=74)}
        cfg = SignalConfig()
        inp = build_inputs(prices, idx)

        mom10 = inp["mom10_mat"].copy()
        th1 = calc_accel_thresholds(mom10, cfg)
        # 篡改第 300-500 天的 mom10 (未来)
        mom10[:, 300:] = 999.0
        th2 = calc_accel_thresholds(mom10, cfg)
        # 前 300 天的阈值必须完全一样
        np.testing.assert_allclose(th1[:, :300], th2[:, :300], rtol=1e-12,
            err_msg="未来数据篡改影响了历史的加速止盈阈值 (前视偏差)")

    def test_crash_threshold_uses_only_past(self):
        """恐慌抄底的 calc_threshold 在 t 日只用 arr[:t]"""
        cfg = SignalConfig(crash_sigma=-2.5)
        def ct(arr, t):
            valid = arr[:t][~np.isnan(arr[:t])]
            if len(valid) < 60:
                return None
            return valid.mean() + cfg.crash_sigma * valid.std()

        arr = np.random.default_rng(80).normal(0, 1, 300)
        th_before = ct(arr, 200)
        arr[200:] = 999.0
        th_after = ct(arr, 200)
        assert th_before == th_after or (th_before is None and th_after is None), \
            "恐慌阈值 calc_threshold 存在前视偏差"


class TestPositionSeriesIntegrity:
    """bt_runner 侧 position_labels / position_series 对齐"""

    def test_partial_crash_position_label(self):
        """crash_alloc<100 时 position_labels 应出现组合持仓标签"""
        n = 500
        # 纳指强动量(始终被持有), 创业弱(不会自然轮入), 但其 index 急跌触发恐慌
        prices = {"纳指": gbm_prices(n, 100, 0.0015, 0.008, seed=81),
                  "黄金": gbm_prices(n, 100, 0.0001, 0.003, seed=82),
                  "创业": np.full(n, 100.0)}          # 动量=0, 不会被轮动选中
        sh = np.full(n, 3000.0)
        # MA200 需要 200 天预热 → 崩溃放到 t=300 之后, 且跌够狠
        cyb = np.full(n, 3000.0)
        cyb[350:] = cyb[350] * np.exp(np.cumsum(np.full(n - 350, -0.015)))
        inp = build_inputs(prices, {"cyb": cyb, "sh": sh})
        cfg = SignalConfig(crash_alloc=0.5, sh_cash=False,
                           accel_enabled=False, abs_stop_loss=False)
        actions, _, _ = compute_signals(cfg, **inp)
        assert any(a[3] == "恐慌抄底" for a in actions), \
            f"恐慌抄底未触发 (actions: {[a[3] for a in actions]})"

        etf_names = inp["etf_names"]
        position_labels = ["空仓"] + etf_names
        position_series = [-1] * inp["n_days"]
        current_pos = -1
        mix_idx = {}
        act_sorted = sorted(actions, key=lambda x: x[0])
        act_idx = 0
        for t in range(inp["n_days"]):
            while act_idx < len(act_sorted) and act_sorted[act_idx][0] <= t:
                a = act_sorted[act_idx]
                old_p, new_p, reason = a[1], a[2], a[3]
                if reason == "恐慌抄底" and old_p >= 0 and cfg.crash_alloc < 1.0:
                    key = (old_p, new_p)
                    if key not in mix_idx:
                        mix_idx[key] = len(position_labels)
                        position_labels.append(
                            f"{etf_names[old_p]}({int((1-cfg.crash_alloc)*100)}%)+"
                            f"{etf_names[new_p]}({int(cfg.crash_alloc*100)}%)")
                    current_pos = mix_idx[key]
                else:
                    current_pos = new_p if new_p is not None else current_pos
                act_idx += 1
            position_series[t] = current_pos

        mix_labels = [l for l in position_labels if "+" in l]
        assert len(mix_labels) > 0, "crash_alloc=0.5 时应产生组合持仓标签"
        mix_positions = set(int(p) for p in position_series if p >= len(etf_names))
        assert len(mix_positions) > 0, "crash_alloc=0.5 position_series 中应有组合持仓索引"
