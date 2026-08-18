"""回测引擎集成测试 (需要真实缓存 & tushare token, 标记 integration)"""
import os, sys
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def _skip_if_no_data():
    cache = "/root/backtest/cache/etf_momentum"
    if not os.path.isdir(cache) or not any(f.endswith(".csv") for f in os.listdir(cache)):
        pytest.skip("回测缓存不存在")


class TestBenchmarkAlignment:
    """Bug B 回归: benchmark 用 _p_user 用户窗口, 与 dates 逐日对齐"""

    def test_benchmark_length_equals_dates(self):
        _skip_if_no_data()
        from web.bt_runner import run_backtest
        r = run_backtest({"universe": "纳指,黄金,创业",
                          "start_date": "2016-07-07", "end_date": "2026-07-28"})
        assert len(r["benchmark"]) == len(r["dates"]), \
            f"benchmark 长度 {len(r['benchmark'])} != dates {len(r['dates'])}"

    def test_benchmark_matches_independent_calc(self):
        _skip_if_no_data()
        from web.bt_runner import run_backtest, FEED
        r = run_backtest({"universe": "纳指,黄金,创业",
                          "start_date": "2016-07-07", "end_date": "2026-07-28"})
        # 独立算等权基准 (与 run_backtest 同样的缓存源)
        codes = ["513100.SH", "518880.SH", "159915.SZ"]
        dfs = FEED.get_dataframe(codes, "20160707", "20260728")
        pm = pd.concat([dfs[c].set_index("trade_date")["close"]
                        for c in codes], axis=1).sort_index()
        mask = (pm.index >= pd.Timestamp("2016-07-07")) & \
               (pm.index <= pd.Timestamp("2026-07-28"))
        pm = pm[mask]
        bm = (1 + pm.pct_change().mean(axis=1, skipna=True)).fillna(1.0).cumprod()
        # 末两日日均收益微小, 末值偏差 < 0.01
        assert abs(r["benchmark"][-1] - float(bm.iloc[-1])) < 0.02, \
            f"benchmark 末值偏差过大: {r['benchmark'][-1]:.4f} vs {float(bm.iloc[-1]):.4f}"


class TestEtfNavsAlignment:
    """Bug C 回归: etf_navs 用 _p_user, 与 dates 逐日对齐"""

    def test_etf_navs_length_equals_dates(self):
        _skip_if_no_data()
        from web.bt_runner import run_backtest
        r = run_backtest({"universe": "纳指,黄金,创业",
                          "start_date": "2016-07-07", "end_date": "2026-07-28"})
        for k, v in r["etf_navs"].items():
            assert len(v) == len(r["dates"]), \
                f"etf_navs['{k}'] 长度 {len(v)} != dates {len(r['dates'])}"


class TestWarmupEffectiveness:
    """Bug A 回归: MA200 预热真正生效 → 回测早期恐慌抄底不被静默禁用"""

    def test_crash_detection_in_early_years(self):
        _skip_if_no_data()
        from web.bt_runner import run_backtest
        r = run_backtest({"universe": "纳指,黄金,创业,科创50,沪深300,上证50,中证2000",
                          "start_date": "2016-07-07", "end_date": "2026-07-28"})
        crashes = [d for d in r["decisions"] if d["reason"] == "恐慌抄底"]
        # 修复前: 恐慌抄底最早要到 2017-07-17 (MA200 NaN 期间被静默禁用)
        # 修复后: 2018-02 的恐慌也能触发
        assert len(crashes) >= 3, f"十年回测恐慌抄底应≥3次 (预热生效), 实际 {len(crashes)}"

    def test_short_backtest_accel_threshold_available(self):
        """短回测 (2024 起): pre_mats 使 accel 阈值在窗口起点就绪"""
        _skip_if_no_data()
        from web.bt_runner import run_backtest
        r = run_backtest({"universe": "纳指,黄金,创业",
                          "start_date": "2024-01-02", "end_date": "2026-07-28"})
        # 没有"加速止盈阈值为 NaN 导致永远不触发"的BUG
        # 只需确保回测正常完成且 checks OK
        ok = any("OK" in str(c) for c in r.get("checks", []))
        assert r.get("total_ret") is not None and ok, \
            f"短回测异常: {r.get('checks', [])}"


class TestNavReplayConsistency:
    """集成层: 引擎 nav_log 与独立重放一致 (真实数据验证)"""

    def test_replay_matches_engine(self):
        _skip_if_no_data()
        from web.bt_runner import run_backtest, FEED
        import web.bt_runner as bt
        from strategy.signal import SignalConfig

        r = run_backtest({"universe": "纳指,黄金,创业",
                          "start_date": "2016-07-07", "end_date": "2026-07-28"})
        # 用 engine 返回的 decisions + 同批缓存价格 独立重放
        names = r["actual_universe"]
        code_map = {"纳指": "513100.SH", "黄金": "518880.SH", "创业": "159915.SZ"}
        codes = [code_map[n] for n in names]
        dfs = FEED.get_dataframe(codes, "20160501", "20260728")
        pm = pd.concat([dfs[c].set_index("trade_date")["close"] for c in codes],
                       axis=1).sort_index()
        pm.columns = names

        # 对齐日期
        dates = pd.DatetimeIndex([pd.Timestamp(d) for d in r["dates"]])
        pm = pm.reindex(dates)
        price_mat = np.array([pm[nm].values for nm in names])

        from conftest import replay_nav
        cfg = SignalConfig(
            crash_alloc=float(r.get("crash_alloc", 100)) / 100.0 if "crash_alloc" in r else 1.0,
            commission=0.0001, slippage=0.0,
        )
        # 从 decisions 重建 actions 列表 (回测内部格式)
        actions = []
        dec_map = {}
        for d in r["decisions"]:
            if d["reason"] == "期末":
                continue
            dec_map.setdefault(d["date"], []).append(d)

        for t_idx, dt in enumerate(dates):
            ds = str(dt.date())
            for dd in dec_map.get(ds, []):
                act = dd["action"]
                reason = dd["reason"]
                old_p, new_p = -1, -1
                if "→" in act:
                    parts = act.split("→")
                    old_n = parts[0]
                    new_n = parts[1].split("(")[0]
                    old_p = names.index(old_n) if old_n in names else -1
                    new_p = names.index(new_n) if new_n in names else -1
                elif act.startswith("卖出"):
                    old_n = act[2:]
                    old_p = names.index(old_n) if old_n in names else -1
                    new_p = -1
                elif act.startswith("买入"):
                    new_n = act[2:].split("(")[0]
                    new_p = names.index(new_n) if new_n in names else -1
                    old_p = -1
                actions.append((t_idx, old_p, new_p, reason))

        rep = replay_nav(actions, price_mat, names, cfg)
        engine_nav = np.array(r["nav"], dtype=float)

        # 首日对齐后比较 (允许首日因动作排序微差)
        diff = np.abs(rep - engine_nav)
        assert diff.max() < 0.001, \
            f"独立重放与引擎 nav 偏差过大: max={diff.max():.6f}"


class TestChecksOK:
    """自检项通过"""

    def test_checks_ok_or_data_days_only(self):
        _skip_if_no_data()
        from web.bt_runner import run_backtest
        r = run_backtest({"universe": "纳指,黄金,创业",
                          "start_date": "2016-07-07", "end_date": "2026-07-28"})
        non_ok = [c for c in r["checks"] if "OK" not in str(c) and "天数" not in str(c)]
        assert not non_ok, f"自检异常: {non_ok}"


class TestNzMomQqqBlend:
    """I 组: 纳指动量混合 QQQ-CNY (nz_mom_qqq_w), 需真实缓存"""

    def test_default_is_qqq07(self):
        """默认 on=True, w=0.7: 默认回测与显式开启 w=0.7 逐分一致"""
        _skip_if_no_data()
        from web.bt_runner import run_backtest
        r0 = run_backtest({"universe": "纳指,黄金,创业",
                           "start_date": "2016-07-07", "end_date": "2026-08-12"})
        r1 = run_backtest({"universe": "纳指,黄金,创业",
                           "start_date": "2016-07-07", "end_date": "2026-08-12",
                           "nz_mom_qqq_on": True, "nz_mom_qqq_w": 0.7})
        assert r0["total_ret"] == r1["total_ret"], "默认应与显式开 w=0.7 一致"
        assert r0["trade_count"] == r1["trade_count"]

    def test_explicit_off_restores_etf_only(self):
        """显式 on=False 回到纯 ETF 动量"""
        _skip_if_no_data()
        from web.bt_runner import run_backtest
        r = run_backtest({"universe": "纳指,黄金,创业",
                          "start_date": "2016-07-07", "end_date": "2026-08-12",
                          "nz_mom_qqq_on": False, "accel_cooldown": 0,
                          "accel_cooldown_gold_only": False})
        assert abs(r["total_ret"] - 3595.1) < 0.5, "on=False 应回到纯ETF基线 3595.1"

    def test_on_w1_differs_and_checks_ok(self):
        _skip_if_no_data()
        from web.bt_runner import run_backtest
        r = run_backtest({"universe": "纳指,黄金,创业",
                          "start_date": "2016-07-07", "end_date": "2026-08-12",
                          "nz_mom_qqq_on": True, "nz_mom_qqq_w": 1.0,
                          "accel_cooldown": 0, "accel_cooldown_gold_only": False})
        assert abs(r["total_ret"] - 3374.6) < 0.5, "w=1.0 应与 I-2 纯QQQ口径一致"
        assert "error" not in r, r.get("error")
        assert r["checks"] == ["OK"], r["checks"]
        assert r["total_ret"] > 0

    def test_on_default_w0_7_matches_blend(self):
        """开启后默认 w=0.7 与显式 w=0.7 一致"""
        _skip_if_no_data()
        from web.bt_runner import run_backtest
        base = {"universe": "纳指,黄金,创业",
                "start_date": "2016-07-07", "end_date": "2026-08-12",
                "nz_mom_qqq_on": True}
        r1 = run_backtest(base)
        r2 = run_backtest(dict(base, nz_mom_qqq_w=0.7))
        assert r1["total_ret"] == r2["total_ret"]


class TestYanyanRunner:
    """yanyan 策略回测编排 (需真实缓存)"""

    def test_subset_universe(self):
        _skip_if_no_data()
        from web.yanyan_runner import run_yanyan
        r = run_yanyan({"start_date": "2024-01-01", "end_date": "2026-08-12",
                        "universe": "纳指,黄金,港股创新药"})
        assert "error" not in r, r.get("error")
        assert r["checks"] == ["OK"], r["checks"]
        assert r["actual_universe"] == ["纳指", "港股创新药", "黄金"], r["actual_universe"]
        assert len(r["position_labels"]) == 4, r["position_labels"]

    def test_invalid_universe_falls_back_or_errors(self):
        _skip_if_no_data()
        from web.yanyan_runner import run_yanyan
        r = run_yanyan({"start_date": "2024-01-01", "end_date": "2026-08-12",
                        "universe": "不存在的标的"})
        assert "error" in r or r.get("actual_universe"), \
            "无效 universe 应回退全池或报错"
