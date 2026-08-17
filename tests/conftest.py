"""测试公共fixtures与工具 — 全部合成数据, 无需网络/tushare token"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from strategy.signal import SignalConfig, compute_signals


# ─────────────────────────────────────────────────────────────
# 独立重放器: 不借用引擎内部状态, 从 actions + 价格矩阵按文档语义重建净值
# 用于验证 compute_signals 的自洽性 (任何引擎内部算错都会暴露为重放偏差)
# ─────────────────────────────────────────────────────────────
def replay_nav(actions, price_mat, etf_names, cfg):
    """按 compute_signals 的交易语义逐日重放:
      - 动作在信号日收盘执行, 当日收益归旧仓
      - 普通卖出/买入各扣 (1-c); 切换双边
      - 恐慌抄底: 空仓入场扣 (1-c); 已持其他按 alloc 扣 (1-c·alloc)²,
        组合仓期间每日按定权重 (1-alloc)·base + alloc·创业 计息
      - 任何非抄底动作清理组合仓
    返回 (n_days,) 净值序列, 从 vs 起才有意义 (此前无动作, nav=1)。
    """
    n_days = price_mat.shape[1]
    nav = 1.0
    pos = -1
    crash_alloc = 0.0
    crash_base = -1
    sat = -1          # 分档切换组合: 卫星腿
    sat_base = -1
    sat_alloc = 0.0
    cyb = etf_names.index("创业") if "创业" in etf_names else -1
    c = cfg.commission + cfg.slippage
    act_map = {}
    for a in actions:
        act_map.setdefault(a[0], []).append(a)

    out = np.ones(n_days)
    for t in range(n_days):
        if t > 0:
            if crash_alloc > 0:
                dr = 0.0
                if crash_base >= 0:
                    b0, b1 = price_mat[crash_base, t-1], price_mat[crash_base, t]
                    if not np.isnan(b0) and not np.isnan(b1) and b0 > 0:
                        dr += (1 - crash_alloc) * (b1 / b0 - 1)
                c0, c1 = price_mat[cyb, t-1], price_mat[cyb, t]
                if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                    dr += crash_alloc * (c1 / c0 - 1)
                nav *= (1 + dr)
            elif sat_alloc > 0:
                dr = 0.0
                b0, b1 = price_mat[sat_base, t-1], price_mat[sat_base, t]
                if not np.isnan(b0) and not np.isnan(b1) and b0 > 0:
                    dr += (1 - sat_alloc) * (b1 / b0 - 1)
                s0, s1 = price_mat[sat, t-1], price_mat[sat, t]
                if not np.isnan(s0) and not np.isnan(s1) and s0 > 0:
                    dr += sat_alloc * (s1 / s0 - 1)
                nav *= (1 + dr)
            elif pos >= 0:
                p0, p1 = price_mat[pos, t-1], price_mat[pos, t]
                if not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
                    nav *= p1 / p0
        for _, old_p, new_p, reason in act_map.get(t, []):
            if reason == "恐慌抄底":
                alloc = min(max(float(cfg.crash_alloc), 0.0), 1.0)
                if old_p < 0:
                    nav *= (1 - c * alloc)
                else:
                    nav *= (1 - c * alloc) ** 2
                crash_alloc = alloc
                crash_base = old_p if old_p >= 0 else -1
                pos = new_p
            elif reason == "恐慌换仓":
                nav *= (1 - c)          # 分档组合双腿全卖
                pos = -1
                sat = -1; sat_alloc = 0.0; sat_base = -1
            elif reason == ">3%切换(50%)":
                nav *= (1 - c * 0.5) ** 2
                sat = new_p; sat_base = old_p; sat_alloc = 0.5
            elif reason in ("分档升级全切", "核心止损", "核心止盈"):
                nav *= (1 - c * sat_alloc) ** 2 if reason == "分档升级全切" \
                    else (1 - c * sat_alloc)
                pos = new_p if new_p >= 0 else -1
                sat = -1; sat_alloc = 0.0; sat_base = -1
            elif reason in ("卫星回撤", "卫星止损", "卫星止盈"):
                nav *= (1 - c * sat_alloc)
                pos = sat_base if sat_base >= 0 else pos
                sat = -1; sat_alloc = 0.0; sat_base = -1
            elif reason == "卫星追新":
                nav *= (1 - c * sat_alloc) ** 2
                sat = new_p
            elif reason in ("组合止损", "组合止盈"):
                nav *= (1 - c)
                pos = -1
                sat = -1; sat_alloc = 0.0; sat_base = -1
            else:
                crash_alloc = 0.0
                crash_base = -1
                sat = -1; sat_alloc = 0.0; sat_base = -1
                if old_p >= 0:
                    nav *= (1 - c)
                if new_p is not None and new_p >= 0:
                    nav *= (1 - c)
                pos = new_p if new_p is not None else pos
        out[t] = nav
    return out


# ─────────────────────────────────────────────────────────────
# 合成数据工具
# ─────────────────────────────────────────────────────────────
def build_inputs(prices: dict, index_prices: dict):
    """从 {名字: 价格序列} 构建 compute_signals 所需的全部输入。
    prices: ETF 池; index_prices: 必须含 'cyb'(399006) 与 'sh'(000001)。
    全部序列等长。
    """
    etf_names = list(prices.keys())
    n_days = len(next(iter(prices.values())))
    price_mat = np.array([prices[nm] for nm in etf_names], dtype=float)

    def mom(arr, w):
        m = np.full(n_days, np.nan)
        m[w:] = (arr[w:] / arr[:-w] - 1) * 100
        return m

    mom20_mat = np.array([mom(prices[nm], 20) for nm in etf_names])
    mom10_mat = np.array([mom(prices[nm], 10) for nm in etf_names])
    sh_mom = mom(np.asarray(index_prices["sh"], dtype=float), 20)
    return dict(
        etf_names=etf_names, price_mat=price_mat,
        mom20_mat=mom20_mat, mom10_mat=mom10_mat, sh_mom=sh_mom,
        cyb_idx_p=np.asarray(index_prices["cyb"], dtype=float),
        sh_idx_p=np.asarray(index_prices["sh"], dtype=float),
        n_days=n_days,
    )


def gbm_prices(n, start=100.0, mu=0.0005, sigma=0.01, seed=0):
    """几何布朗运动合成价格 (确定性种子)"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n)
    return start * np.exp(np.cumsum(rets))


@pytest.fixture
def base_cfg():
    return SignalConfig()


@pytest.fixture
def calm_market():
    """平静市场: 3 ETF 慢涨 + 指数平稳, 400 天"""
    n = 400
    prices = {
        "纳指": gbm_prices(n, 100, 0.0008, 0.010, seed=1),
        "黄金": gbm_prices(n, 100, 0.0004, 0.006, seed=2),
        "创业": gbm_prices(n, 100, 0.0006, 0.014, seed=3),
    }
    index = {
        "cyb": gbm_prices(n, 3000, 0.0003, 0.010, seed=4),
        "sh": gbm_prices(n, 3000, 0.0002, 0.008, seed=5),
    }
    return build_inputs(prices, index)
