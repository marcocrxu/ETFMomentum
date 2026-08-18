"""yanyan 策略 — 体制层(牛熊开关) + 动量层(风险调整动量轮动) · 2026-08-17

规则 (用户原文):
1. 体制层: 沪深300(510300) 牛熊开关; 熊市 = 沪深300 现价 < MA200
   - 熊市: A股池禁止持仓, 只允许 跨境池(纳指/港股创新药) + 黄金
2. 动量层:
   - score = MOM20 / VOL20
   - 进场: 合格池 score 第一 且 MOM20 > 0
   - 离场: 持仓 MOM20 转负
   - 轮动缓冲: 挑战者 MOM20 须超持仓 2% 才切换 (防频繁换手)
   - 过热保护: 持仓 MOM20 > 40% 进入过热态, 离场收紧为 MOM5 < 0;
     过热标的再进场须 MOM5 > Q (默认 Q=3) 确认
3. 标的池 11 只: A股 7 (沪深300/中证500/创业板/科创50/有色/红利低波/中证2000)
   + 跨境 2 (纳指/港股创新药) + 黄金 + 货币(511880 空仓停靠, 不入排名)

口径假设 (可改):
- 熊市判定 MA 周期 = regime_ma (默认200); MA 未就绪前按牛市处理
- 时变宇宙: 新上市标的在数据足够后自然加入排名 (NaN 掩码)
- 费用: 买/卖各扣 (1-commission-slippage); 空仓停靠货基的进出也各计一次费 (与主引擎 cash_park_code 同口径)
- 过热再进场闸门按标的独立计时, MOM5>Q 后自动解除; 只约束过热离场后的再进场
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class YanyanConfig:
    # 体制层
    regime_ma: int = 200          # 沪深300 牛熊均线周期
    # 动量层
    mom_period: int = 20          # MOM20 周期
    vol_period: int = 20          # VOL20 周期
    switch_buffer: float = 2.0    # 轮动缓冲: 挑战者 MOM20 须超持仓 2%
    overheat_mom: float = 40.0    # 过热态门槛: 持仓 MOM20 > 40%
    overheat_q: float = 3.0       # 过热标的再进场确认: MOM5 > Q
    # 执行
    commission: float = 0.0001    # 手续费率(单边)
    slippage: float = 0.0         # 滑点(单边)
    cash_park: bool = True        # 空仓停靠 511880 (收益计入净值, 进出各计费)
    cash_skip_buffer: bool = True # 空仓入场同样要求缓冲 (挑战者 MOM20 超过 2% 才买, 防贴线抖动; 0=按原文>0即买)


# 标的池: name -> (code, group)
# group: A = A股池(受牛熊开关约束), X = 跨境池(不受约束), G = 黄金(牛市备胎/熊市参赛), C = 货币(空仓停靠, 不入排名)
UNIVERSE = [
    ("沪深300", "510300.SH", "A"),
    ("中证500", "510500.SH", "A"),
    ("创业板", "159915.SZ", "A"),
    ("科创50", "588080.SH", "A"),
    ("有色", "512400.SH", "A"),
    ("红利低波", "512890.SH", "A"),
    ("中证2000", "563300.SH", "A"),
    ("纳指", "513100.SH", "X"),
    ("港股创新药", "513120.SH", "X"),
    ("黄金", "518880.SH", "G"),
    ("货币", "511880.SH", "C"),
]


def rolling_vol20(price_mat: np.ndarray, w: int = 20) -> np.ndarray:
    """20日收益波动率 (%/日), 逐标的"""
    vol = np.full(price_mat.shape, np.nan)
    for i in range(price_mat.shape[0]):
        vol[i] = pd.Series(price_mat[i]).pct_change().rolling(w).std().values * 100.0
    return vol


def mom(arr: np.ndarray, w: int) -> np.ndarray:
    m = np.full(arr.shape[0], np.nan)
    m[w:] = (arr[w:] / arr[:-w] - 1) * 100
    return m


def compute_yanyan(cfg: YanyanConfig, names: List[str], groups: List[str],
                   price_mat: np.ndarray,       # (n_assets, n_days) 含货币(最后)
                   hs300_p: np.ndarray,         # (n_days,) 沪深300 现价
                   n_days: int,
                   ma200_hs: np.ndarray = None,
                   ) -> tuple:
    """返回 (actions, nav, nav_log)
    actions: [(day_idx, old_idx, new_idx, reason), ...], -1=空仓(货基停靠)"""
    n_assets = len(names)
    cost = cfg.commission + cfg.slippage
    mom20_mat = np.array([mom(price_mat[i], cfg.mom_period) for i in range(n_assets)])
    mom5_mat = np.array([mom(price_mat[i], 5) for i in range(n_assets)])
    vol_mat = rolling_vol20(price_mat, cfg.vol_period)
    if ma200_hs is None:
        ma200_hs = pd.Series(hs300_p).rolling(cfg.regime_ma).mean().values

    with np.errstate(divide="ignore", invalid="ignore"):
        score_mat = np.where((vol_mat > 0) & ~np.isnan(mom20_mat),
                             mom20_mat / np.where(vol_mat > 0, vol_mat, np.nan), np.nan)

    # 有效起点: 沪深300 MA 就绪 且 ≥2 标的可排名
    vs = 0
    hs_ok = np.where(~np.isnan(ma200_hs))[0]
    if len(hs_ok) > 0:
        vs = max(vs, hs_ok[0])
    valid_cnt = np.sum(~np.isnan(score_mat), axis=0)
    cand = np.where(valid_cnt >= 2)[0]
    if len(cand) > 0:
        vs = max(vs, cand[0])
    vs = min(vs, n_days - 1)

    nav = 1.0
    pos = -1
    hs_day = -1          # 持仓入场日
    hn = 1.0             # 入场净值
    overheat_gate = np.zeros(n_assets, dtype=bool)   # 过热再进场闸门 (按标的)
    actions = []
    nav_log = []

    def _is_bear(t):
        return bool(not np.isnan(hs300_p[t]) and not np.isnan(ma200_hs[t])
                    and hs300_p[t] < ma200_hs[t])

    def _eligible_mask(t, bear):
        ok = ~np.isnan(score_mat[:, t])
        if bear:
            # 熊市: A股池禁止, 跨境+黄金可用
            for i, g in enumerate(groups):
                if g == "A":
                    ok[i] = False
        # 货币不入排名
        for i, g in enumerate(groups):
            if g == "C":
                ok[i] = False
        return ok

    def _challenger(t, ok):
        """合格池内 score 第一 (可为 -1)"""
        sc = np.where(ok, score_mat[:, t], np.nan)
        if not np.any(~np.isnan(sc)):
            return -1
        return int(np.nanargmax(sc))

    for t in range(vs, n_days):
        # ── 净值更新 (当日收益归旧仓) ──
        if pos >= 0:
            p0 = price_mat[pos, max(0, t - 1)]
            p1 = price_mat[pos, t]
            if not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
                nav *= p1 / p0
        else:
            # 空仓停靠货基
            if cfg.cash_park and len(price_mat) > 0:
                cash_i = None
                for i, g in enumerate(groups):
                    if g == "C":
                        cash_i = i
                        break
                if cash_i is not None:
                    p0 = price_mat[cash_i, max(0, t - 1)]
                    p1 = price_mat[cash_i, t]
                    if not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
                        nav *= p1 / p0

        bear = _is_bear(t)
        ok = _eligible_mask(t, bear)
        held_ok = ok[pos] if pos >= 0 else False
        m20_pos = mom20_mat[pos, t] if pos >= 0 else np.nan
        m5_pos = mom5_mat[pos, t] if pos >= 0 else np.nan

        exit_now = False
        exit_reason = ""
        # 1) 牛熊开关: 熊市清 A 股持仓
        if pos >= 0 and groups[pos] == "A" and bear:
            exit_now = True
            exit_reason = "牛熊开关"
        # 2) 过热离场: MOM20>40% 且 MOM5<0
        elif pos >= 0 and not np.isnan(m20_pos) and m20_pos > cfg.overheat_mom \
                and not np.isnan(m5_pos) and m5_pos < 0:
            exit_now = True
            exit_reason = "过热离场"
        # 3) 常规离场: 持仓 MOM20 转负
        elif pos >= 0 and not np.isnan(m20_pos) and m20_pos < 0:
            exit_now = True
            exit_reason = "动量转负"

        if exit_now:
            # 离场 → 空仓 (货基停靠)
            nav *= (1 - cost)
            if cfg.cash_park:
                nav *= (1 - cost)          # 买入货基计费 (与主引擎口径一致)
            if exit_reason == "过热离场":
                overheat_gate[pos] = True
            actions.append((t, pos, -1, exit_reason))
            pos = -1
            hs_day = -1
            hn = nav

        # ── 进场 / 切换 ──
        if pos == -1:
            chal = _challenger(t, ok)
            if chal >= 0:
                m20_c = mom20_mat[chal, t]
                # 过热闸门: 过热标的须 MOM5>Q 才允许再进场
                if overheat_gate[chal] and not np.isnan(mom5_mat[chal, t]) \
                        and mom5_mat[chal, t] <= cfg.overheat_q:
                    chal = -1
            if chal >= 0:
                m20_c = mom20_mat[chal, t]
                m5_c = mom5_mat[chal, t]
                if not np.isnan(m20_c) and m20_c > 0:
                    buy = True
                    reason = "进场"
                    if cfg.cash_skip_buffer:
                        # 空仓入场也要求缓冲: 挑战者 MOM20 须 > 2% (原文"且 MOM20>0"的防贴线强化, 可关)
                        if m20_c <= cfg.switch_buffer:
                            buy = False
                    if buy:
                        nav *= (1 - cost)
                        if cfg.cash_park:
                            nav *= (1 - cost)      # 卖出货基计费
                        actions.append((t, -1, chal, reason))
                        pos = chal
                        hs_day = t
                        hn = nav
                        if overheat_gate[chal] and not np.isnan(m5_c) and m5_c > cfg.overheat_q:
                            overheat_gate[chal] = False
        else:
            # 切换: 挑战者 score 第一, MOM20 须超持仓 2% (缓冲)
            chal = _challenger(t, ok)
            if chal >= 0 and chal != pos and not np.isnan(mom20_mat[chal, t]) \
                    and not np.isnan(m20_pos):
                m20_c = mom20_mat[chal, t]
                if overheat_gate[chal] and not np.isnan(mom5_mat[chal, t]) \
                        and mom5_mat[chal, t] <= cfg.overheat_q:
                    chal = -1
                if chal >= 0 and m20_c - m20_pos >= cfg.switch_buffer:
                    nav *= (1 - cost) ** 2
                    actions.append((t, pos, chal, "轮动切换"))
                    pos = chal
                    hs_day = t
                    hn = nav
                    if overheat_gate[chal] and not np.isnan(mom5_mat[chal, t]) \
                            and mom5_mat[chal, t] > cfg.overheat_q:
                        overheat_gate[chal] = False

        nav_log.append(nav)

    # 期末平仓 (仅统计口径, 不入交易记录)
    if pos >= 0:
        nav *= (1 - cost)
    return actions, nav, nav_log
