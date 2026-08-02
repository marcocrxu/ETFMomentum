"""核心信号函数 — 回测和实盘共用同一套逻辑"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SignalConfig:
    sh_cash: bool = True           # 上证空仓
    cash_skip_gt: float = 3.0      # 空仓>X%即买
    confirm_days: int = 2          # 确认天数
    min_hold: int = 10             # 最短持有
    min_gap: float = 1.0           # 最小动量差
    stop_loss: float = -5.0        # 止损线
    exception_gap: float = 3.0     # 例外动量差
    commission: float = 0.0001     # 手续费率(单边)
    slippage: float = 0.0          # 滑点/冲击成本(单边)
    accel_sigma: float = 3.0       # 加速止盈 σ
    accel_enabled: bool = True     # 加速止盈开关
    crash_sigma: float = -2.5      # 恐慌抄底 σ
    crash_enabled: bool = True     # 恐慌抄底开关
    crash_lock: int = 10           # 抄底锁仓天数
    method: str = "sigma"          # sigma / percentile / ewma_z
    method_pct: float = 2.0        # percentile: 使用P值 (仅method=percentile)
    method_halflife: int = 252     # ewma_z: 半衰期 (仅method=ewma_z)


def compute_signals(
    cfg: SignalConfig,
    etf_names: List[str],
    price_mat: np.ndarray,      # (n_etfs, n_days) ETF价格
    mom20_mat: np.ndarray,      # (n_etfs, n_days) 20日动量
    mom10_mat: np.ndarray,      # (n_etfs, n_days) 10日动量
    sh_mom: np.ndarray,         # (n_days,) 上证20日动量
    cyb_idx_p: np.ndarray,      # (n_days,) 创业板指价格
    sh_idx_p: np.ndarray,       # (n_days,) 上证指价格
    n_days: int,
) -> list:
    """
    返回 [(day_idx, action_type, target_etf_idx, reason), ...]
    action_type: 'BUY'|'SELL'|'CASH'|'HOLD'
    target_etf_idx: 0..n_etfs-1 or -1 for cash
    """

    # ── 加速止盈阈值 (可插拔校准) ──
    accel_th = {}
    for i in range(len(etf_names)):
        valid = mom10_mat[i][~np.isnan(mom10_mat[i])]
        if len(valid) >= 60:
            if cfg.method == "percentile":
                accel_th[i] = np.percentile(valid, 100 - cfg.method_pct)
            elif cfg.method == "ewma_z":
                s = pd.Series(mom10_mat[i]).ewm(halflife=cfg.method_halflife, min_periods=60).mean().iloc[-1]
                v = pd.Series(mom10_mat[i]).ewm(halflife=cfg.method_halflife, min_periods=60).var().iloc[-1]
                accel_th[i] = float(s + cfg.accel_sigma * np.sqrt(v)) if not np.isnan(s) and not np.isnan(v) else 99.0
            else:
                accel_th[i] = valid.mean() + cfg.accel_sigma * valid.std()
        else:
            accel_th[i] = 99.0

    # 同样方法用于恐慌阈值计算（在循环内使用）
    def calc_threshold(arr, t, sigma_val):
        valid = arr[:t][~np.isnan(arr[:t])]
        if len(valid) < 60: return None
        if cfg.method == "percentile":
            return np.percentile(valid, cfg.method_pct)
        elif cfg.method == "ewma_z":
            s = pd.Series(arr[:t]).ewm(halflife=cfg.method_halflife, min_periods=60).mean().iloc[-1]
            v = pd.Series(arr[:t]).ewm(halflife=cfg.method_halflife, min_periods=60).var().iloc[-1]
            return float(s + sigma_val * np.sqrt(v)) if not np.isnan(s) and not np.isnan(v) else None
        else:
            return valid.mean() + sigma_val * valid.std()

    # ── 恐慌抄底: 10日动量 for 399006 + 000001 ──
    sw = 10
    mom10_cyb = np.full(n_days, np.nan)
    mom10_sh_idx = np.full(n_days, np.nan)
    mom10_cyb[sw:] = (cyb_idx_p[sw:] / cyb_idx_p[:-sw] - 1) * 100
    mom10_sh_idx[sw:] = (sh_idx_p[sw:] / sh_idx_p[:-sw] - 1) * 100

    ma200_cyb = pd.Series(cyb_idx_p).rolling(200).mean().values
    ma200_sh = pd.Series(sh_idx_p).rolling(200).mean().values

    # ── 有效起点 ──
    # 至少2个标的有动量数据即开跑；新上市标的在数据足够后自然加入竞争，
    # 不会把整个回测起点推迟到最晚上市标的的日期
    vs = 0
    valid_cnt = np.sum(~np.isnan(mom20_mat), axis=0)
    cand = np.where(valid_cnt >= min(2, len(etf_names)))[0]
    if len(cand) > 0: vs = cand[0]
    fv_sh = np.where(~np.isnan(sh_mom))[0]
    if len(fv_sh) > 0: vs = max(vs, fv_sh[0])

    # ── 模拟 ──
    nav = 1.0; pos = -1; hs = -1; hn = 1.0; streak = 0
    crash_held_until = -1
    actions = []
    nav_log = []

    for t in range(vs, n_days):
        # 更新净值（仅用于计算PnL做决策）
        if pos >= 0:
            prev_p = price_mat[pos, max(0, t-1)]
            curr_p = price_mat[pos, t]
            if not np.isnan(prev_p) and not np.isnan(curr_p) and prev_p > 0:
                nav *= curr_p / prev_p

        hd = t - hs if pos >= 0 else 0
        pnl = (nav / hn - 1) * 100 if pos >= 0 and hn > 0 else 0
        ms = mom20_mat[:, t]; vm = ~np.isnan(ms)

        if not vm.any():
            pp = price_mat[pos, t] if pos >= 0 else np.nan; continue
        if np.isnan(sh_mom[t]):
            pp = price_mat[pos, t] if pos >= 0 else np.nan; continue

        ti = int(np.argmax(np.where(vm, ms, -np.inf)))
        sc = (sh_mom[t] > np.nanmax(ms)) if cfg.sh_cash else False
        sw = False; reason = ""

        if ti != pos: streak += 1
        else: streak = 0

        # ── 恐慌抄底 ──
        crash_now = False
        if t >= 252:
            th_sh = calc_threshold(mom10_sh_idx, t, cfg.crash_sigma)
            if th_sh is not None and mom10_sh_idx[t] < th_sh and sh_idx_p[t] < ma200_sh[t]:
                crash_now = True
            if not crash_now:
                th_cyb = calc_threshold(mom10_cyb, t, cfg.crash_sigma)
                if th_cyb is not None and mom10_cyb[t] < th_cyb and cyb_idx_p[t] < ma200_cyb[t]:
                    crash_now = True

        if cfg.crash_enabled and crash_now and t >= crash_held_until:
            # 找到创业板在etf_names中的索引
            cyb_idx = etf_names.index("创业") if "创业" in etf_names else 0
            sw = True; np_pos = cyb_idx; streak = 0
            crash_held_until = t + cfg.crash_lock
            reason = "恐慌抄底"
        elif sc:
            sw = True; np_pos = -1; streak = 0; reason = "上证空仓"
        elif pos == -1:
            lm = ms[ti]
            if lm >= cfg.cash_skip_gt and not np.isnan(lm):
                sw = True; np_pos = ti; streak = 0; reason = ">3%%即买"
            elif streak >= cfg.confirm_days:
                sw = True; np_pos = ti; streak = 0; reason = "确认买入"
        elif ti != pos and t >= crash_held_until:
            gap = ms[ti] - ms[pos]; met = (streak >= cfg.confirm_days)
            if pnl <= cfg.stop_loss:
                sw = True; np_pos = ti; streak = 0; reason = "止损"
            elif gap >= cfg.exception_gap and met:
                sw = True; np_pos = ti; streak = 0; reason = ">3%%切换"
            elif hd >= cfg.min_hold and gap >= cfg.min_gap and met:
                sw = True; np_pos = ti; streak = 0; reason = "正常切换"
        elif cfg.accel_enabled and pos >= 0 and mom10_mat[pos, t] > accel_th.get(pos, 99):
            sw = True; np_pos = -1; streak = 0; reason = "加速止盈"

        if sw:
            cost = cfg.commission + cfg.slippage
            if pos >= 0: nav *= (1 - cost)
            if np_pos >= 0: nav *= (1 - cost); hs = t; hn = nav
            old_pos = pos; pos = np_pos
            actions.append((t, old_pos, np_pos, reason))

        nav_log.append(nav)
        pp = price_mat[pos, t] if pos >= 0 else np.nan

    # 期末平仓（仅用于回测统计，不在交易记录中显示）
    if pos >= 0:
        nav *= (1 - cfg.commission - cfg.slippage)

    return actions, nav, nav_log


def compute_returns(actions: list, price_mat: np.ndarray,
                    etf_names: List[str], n_days: int,
                    start_day: int = 0) -> dict:
    """从信号序列和价格矩阵重建净值。手续费已在compute_signals中处理, 此处不再重复扣除。"""
    nav = 1.0; pos = -1
    hs = -1; hn = 1.0
    trades = []
    nav_log = [1.0] * start_day

    act_map = {}
    for a in actions:
        act_map.setdefault(a[0], []).append(a)

    for t in range(start_day, n_days):
        if t in act_map:
            for _, old_p, new_p, reason in act_map[t]:
                if old_p >= 0:
                    ret = (nav / hn - 1) * 100 if hn > 0 else 0
                    trades.append({
                        "entry_idx": hs, "exit_idx": t,
                        "asset": etf_names[old_p],
                        "days": t - hs, "ret": round(ret, 2),
                        "reason": reason,
                    })
                if new_p >= 0:
                    hs = t; hn = nav
                pos = new_p if new_p is not None else pos

        if pos >= 0:
            prev_p = price_mat[pos, max(start_day, t - 1)]
            curr_p = price_mat[pos, t]
            if not np.isnan(prev_p) and not np.isnan(curr_p) and prev_p > 0:
                nav *= curr_p / prev_p
        nav_log.append(nav)

    total_ret = (nav - 1) * 100
    n_days_traded = n_days - start_day
    ann_ret = (nav ** (1 / max(n_days_traded / 252, 0.01)) - 1) * 100

    nav_arr = np.array(nav_log)
    cm = np.maximum.accumulate(nav_arr)
    dd = (nav_arr - cm) / cm * 100

    return {
        "total_ret": round(total_ret, 2),
        "ann_ret": round(ann_ret, 2),
        "max_dd": round(float(np.min(dd)), 2) if len(dd) > 0 else 0,
        "trades": trades,
        "nav_log": nav_log,
    }
