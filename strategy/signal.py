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
    crash_lock: int = 7            # 抄底锁仓天数 (与 web UI / bt_runner 默认一致)
    method: str = "sigma"          # sigma / percentile / ewma_z
    method_pct: float = 2.0        # percentile: 使用P值 (仅method=percentile)
    method_halflife: int = 252     # ewma_z: 半衰期 (仅method=ewma_z)
    abs_mom_enabled: bool = False  # 绝对动量门: 全池动量为负时强制空仓 (抄底锁仓期内豁免)
    defensive_enabled: bool = False  # 防御切换: 空仓信号触发时改持防御资产(动量最强且为正)
    risk_adj_mom: bool = False     # 风险调整动量: 排名用 动量/20日波动率 (阈值仍用原始动量)
    accel_cooldown: int = 0        # 加速止盈后N天禁止买回同一标的 (0=关闭, 恐慌抄底豁免)
    sh_cash_cooldown: int = 3      # 上证空仓后再入场冷却天数 (0=关闭, 恐慌抄底豁免; 2026-08 起默认3天)
    sh_cash_exempt: bool = False   # 上证空仓仅对A股标的生效 (豁免纳指/黄金)
    multi_mom: bool = False        # 多周期动量: 排名分 = 3×mom10+2×mom20+1×mom60
    fip_tiebreak: bool = False     # FIP动量质量: 动量差<min_gap时选上涨路径更平滑者
    top_k: int = 1                 # 持仓数: 1=Top-1全仓(原行为); >1=Top-K等权轮动(独立引擎)
    mom_skip: int = 0              # 排名动量跳过最近N天(短期反转规避), 0=关闭


# 防御资产池: 空仓/防御信号触发时的候选持有标的
DEFENSIVE_NAMES = {"黄金", "红利低波", "5年国债"}

# 上证空仓豁免名单: 与A股低相关的资产, 豁免模式下不受上证空仓过滤
SH_CASH_EXEMPT_NAMES = {"纳指", "黄金"}


def _compute_signals_topk(
    cfg: SignalConfig,
    etf_names: List[str],
    price_mat: np.ndarray,
    mom20_mat: np.ndarray,
    mom10_mat: np.ndarray,
    sh_mom: np.ndarray,
    cyb_idx_p: np.ndarray,
    sh_idx_p: np.ndarray,
    n_days: int,
) -> tuple:
    """Top-K 等权轮动引擎 (top_k>1 时独立于 Top-1 路径)

    持有动量得分最强的 K 个标的, 每日再平衡为等权, 空余仓位持现金。
    复用上证空仓/恐慌抄底/止损/加速止盈/确认入场规则;
    不支持防御切换/豁免/多周期/风险调整/FIP (均实测负优化, 不与 Top-K 组合)。
    返回与 Top-1 相同的 (actions, nav, nav_log)。
    """
    n_etfs = len(etf_names)
    K = min(cfg.top_k, n_etfs)

    # ── 加速止盈阈值 (与 Top-1 相同的 expanding 校准) ──
    accel_th_mat = np.full((n_etfs, n_days), np.nan)
    for i in range(n_etfs):
        s = pd.Series(mom10_mat[i])
        if cfg.method == "percentile":
            th = s.expanding(min_periods=60).quantile((100 - cfg.method_pct) / 100.0)
        elif cfg.method == "ewma_z":
            e = s.ewm(halflife=cfg.method_halflife, min_periods=60)
            th = e.mean() + cfg.accel_sigma * np.sqrt(e.var())
        else:
            ex = s.expanding(min_periods=60)
            th = ex.mean() + cfg.accel_sigma * ex.std()
        accel_th_mat[i] = th.shift(1).values

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

    sw_ = 10
    mom10_cyb = np.full(n_days, np.nan)
    mom10_sh_idx = np.full(n_days, np.nan)
    mom10_cyb[sw_:] = (cyb_idx_p[sw_:] / cyb_idx_p[:-sw_] - 1) * 100
    mom10_sh_idx[sw_:] = (sh_idx_p[sw_:] / sh_idx_p[:-sw_] - 1) * 100
    ma200_cyb = pd.Series(cyb_idx_p).rolling(200).mean().values
    ma200_sh = pd.Series(sh_idx_p).rolling(200).mean().values

    # ── 有效起点 (与 Top-1 一致) ──
    vs = 0
    valid_cnt = np.sum(~np.isnan(mom20_mat), axis=0)
    cand = np.where(valid_cnt >= min(2, n_etfs))[0]
    if len(cand) > 0: vs = cand[0]
    fv_sh = np.where(~np.isnan(sh_mom))[0]
    if len(fv_sh) > 0: vs = max(vs, fv_sh[0])

    # ── 模拟 ──
    nav = 1.0
    w = np.zeros(n_etfs)                 # 当前权重 (每日漂移)
    entry_day = np.full(n_etfs, -1)
    entry_px = np.full(n_etfs, np.nan)
    streak = 0; top1_prev = -1
    crash_held_until = -1
    sh_cash_block_until = -1
    accel_block_idx = -1; accel_block_until = -1
    actions = []
    nav_log = []
    cost_rate = cfg.commission + cfg.slippage

    for t in range(vs, n_days):
        # ── 用当前权重更新净值并漂移 ──
        prev_p = price_mat[:, max(0, t - 1)]
        curr_p = price_mat[:, t]
        held_mask = w > 1e-9
        if held_mask.any():
            ok = held_mask & ~np.isnan(prev_p) & ~np.isnan(curr_p) & (prev_p > 0)
            pr = np.zeros(n_etfs)
            pr[ok] = curr_p[ok] / prev_p[ok] - 1
            port_ret = float(np.sum(w * pr))
            nav *= (1 + port_ret)
            grow = np.ones(n_etfs); grow[ok] = curr_p[ok] / prev_p[ok]
            if 1 + port_ret > 0:
                w = w * grow / (1 + port_ret)

        ms = mom20_mat[:, t]; vm = ~np.isnan(ms)
        if not vm.any() or np.isnan(sh_mom[t]):
            continue

        score = np.where(vm, ms, np.nan)
        if accel_block_idx >= 0 and t < accel_block_until:
            score[accel_block_idx] = np.nan
        sv = ~np.isnan(score)

        # top1 与 streak (用于入场/切换确认)
        ti = int(np.argmax(np.where(sv, score, -np.inf))) if sv.any() else -1
        streak = streak + 1 if ti != top1_prev else 0
        top1_prev = ti

        # ── 上证空仓 ──
        sc = (sh_mom[t] > np.nanmax(ms)) if cfg.sh_cash else False
        if sc and cfg.sh_cash_cooldown > 0:
            sh_cash_block_until = t + cfg.sh_cash_cooldown

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

        held_idx = [i for i in range(n_etfs) if w[i] > 1e-9]
        buys = []; sells = []
        cyb_pool_idx = etf_names.index("创业") if "创业" in etf_names else -1

        if cfg.crash_enabled and crash_now and t >= crash_held_until and cyb_pool_idx >= 0:
            crash_held_until = t + cfg.crash_lock
            if cyb_pool_idx not in held_idx:
                if len(held_idx) >= K:
                    worst = min(held_idx, key=lambda i: score[i] if not np.isnan(score[i]) else -np.inf)
                    sells.append((worst, "恐慌换仓")); held_idx.remove(worst)
                buys.append((cyb_pool_idx, "恐慌抄底")); held_idx.append(cyb_pool_idx)
        elif sc:
            for i in held_idx: sells.append((i, "上证空仓"))
            held_idx = []
        elif cfg.abs_mom_enabled and sv.any() and np.nanmax(ms) < 0 and t >= crash_held_until:
            for i in held_idx: sells.append((i, "绝对动量空仓"))
            held_idx = []
        else:
            # ── 正常轮动: 目标集 = 得分前 K ──
            order = np.argsort(np.where(sv, score, -np.inf))[::-1]
            target = [int(i) for i in order[:K] if sv[i]]
            bought = set()

            # 1) 逐个检查持仓: 止损 / 加速止盈 / 正常切换
            for i in list(held_idx):
                if i in target: continue
                px_now = price_mat[i, t]
                pnl = (px_now / entry_px[i] - 1) * 100 if (entry_px[i] > 0 and not np.isnan(px_now)) else 0.0
                hd = t - entry_day[i]
                reason = None
                if t >= crash_held_until and pnl <= cfg.stop_loss:
                    reason = "止损"
                elif (not np.isnan(accel_th_mat[i, t])
                        and mom10_mat[i, t] > accel_th_mat[i, t]):
                    reason = "加速止盈"
                elif t >= crash_held_until and streak >= cfg.confirm_days and not np.isnan(score[i]):
                    j = next((x for x in target if x not in held_idx and x not in bought), None)
                    if j is not None:
                        gap = score[j] - score[i]
                        if gap >= cfg.exception_gap:
                            reason = ">3%%切换"
                        elif hd >= cfg.min_hold and gap >= cfg.min_gap:
                            reason = "正常切换"
                if reason is None: continue
                sells.append((i, reason)); held_idx.remove(i)
                if reason == "加速止盈" and cfg.accel_cooldown > 0:
                    accel_block_idx = i; accel_block_until = t + cfg.accel_cooldown
                j = next((x for x in target if x not in held_idx and x not in bought), None)
                if j is not None and len(held_idx) < K:
                    bought.add(j); buys.append((j, reason)); held_idx.append(j)

            # 2) 空余仓位入场 (空仓冷却期内禁止)
            if t >= sh_cash_block_until:
                for j in target:
                    if len(held_idx) >= K: break
                    if j in held_idx or j in bought: continue
                    lm = ms[j]
                    strong = not np.isnan(lm) and lm >= cfg.cash_skip_gt
                    if strong or streak >= cfg.confirm_days:
                        bought.add(j)
                        buys.append((j, ">3%%即买" if strong else "确认买入"))
                        held_idx.append(j)

        # ── 执行: 重建等权, 按换手率扣成本 ──
        if sells or buys:
            new_w = np.zeros(n_etfs)
            if held_idx:
                for i in held_idx: new_w[i] = 1.0 / len(held_idx)
            turnover = float(np.sum(np.abs(new_w - w)))
            if turnover > 0:
                nav *= (1 - cost_rate * turnover)
            for i, reason in sells:
                actions.append((t, i, -1, reason))
                entry_day[i] = -1; entry_px[i] = np.nan
            for j, reason in buys:
                actions.append((t, -1, j, reason))
                entry_day[j] = t; entry_px[j] = price_mat[j, t]
            w = new_w

        nav_log.append(nav)

    # 期末平仓
    if w.any():
        nav *= (1 - cost_rate * float(np.sum(w)))
    return actions, nav, nav_log


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

    if cfg.top_k > 1:
        return _compute_signals_topk(cfg, etf_names, price_mat, mom20_mat,
                                     mom10_mat, sh_mom, cyb_idx_p, sh_idx_p, n_days)

    # ── 加速止盈阈值 (可插拔校准) ──
    # expanding 逐日计算, shift(1) 保证 t 日阈值只用 t-1 及之前的数据
    # (与恐慌阈值 calc_threshold 的 arr[:t] 口径一致, 修复原全样本计算的前视偏差)
    n_etfs = len(etf_names)
    accel_th_mat = np.full((n_etfs, n_days), np.nan)
    for i in range(n_etfs):
        s = pd.Series(mom10_mat[i])
        if cfg.method == "percentile":
            th = s.expanding(min_periods=60).quantile((100 - cfg.method_pct) / 100.0)
        elif cfg.method == "ewma_z":
            e = s.ewm(halflife=cfg.method_halflife, min_periods=60)
            th = e.mean() + cfg.accel_sigma * np.sqrt(e.var())
        else:
            ex = s.expanding(min_periods=60)
            th = ex.mean() + cfg.accel_sigma * ex.std()
        accel_th_mat[i] = th.shift(1).values

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
    accel_block_idx = -1; accel_block_until = -1   # 加速止盈冷却状态
    sh_cash_block_until = -1                        # 空仓再入场冷却状态
    actions = []
    nav_log = []

    def _def_target(ms, vm):
        """防御资产中动量最强且为正者, 无则 -1 (防御资产也走绝对动量)"""
        best, best_m = -1, 0.0
        for i, nm in enumerate(etf_names):
            if nm in DEFENSIVE_NAMES and vm[i] and ms[i] > best_m:
                best, best_m = i, ms[i]
        return best

    # ── 20日波动率 (仅风险调整动量开启时使用) ──
    vol_mat = np.full(price_mat.shape, np.nan)
    if cfg.risk_adj_mom:
        for i in range(len(etf_names)):
            vol_mat[i] = pd.Series(price_mat[i]).pct_change().rolling(20).std().values

    # ── 60日动量 (仅多周期动量开启时使用) ──
    mom60_mat = None
    if cfg.multi_mom:
        mom60_mat = np.full(price_mat.shape, np.nan)
        mom60_mat[:, 60:] = (price_mat[:, 60:] / price_mat[:, :-60] - 1) * 100

    # ── 20日上涨天数占比 (仅FIP开启时使用) ──
    up_ratio = None
    if cfg.fip_tiebreak:
        up_ratio = np.full(price_mat.shape, np.nan)
        for i in range(len(etf_names)):
            up_ratio[i] = (pd.Series(price_mat[i]).pct_change() > 0).rolling(20).mean().values

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

        ti_raw = int(np.argmax(np.where(vm, ms, -np.inf)))

        # ── 排名分数: 多周期动量 → 风险调整波动率除法, 阈值判断仍用原始 ms ──
        if cfg.multi_mom:
            base = 3.0 * mom10_mat[:, t] + 2.0 * ms + mom60_mat[:, t]
            score = np.where(vm & ~np.isnan(mom10_mat[:, t]) & ~np.isnan(mom60_mat[:, t]), base, np.nan)
        else:
            score = np.where(vm, ms, np.nan)

        if cfg.risk_adj_mom:
            v_t = vol_mat[:, t]
            with np.errstate(divide='ignore', invalid='ignore'):
                score = np.where(~np.isnan(score) & (v_t > 0), score / np.where(v_t > 0, v_t, 1.0), np.nan)

        # ── 加速止盈冷却: N天内把被止盈标的移出候选 (恐慌抄底豁免) ──
        if accel_block_idx >= 0 and t < accel_block_until:
            score = np.where(np.arange(len(ms)) == accel_block_idx, np.nan, score)

        sv = ~np.isnan(score)
        ti = int(np.argmax(np.where(sv, score, -np.inf))) if sv.any() else ti_raw

        # ── FIP 动量质量: 前两名动量差<min_gap时, 选上涨路径更平滑者 ──
        if cfg.fip_tiebreak and sv.sum() >= 2:
            order = np.argsort(np.where(sv, score, -np.inf))
            b, s2 = int(order[-1]), int(order[-2])
            if (abs(ms[b] - ms[s2]) < cfg.min_gap
                    and not np.isnan(up_ratio[b, t]) and not np.isnan(up_ratio[s2, t])
                    and up_ratio[s2, t] > up_ratio[b, t]):
                ti = s2

        if cfg.sh_cash and cfg.sh_cash_exempt:
            # 豁免模式: 上证只与非豁免(A股)标的的动量比较
            ne = [ms[i] for i, nm in enumerate(etf_names)
                  if nm not in SH_CASH_EXEMPT_NAMES and vm[i]]
            sc = (sh_mom[t] > max(ne)) if ne else False
        elif cfg.sh_cash:
            sc = sh_mom[t] > np.nanmax(ms)
        else:
            sc = False
        sw = False; reason = ""; accel_exit = False

        # ── 空仓冷却: 上证空仓信号存续期间每天顺延, 冷却期内禁止再入场 ──
        if sc and cfg.sh_cash_cooldown > 0:
            sh_cash_block_until = t + cfg.sh_cash_cooldown

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

        cyb_pool_idx = etf_names.index("创业") if "创业" in etf_names else -1
        if (cfg.crash_enabled and crash_now and t >= crash_held_until
                and cyb_pool_idx >= 0):
            # 恐慌抄底只买创业; 创业不在池时跳过 (原实现会静默买 index 0)
            sw = True; np_pos = cyb_pool_idx; streak = 0
            crash_held_until = t + cfg.crash_lock
            reason = "恐慌抄底"
        elif sc:
            if cfg.sh_cash_exempt and pos >= 0 and etf_names[pos] in SH_CASH_EXEMPT_NAMES:
                pass  # 持有豁免标的(纳指/黄金), 上证空仓不出场
            elif cfg.sh_cash_exempt and pos == -1:
                # 空仓时仍可按正常入场规则买豁免标的
                ex_idx = [i for i, nm in enumerate(etf_names)
                          if nm in SH_CASH_EXEMPT_NAMES and vm[i]]
                bought = False
                if ex_idx:
                    ei = max(ex_idx, key=lambda i: ms[i])
                    lm = ms[ei]
                    if not np.isnan(lm) and (lm >= cfg.cash_skip_gt or streak >= cfg.confirm_days):
                        sw = True; np_pos = ei; streak = 0
                        reason = ">3%%即买(豁免)" if lm >= cfg.cash_skip_gt else "确认买入(豁免)"
                        bought = True
                if not bought:
                    sw = True; np_pos = -1; streak = 0; reason = "上证空仓"
            elif cfg.defensive_enabled:
                tgt = _def_target(ms, vm)
                if tgt >= 0:
                    if tgt != pos:
                        sw = True; np_pos = tgt; streak = 0; reason = "防御切换"
                    # tgt == pos: 已持有最优防御资产, 不动
                else:
                    sw = True; np_pos = -1; streak = 0; reason = "上证空仓"
            else:
                sw = True; np_pos = -1; streak = 0; reason = "上证空仓"
        elif cfg.abs_mom_enabled and ms[ti] < 0 and t >= crash_held_until:
            # 绝对动量门: 全池(含防御)动量为负 → 空仓, 且阻止下方买入分支
            # (锁仓期内豁免, 避免抄底仓位次日被门卖掉)
            if pos >= 0:
                sw = True; np_pos = -1; streak = 0; reason = "绝对动量空仓"
        elif pos == -1:
            if t < sh_cash_block_until:
                pass  # 空仓冷却期内, 禁止再入场 (恐慌抄底走独立分支, 不受影响)
            else:
                lm = ms[ti]
                if lm >= cfg.cash_skip_gt and not np.isnan(lm):
                    sw = True; np_pos = ti; streak = 0; reason = ">3%%即买"
                elif streak >= cfg.confirm_days:
                    sw = True; np_pos = ti; streak = 0; reason = "确认买入"
        elif ti != pos and t >= crash_held_until:
            # 注意: 止损只在存在更优候选(ti != pos)时检查——持仓仍是动量第一名时
            # 永不止损 (相对动量取舍: 全池齐跌且持仓最抗跌时无处可切, 保持持有)。
            # 这意味着实际亏损可能显著超过 stop_loss 线 (如 2026-06 纳指 -9.79%)。
            gap = ms[ti] - ms[pos]; met = (streak >= cfg.confirm_days)
            if pnl <= cfg.stop_loss:
                sw = True; np_pos = ti; streak = 0; reason = "止损"
            elif gap >= cfg.exception_gap and met:
                sw = True; np_pos = ti; streak = 0; reason = ">3%%切换"
            elif hd >= cfg.min_hold and gap >= cfg.min_gap and met:
                sw = True; np_pos = ti; streak = 0; reason = "正常切换"
        elif (cfg.accel_enabled and pos >= 0
                and not np.isnan(accel_th_mat[pos, t])
                and mom10_mat[pos, t] > accel_th_mat[pos, t]):
            sw = True; np_pos = -1; streak = 0; reason = "加速止盈"
            accel_exit = True

        if sw:
            cost = cfg.commission + cfg.slippage
            if pos >= 0: nav *= (1 - cost)
            if np_pos >= 0: nav *= (1 - cost); hs = t; hn = nav
            old_pos = pos; pos = np_pos
            if accel_exit and old_pos >= 0 and cfg.accel_cooldown > 0:
                accel_block_idx = old_pos
                accel_block_until = t + cfg.accel_cooldown
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
