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
    stop_sigma: float = 0.0        # 自适应止损: 止损线 = -k×持仓自身20日波动率(不同标的不同线; 0=关闭, 波动率缺失回退固定stop_loss)
    stop_sigma_floor: bool = False # 自适应止损仅放宽: 止损线 = min(stop_loss, -kσ) (固定线为下限, 只对高波动标的放宽; 需配合 stop_sigma>0)
    entry_mom_cap: float = 0.0    # 入场动量耗尽门: 空仓入场时候选20日动量>X%视为耗尽不买 (0=关闭; 恐慌抄底豁免)
    entry_decel_on: bool = False  # 入场减速门: mom10<mom20 视为动量减速, 空仓不买 (恐慌抄底豁免)
    entry_bias_cap: float = 0.0   # 入场乖离门: 候选现价高于20日均线X%视为超买不买 (0=关闭; 恐慌抄底豁免)
    vol_scale_target: float = 0.0 # 波动率缩放仓位: 仓位=min(1, 目标年化波动/持仓年化已实现波动), 只降不升 (0=关闭; 恐慌抄底锁仓期豁免)
    trend_ma: int = 0             # 资产级长均线趋势门: 候选价须高于自身MA(N)才买入 (0=关闭)
    trend_ma_exempt: bool = False # 趋势门豁免纳指/黄金 (长期趋势资产不误杀)
    rank_sticky: int = 0          # 排名缓冲: 持仓仍在前N名时不切换 (0=关闭, 1=默认行为; 止损/空仓/止盈不受影响)
    cash_park_code: str = ""      # 空仓停泊标的代码 (如 511010.SH 5年国债 / 511880.SH 银华日利货基; 空=关; 不入排名池, 买卖各计一次手续费)
    exception_gap: float = 3.0     # 例外动量差
    exception_gap_max: float = 0.0   # 快速通道上限: gap≥此值不走快速通道改走慢通道 (0=关闭; 大gap=过度延伸易反转)
    exception_gap_sigma: float = 0.0 # 快速通道σ自适应: 门槛=max(exception_gap, k×滚动60日gapσ) (0=关闭)
    fast_lane_staged: bool = False  # 快速通道分档切换: gap∈[exception_gap, fast_full_gap) 切50%成核心+卫星组合, ≥fast_full_gap 切100% (0=关闭=旧行为; 仅Top-1路径)
    fast_full_gap: float = 5.0      # 分档切换全切档位 (仅 fast_lane_staged=True 时生效)
    vix_gate: float = 0.0            # VIX恐慌门: VIX>此值时禁止新买/切换纳指 (0=关闭; 美股侧恐慌保护, 2026-08-18 研究)
    vix_spike_exit: float = 0.0      # VIX异动离场: VIX单日涨幅>=此%时, 持纳指立即离场 (0=关闭)
    nz_mom_qqq_on: bool = True       # 纳指动量混入QQQ: (1-w)×513100 + w×QQQ-CNY (默认开, 2026-08-18 用户决议; 数据准备层消费)
    nz_mom_qqq_w: float = 0.7        # QQQ 权重 (0.7 = 70% QQQ + 30% ETF)
    skew_gate: float = 0.0           # SKEW尾部风险门: SKEW>此值时禁止新买/切换纳指 (0=关闭; I-7: 150 三窗口正)
    fast_lane_ma: int = 0            # 快速通道追涨闸门: 挑战者现价须高于自身MA(N)才走快速通道 (0=关闭; 2026-08-17 消融为负优化默认关: 被拦切换恰是历史最优档83%胜率/均盈+4.5%; 被拦时落回慢通道=min_hold后正常切换)
    commission: float = 0.0001     # 手续费率(单边)
    slippage: float = 0.0          # 滑点/冲击成本(单边)
    accel_sigma: float = 3.0       # 加速止盈 σ
    accel_enabled: bool = True     # 加速止盈开关
    accel_exempt_names: str = ""    # 免加速止盈的标的(逗号分隔, 如 创业): 高波动标的的mom10尖峰是常态, 止盈=卖飞 (2026-08-18 研究)
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
    accel_cooldown_gold_only: bool = False  # 止盈冷却仅对黄金生效 (创业的止盈后买回历史上是正收益, 不应误伤)
    accel_reentry_gate: bool = False  # 止盈再入场闸门: 加速止盈后须 mom10 回落到自身止盈阈值下方才允许买回 (0=关闭; 2026-08-18 研究, 打断黄金blow-off循环)
    stop_cooldown: int = 0         # 止损/绝对止损后N天禁止买回同一标的 (0=关闭, 恐慌抄底豁免; 按标的独立计时)
    sh_cash_cooldown: int = 3      # 上证空仓后再入场冷却天数 (0=关闭, 恐慌抄底豁免; 2026-08 起默认3天)
    sh_cash_exempt: bool = False   # 上证空仓仅对A股标的生效 (豁免纳指/黄金)
    multi_mom: bool = False        # 多周期动量: 排名分 = 3×mom10+2×mom20+1×mom60
    fip_tiebreak: bool = False     # FIP动量质量: 动量差<min_gap时选上涨路径更平滑者
    abs_stop_loss: bool = True     # 绝对止损: 持仓仍动量第一且pnl<=stop_loss时清到空仓(补盲区, 2026-08 起默认开)
    crash_alloc: float = 1.0       # 恐慌抄底切换仓位比例 (0~1): 已持其他标的时切此比例到创业; 空仓则强制全仓
    top_k: int = 1                 # 持仓数: 1=Top-1全仓(原行为); >1=Top-K等权轮动(独立引擎)
    mom_skip: int = 0              # 排名动量跳过最近N天(短期反转规避), 0=关闭
    confirm_bind_identity: bool = False  # 确认计数绑定候选身份(B1修复): 同一挑战者连续N天第一才计数;
                                         # False=旧语义(ti!=pos即累计, 每日轮换第一名也会计满)
    fast_min_hold: int = 0         # 快速通道(>3%切换)最短持有天数: 持仓不足N天不被例外动量差切走
                                   # (0=关闭=旧行为; 止损保护不受影响; E2候选开关, 先消融)


# 防御资产池: 空仓/防御信号触发时的候选持有标的
DEFENSIVE_NAMES = {"黄金", "红利低波", "5年国债"}

# 上证空仓豁免名单: 与A股低相关的资产, 豁免模式下不受上证空仓过滤
SH_CASH_EXEMPT_NAMES = {"纳指", "黄金"}


def ols_momentum(arr: np.ndarray, w: int) -> np.ndarray:
    """OLS 斜率动量 (与 stock-bot 生产 cron 同口径):
    对过去 w 天收盘价做最小二乘线性回归, 归一化为 % 动量。
    slope = Σ(x-x̄)(y-ȳ)/Σ(x-x̄)², momentum = slope × w / mean(y) × 100
    抗端点毛刺, 与价格比动量(price[t]/price[t-w]-1)互为可选口径。
    """
    n = len(arr)
    x = np.arange(w, dtype=float)
    xm = x - x.mean()
    denom = float((xm * xm).sum())

    def _f(y):
        ym = y.mean()
        if ym <= 0 or np.isnan(ym):
            return np.nan
        return (xm * y).sum() / denom * w / ym * 100.0

    return pd.Series(arr).rolling(w).apply(_f, raw=True).values


# ── 派生矩阵 (可在全量历史上预计算后切片传入, 消除回测窗口起点的冷启动空白) ──
def calc_accel_thresholds(mom10_mat: np.ndarray, cfg: "SignalConfig") -> np.ndarray:
    """加速止盈阈值矩阵。expanding 逐日计算, shift(1) 保证 t 日阈值只用 t-1 及之前的数据
    (与恐慌阈值 calc_threshold 的 arr[:t] 口径一致, 修复原全样本计算的前视偏差)"""
    accel_th_mat = np.full(mom10_mat.shape, np.nan)
    for i in range(mom10_mat.shape[0]):
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
    return accel_th_mat


def calc_mom60(price_mat: np.ndarray) -> np.ndarray:
    """60日动量 (multi_mom 用)"""
    m = np.full(price_mat.shape, np.nan)
    m[:, 60:] = (price_mat[:, 60:] / price_mat[:, :-60] - 1) * 100
    return m


def calc_vol20(price_mat: np.ndarray) -> np.ndarray:
    """20日日收益波动率 (risk_adj_mom 用)"""
    vol = np.full(price_mat.shape, np.nan)
    for i in range(price_mat.shape[0]):
        vol[i] = pd.Series(price_mat[i]).pct_change().rolling(20).std().values
    return vol


def calc_up_ratio20(price_mat: np.ndarray) -> np.ndarray:
    """20日上涨天数占比 (FIP 动量质量用)"""
    up = np.full(price_mat.shape, np.nan)
    for i in range(price_mat.shape[0]):
        up[i] = (pd.Series(price_mat[i]).pct_change() > 0).rolling(20).mean().values
    return up


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
    ma200_cyb: np.ndarray = None,
    ma200_sh: np.ndarray = None,
    pre_days: int = 0,
    pre_mats: dict = None,
) -> tuple:
    """Top-K 等权轮动引擎 (top_k>1 时独立于 Top-1 路径)

    持有动量得分最强的 K 个标的, 每日再平衡为等权, 空余仓位持现金。
    复用上证空仓/恐慌抄底/止损/加速止盈/确认入场规则;
    不支持防御切换/豁免/多周期/风险调整/FIP (均实测负优化, 不与 Top-K 组合)。
    返回与 Top-1 相同的 (actions, nav, nav_log)。
    """
    n_etfs = len(etf_names)
    K = min(cfg.top_k, n_etfs)

    # ── 加速止盈阈值 (优先用全量历史预计算值, 防窗口起点冷启动) ──
    accel_th_mat = (pre_mats or {}).get("accel_th")
    if accel_th_mat is None:
        accel_th_mat = calc_accel_thresholds(mom10_mat, cfg)

    # ── 20日波动率 (自适应止损用) ──
    vol_mat = None
    if cfg.stop_sigma > 0:
        vol_mat = (pre_mats or {}).get("vol")
        if vol_mat is None:
            vol_mat = calc_vol20(price_mat)

    # ── 快速通道追涨闸门均线 (与 Top-1 同口径) ──
    ma_gate_mat = None
    if cfg.fast_lane_ma > 0:
        ma_gate_mat = (pre_mats or {}).get("ma_fast")
        if ma_gate_mat is None:
            ma_gate_mat = np.full(price_mat.shape, np.nan)
            for _i in range(price_mat.shape[0]):
                ma_gate_mat[_i] = pd.Series(price_mat[_i]).rolling(cfg.fast_lane_ma).mean().values

    def stop_line(i, t):
        """止损线: stop_sigma>0 时 = -k×标的自身20日波动率; floor 时固定线为下限(只放宽不收紧); 缺失回退固定 stop_loss"""
        if cfg.stop_sigma > 0 and vol_mat is not None and not np.isnan(vol_mat[i, t]) and vol_mat[i, t] > 0:
            s = -cfg.stop_sigma * vol_mat[i, t] * 100.0
            return min(cfg.stop_loss, s) if cfg.stop_sigma_floor else s
        return cfg.stop_loss

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
    _nf = len(cyb_idx_p)
    mom10_cyb = np.full(_nf, np.nan)
    mom10_sh_idx = np.full(_nf, np.nan)
    mom10_cyb[sw_:] = (cyb_idx_p[sw_:] / cyb_idx_p[:-sw_] - 1) * 100
    mom10_sh_idx[sw_:] = (sh_idx_p[sw_:] / sh_idx_p[:-sw_] - 1) * 100
    ma200_cyb = pd.Series(cyb_idx_p).rolling(200).mean().values if ma200_cyb is None else ma200_cyb
    ma200_sh = pd.Series(sh_idx_p).rolling(200).mean().values if ma200_sh is None else ma200_sh

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
    stop_block_until = np.full(n_etfs, -1)   # 止损冷却状态 (按标的独立计时)
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
            nav_log.append(nav)     # 无数据日也记录, 保持 nav_log 与日期逐日对齐
            continue

        score = np.where(vm, ms, np.nan)
        if accel_block_idx >= 0 and t < accel_block_until:
            score[accel_block_idx] = np.nan
        if cfg.stop_cooldown > 0:
            score[t < stop_block_until] = np.nan   # 止损冷却期内移出候选
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
        # 用完整历史索引 tf = t + pre_days 计算阈值/MA200, 短回测区间不再被静默禁用
        crash_now = False
        tf = t + pre_days
        th_sh = calc_threshold(mom10_sh_idx, tf, cfg.crash_sigma)
        if th_sh is not None and mom10_sh_idx[tf] < th_sh and sh_idx_p[tf] < ma200_sh[tf]:
            crash_now = True
        if not crash_now:
            th_cyb = calc_threshold(mom10_cyb, tf, cfg.crash_sigma)
            if th_cyb is not None and mom10_cyb[tf] < th_cyb and cyb_idx_p[tf] < ma200_cyb[tf]:
                crash_now = True

        held_idx = [i for i in range(n_etfs) if w[i] > 1e-9]
        buys = []; sells = []
        cyb_pool_idx = etf_names.index("创业") if "创业" in etf_names else -1

        if cfg.crash_enabled and crash_now and t >= crash_held_until and cyb_pool_idx >= 0:
            if cyb_pool_idx not in held_idx:
                crash_held_until = t + cfg.crash_lock
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
                if t >= crash_held_until and pnl <= stop_line(i, t):
                    reason = "止损"
                elif (not np.isnan(accel_th_mat[i, t])
                        and mom10_mat[i, t] > accel_th_mat[i, t]):
                    reason = "加速止盈"
                elif t >= crash_held_until and streak >= cfg.confirm_days and not np.isnan(score[i]):
                    j = next((x for x in target if x not in held_idx and x not in bought), None)
                    if j is not None:
                        gap = score[j] - score[i]
                        if (gap >= cfg.exception_gap and hd >= cfg.fast_min_hold
                                and (cfg.exception_gap_max <= 0 or gap < cfg.exception_gap_max)
                                and (cfg.fast_lane_ma <= 0 or np.isnan(ma_gate_mat[j, t])
                                     or price_mat[j, t] > ma_gate_mat[j, t])):
                            reason = ">3%切换"
                        elif hd >= cfg.min_hold and gap >= cfg.min_gap:
                            reason = "正常切换"
                if reason is None: continue
                sells.append((i, reason)); held_idx.remove(i)
                if reason == "加速止盈" and cfg.accel_cooldown > 0:
                    accel_block_idx = i; accel_block_until = t + cfg.accel_cooldown
                if reason == "止损" and cfg.stop_cooldown > 0:
                    stop_block_until[i] = t + cfg.stop_cooldown
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
                        buys.append((j, ">3%即买" if strong else "确认买入"))
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
    ma200_cyb: np.ndarray = None,   # 预计算的创业板MA200 (完整历史, 短回测防NaN)
    ma200_sh: np.ndarray = None,    # 预计算的上证MA200
    pre_days: int = 0,              # cyb_idx_p/sh_idx_p 相对决策区起点的预热天数偏移
    pre_mats: dict = None,          # 全量历史预计算的派生矩阵(已切片到决策区):
                                    #   accel_th / mom60 / vol / up_ratio, 缺省则内部冷启动计算
) -> list:
    """
    返回 [(day_idx, action_type, target_etf_idx, reason), ...]
    action_type: 'BUY'|'SELL'|'CASH'|'HOLD'
    target_etf_idx: 0..n_etfs-1 or -1 for cash
    """

    if cfg.top_k > 1:
        return _compute_signals_topk(cfg, etf_names, price_mat, mom20_mat,
                                     mom10_mat, sh_mom, cyb_idx_p, sh_idx_p, n_days,
                                     ma200_cyb=ma200_cyb, ma200_sh=ma200_sh,
                                     pre_days=pre_days, pre_mats=pre_mats)

    # ── 加速止盈阈值 (优先用全量历史预计算值; 内部冷启动仅作兜底,
    #    冷启动时窗口前60天阈值NaN即加速止盈静默失效, 短回测尤甚) ──
    n_etfs = len(etf_names)
    accel_th_mat = (pre_mats or {}).get("accel_th")
    if accel_th_mat is None:
        accel_th_mat = calc_accel_thresholds(mom10_mat, cfg)

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
    # cyb_idx_p/sh_idx_p 为完整历史序列(含预热), 决策时用 tf = t + pre_days 访问,
    # 保证短回测区间下阈值/MA200 依然用足量历史计算
    sw = 10
    _nf = len(cyb_idx_p)
    mom10_cyb = np.full(_nf, np.nan)
    mom10_sh_idx = np.full(_nf, np.nan)
    mom10_cyb[sw:] = (cyb_idx_p[sw:] / cyb_idx_p[:-sw] - 1) * 100
    mom10_sh_idx[sw:] = (sh_idx_p[sw:] / sh_idx_p[:-sw] - 1) * 100

    ma200_cyb = pd.Series(cyb_idx_p).rolling(200).mean().values if ma200_cyb is None else ma200_cyb
    ma200_sh = pd.Series(sh_idx_p).rolling(200).mean().values if ma200_sh is None else ma200_sh

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
    prev_ti = -1                     # B1 修复用: 上一日第一名身份
    gap_hist = []                    # 快速通道σ自适应: 每日动量领先差历史(最多60天)
    crash_held_until = -1
    accel_block_idx = -1; accel_block_until = -1   # 加速止盈冷却状态
    accel_gate = np.zeros(n_etfs, dtype=bool)       # 止盈再入场闸门 (按标的, mom10回落解除)
    sh_cash_block_until = -1                        # 空仓再入场冷却状态
    stop_block_until = np.full(n_etfs, -1)          # 止损冷却状态 (按标的独立计时)
    crash_alloc_now = 0.0                           # 恐慌抄底当前仓位比例 (0=无抄底叠加仓)
    crash_base_pos = -1                             # 抄底前的名义主仓 (用于组合净值计算; -1=空仓抄底)
    sat_pos = -1; sat_alloc = 0.0; sat_hs = -1      # 分档切换组合: 卫星腿(-1=无), 权重与入场日
    actions = []
    nav_log = []

    def _leg_pnl(i, entry_day, t):
        """组合仓单腿价格口径盈亏% (无入场日/价格缺失返回0)"""
        p0 = price_mat[i, max(0, entry_day)] if entry_day >= 0 else np.nan
        p1 = price_mat[i, t]
        if not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
            return (p1 / p0 - 1) * 100
        return 0.0

    def _resolve_combo_to(survivor, entry_day, t, nav_now):
        """组合归并到单腿后重定基 hn: 使后续 pnl=幸存腿自身价格收益 (nav 须为扣费后的当前值)"""
        if entry_day >= 0:
            p0 = price_mat[survivor, max(0, entry_day)]
            p1 = price_mat[survivor, t]
            if not np.isnan(p0) and not np.isnan(p1) and p0 > 0 and p1 > 0:
                return nav_now / max(p1 / p0, 1e-9)
        return nav_now

    def _def_target(ms, vm):
        """防御资产中动量最强且为正者, 无则 -1 (防御资产也走绝对动量)"""
        best, best_m = -1, 0.0
        for i, nm in enumerate(etf_names):
            if nm in DEFENSIVE_NAMES and vm[i] and ms[i] > best_m:
                best, best_m = i, ms[i]
        return best

    # ── 20日波动率 (风险调整动量 / 自适应止损 / 波动率缩放仓位 使用) ──
    vol_mat = np.full(price_mat.shape, np.nan)
    if cfg.risk_adj_mom or cfg.stop_sigma > 0 or cfg.vol_scale_target > 0:
        vol_mat = (pre_mats or {}).get("vol")
        if vol_mat is None:
            vol_mat = calc_vol20(price_mat)

    # ── 空仓停泊标的序列 (货基/国债; 不入排名池) ──
    bond_arr = (pre_mats or {}).get("park") if cfg.cash_park_code else None

    # ── 资产级长均线 (趋势门使用; 优先用全量历史预计算值防冷启动) ──
    ma_mat = None
    if cfg.trend_ma > 0:
        ma_mat = (pre_mats or {}).get("ma")
        if ma_mat is None:
            ma_full = np.full(price_mat.shape, np.nan)
            for _i in range(price_mat.shape[0]):
                ma_full[_i] = pd.Series(price_mat[_i]).rolling(cfg.trend_ma).mean().values
            ma_mat = ma_full

    # ── VIX/SKEW 恐慌序列 (pre_mats 传入, 已按决策区对齐且无前视) ──
    vix_arr = (pre_mats or {}).get("vix")
    if vix_arr is not None:
        vix_arr = np.asarray(vix_arr, dtype=float)
    skew_arr = (pre_mats or {}).get("skew")
    if skew_arr is not None:
        skew_arr = np.asarray(skew_arr, dtype=float)

    def _panic_block(i, t, arr, gate):
        """恐慌门: 序列值>gate 时禁止买/切该标的"""
        return bool(gate > 0 and arr is not None and not np.isnan(arr[t]) and arr[t] > gate)

    # ── 快速通道追涨闸门均线 (挑战者须高于自身MA(N)才走快速通道; 优先用全量历史预计算值防冷启动,
    #    内部冷启动兜底时窗口前N-1天为NaN→闸门放行, 仅影响极短回测窗口起点) ──
    ma_gate_mat = None
    if cfg.fast_lane_ma > 0:
        ma_gate_mat = (pre_mats or {}).get("ma_fast")
        if ma_gate_mat is None:
            ma_gate_mat = np.full(price_mat.shape, np.nan)
            for _i in range(price_mat.shape[0]):
                ma_gate_mat[_i] = pd.Series(price_mat[_i]).rolling(cfg.fast_lane_ma).mean().values

    def stop_line(i, t):
        """止损线: stop_sigma>0 时 = -k×标的自身20日波动率(不同标的不同线);
        stop_sigma_floor=True 时固定线为下限(只放宽不收紧);
        波动率缺失/非正时回退固定 stop_loss"""
        if cfg.stop_sigma > 0 and not np.isnan(vol_mat[i, t]) and vol_mat[i, t] > 0:
            s = -cfg.stop_sigma * vol_mat[i, t] * 100.0
            return min(cfg.stop_loss, s) if cfg.stop_sigma_floor else s
        return cfg.stop_loss

    def entry_blocked(i, t):
        """入场动量耗尽门 (空仓入场用, 恐慌抄底/切换买入豁免):
        entry_mom_cap: 候选20日动量>X% 视为耗尽; entry_decel_on: mom10<mom20 视为减速;
        entry_bias_cap: 候选现价高于20日均线X% 视为超买"""
        if not np.isnan(ms[i]):
            if cfg.entry_mom_cap > 0 and ms[i] > cfg.entry_mom_cap:
                return True
            if (cfg.entry_decel_on and not np.isnan(mom10_mat[i, t])
                    and mom10_mat[i, t] < ms[i]):
                return True
        if cfg.entry_bias_cap > 0:
            px = price_mat[i, t]
            w = price_mat[i, max(0, t - 19):t + 1]
            wv = w[~np.isnan(w)]
            if len(wv) >= 10 and not np.isnan(px) and px > 0:
                ma = float(np.mean(wv))
                if ma > 0 and (px / ma - 1) * 100 > cfg.entry_bias_cap:
                    return True
        return False

    def trend_ok(i, t):
        """资产级趋势门: 候选须高于自身MA(trend_ma); 豁免标的/数据缺失时不拦"""
        if ma_mat is None or np.isnan(ma_mat[i, t]):
            return True
        if cfg.trend_ma_exempt and etf_names[i] in SH_CASH_EXEMPT_NAMES:
            return True
        px = price_mat[i, t]
        return not np.isnan(px) and px > ma_mat[i, t]

    def vol_scale(i, t):
        """波动率缩放: min(1, 目标年化波动/持仓年化已实现波动), 只降不升"""
        v = vol_mat[i, t]
        if np.isnan(v) or v <= 0:
            return 1.0
        return min(1.0, cfg.vol_scale_target / (v * np.sqrt(244)))

    # ── 60日动量 (仅多周期动量开启时使用) ──
    mom60_mat = None
    if cfg.multi_mom:
        mom60_mat = (pre_mats or {}).get("mom60")
        if mom60_mat is None:
            mom60_mat = calc_mom60(price_mat)

    # ── 20日上涨天数占比 (仅FIP开启时使用) ──
    up_ratio = None
    if cfg.fip_tiebreak:
        up_ratio = (pre_mats or {}).get("up_ratio")
        if up_ratio is None:
            up_ratio = calc_up_ratio20(price_mat)

    for t in range(vs, n_days):
        # 更新净值（仅用于计算PnL做决策）
        if crash_alloc_now > 0:
            # 组合持仓: 主仓 crash_base_pos (权重 1-alloc) + 创业 (权重 alloc)
            # 空仓抄底时 base_pos=-1, 即全仓创业
            day_ret = 0.0
            if crash_base_pos >= 0:
                b0 = price_mat[crash_base_pos, max(0, t-1)]
                b1 = price_mat[crash_base_pos, t]
                if not np.isnan(b0) and not np.isnan(b1) and b0 > 0:
                    day_ret += (1 - crash_alloc_now) * (b1 / b0 - 1)
            c0 = price_mat[cyb_pool_idx, max(0, t-1)]
            c1 = price_mat[cyb_pool_idx, t]
            if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                day_ret += crash_alloc_now * (c1 / c0 - 1)
            nav *= (1 + day_ret)
        elif sat_pos >= 0:
            # 分档切换组合: 核心(1-alloc)+卫星(alloc) 按定权重计息
            day_ret = 0.0
            b0 = price_mat[pos, max(0, t-1)]; b1 = price_mat[pos, t]
            if not np.isnan(b0) and not np.isnan(b1) and b0 > 0:
                day_ret += (1 - sat_alloc) * (b1 / b0 - 1)
            s0 = price_mat[sat_pos, max(0, t-1)]; s1 = price_mat[sat_pos, t]
            if not np.isnan(s0) and not np.isnan(s1) and s0 > 0:
                day_ret += sat_alloc * (s1 / s0 - 1)
            nav *= (1 + day_ret)
        elif pos >= 0:
            prev_p = price_mat[pos, max(0, t-1)]
            curr_p = price_mat[pos, t]
            if not np.isnan(prev_p) and not np.isnan(curr_p) and prev_p > 0:
                r = curr_p / prev_p - 1
                if cfg.vol_scale_target > 0 and not t < crash_held_until:
                    nav *= 1 + vol_scale(pos, t) * r   # 波动率缩放: 超额部分持现金(抄底锁仓期豁免)
                else:
                    nav *= 1 + r
        elif cfg.cash_park_code and bond_arr is not None:
            pb = bond_arr[max(0, t-1)]; cb = bond_arr[t]
            if not np.isnan(pb) and not np.isnan(cb) and pb > 0:
                nav *= cb / pb     # 空仓停泊: 停泊标的日收益计入净值

        hd = t - hs if pos >= 0 else 0
        pnl = (nav / hn - 1) * 100 if pos >= 0 and hn > 0 else 0
        ms = mom20_mat[:, t]; vm = ~np.isnan(ms)

        if not vm.any():
            pp = price_mat[pos, t] if pos >= 0 else np.nan
            nav_log.append(nav)     # 无数据日也记录, 保持 nav_log 与日期逐日对齐
            continue
        if np.isnan(sh_mom[t]):
            pp = price_mat[pos, t] if pos >= 0 else np.nan
            nav_log.append(nav)     # 同上
            continue

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

        # ── 止损冷却: N天内把被止损标的移出候选 (恐慌抄底豁免) ──
        if cfg.stop_cooldown > 0:
            sb = t < stop_block_until
            if sb.any():
                score = np.where(sb, np.nan, score)

        # ── 止盈再入场闸门: mom10 回落到止盈阈值下方自动解除; 未解除者移出候选 ──
        if cfg.accel_reentry_gate and accel_gate.any():
            for _i in np.where(accel_gate)[0]:
                _th = accel_th_mat[_i, t]
                if np.isnan(_th) or (not np.isnan(mom10_mat[_i, t]) and mom10_mat[_i, t] < _th):
                    accel_gate[_i] = False
            if accel_gate.any():
                score = np.where(accel_gate, np.nan, score)

        sv = ~np.isnan(score)
        ti = int(np.argmax(np.where(sv, score, -np.inf))) if sv.any() else ti_raw
        # 候选全被冷却时 ti 回退为原始动量第一(本身在冷却期): 禁止向其切换/买入,
        # 止损/绝对止损等退出保护不受影响
        ti_blocked = (cfg.stop_cooldown > 0 and ti >= 0
                      and t < stop_block_until[ti])

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

        if ti != pos:
            if cfg.confirm_bind_identity:
                # B1 修复: 只有"同一挑战者连续第一"才累计; 第一名换人立即归零重计
                streak = streak + 1 if ti == prev_ti else 1
            else:
                streak += 1
        else: streak = 0
        prev_ti = ti

        # ── 快速通道σ自适应: 记录每日动量领先差, 滚动σ=其噪声水平 ──
        if cfg.exception_gap_sigma > 0 and pos >= 0 and not np.isnan(ms[pos]):
            g = ms[ti] - ms[pos]
            if not np.isnan(g):
                gap_hist.append(g)
                if len(gap_hist) > 60: gap_hist.pop(0)

        # ── 恐慌抄底 ──
        # 用完整历史索引 tf = t + pre_days 计算阈值/MA200, 短回测区间不再被静默禁用
        crash_now = False
        tf = t + pre_days
        th_sh = calc_threshold(mom10_sh_idx, tf, cfg.crash_sigma)
        if th_sh is not None and mom10_sh_idx[tf] < th_sh and sh_idx_p[tf] < ma200_sh[tf]:
            crash_now = True
        if not crash_now:
            th_cyb = calc_threshold(mom10_cyb, tf, cfg.crash_sigma)
            if th_cyb is not None and mom10_cyb[tf] < th_cyb and cyb_idx_p[tf] < ma200_cyb[tf]:
                crash_now = True

        cyb_pool_idx = etf_names.index("创业") if "创业" in etf_names else -1
        if (cfg.crash_enabled and crash_now and t >= crash_held_until
                and cyb_pool_idx >= 0):
            if sat_pos >= 0:
                # 分档组合被恐慌抄底接管: 先清组合双腿, 再走标准抄底入场
                nav *= (1 - cfg.commission - cfg.slippage)
                actions.append((t, pos, -1, "恐慌换仓"))
                pos = -1; sat_pos = -1; sat_alloc = 0.0; sat_hs = -1
            if pos != cyb_pool_idx:
                crash_held_until = t + cfg.crash_lock
                alloc = min(max(float(cfg.crash_alloc), 0.0), 1.0)
                cost = cfg.commission + cfg.slippage
                if pos == -1:
                    nav *= (1 - cost * alloc)              # 空仓按 alloc 比例买入
                    if cfg.cash_park_code: nav *= (1 - cost) # 空仓停泊: 卖出停泊标的计费
                else:
                    nav *= (1 - cost * alloc) ** 2       # 卖出base的alloc份+买入创业: 双边按成交额
                crash_alloc_now = alloc
                crash_base_pos = pos
                pos = cyb_pool_idx
                hs = t; hn = nav
                streak = 0
                actions.append((t, crash_base_pos, cyb_pool_idx, "恐慌抄底"))
            # 已持创业: 仅刷新锁仓期, 不重复交易
        elif t < crash_held_until:
            # ── 抄底锁仓期: 只允许止损保护退出, 其他动作(轮动/空仓/止盈/动量门)全部锁住 ──
            # 符合"抄底=认准反弹拿住"的设计: 期间不被动量波动切走, 也不被上证空仓误伤
            if pos >= 0 and pnl <= stop_line(pos, t):
                np_pos = ti if (ti != pos and not ti_blocked) else -1   # 有更强候选切过去, 否则清仓
                sw = True; streak = 0; reason = "止损"
        elif sc:
            if cfg.sh_cash_exempt and pos >= 0 and etf_names[pos] in SH_CASH_EXEMPT_NAMES:
                pass  # 持有豁免标的(纳指/黄金), 上证空仓不出场
            elif cfg.sh_cash_exempt and pos == -1:
                # 空仓时仍可按正常入场规则买豁免标的
                ex_idx = [i for i, nm in enumerate(etf_names)
                          if nm in SH_CASH_EXEMPT_NAMES and vm[i]]
                bought = False
                if ex_idx:
                    if cfg.accel_reentry_gate:
                        ex_idx = [i for i in ex_idx if not accel_gate[i]]
                    if ex_idx:
                        ei = max(ex_idx, key=lambda i: ms[i])
                        lm = ms[ei]
                        if (not np.isnan(lm) and not entry_blocked(ei, t) and trend_ok(ei, t)
                                and (lm >= cfg.cash_skip_gt or streak >= cfg.confirm_days)):
                            sw = True; np_pos = ei; streak = 0
                            reason = ">3%即买(豁免)" if lm >= cfg.cash_skip_gt else "确认买入(豁免)"
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
        elif (cfg.vix_spike_exit > 0 and pos >= 0 and etf_names[pos] == "纳指"
                and t >= crash_held_until and vix_arr is not None
                and not np.isnan(vix_arr[t]) and not np.isnan(vix_arr[max(0, t - 1)])
                and vix_arr[max(0, t - 1)] > 0
                and (vix_arr[t] / vix_arr[max(0, t - 1)] - 1) * 100 >= cfg.vix_spike_exit):
            # VIX 异动离场: 美股恐慌单日飙升 → 清纳指到空仓
            sw = True; np_pos = -1; streak = 0; reason = "VIX异动"
        elif sat_pos >= 0:
            # ── 分档切换组合 (核心+卫星): 双腿风控 / 升级全切 / 卫星回撤 / 卫星追新 ──
            base, sat = pos, sat_pos
            entry_base, entry_sat = hs, sat_hs
            cost = cfg.commission + cfg.slippage
            g_sb = ms[sat] - ms[base]
            g_bs = ms[base] - ms[sat]
            hd_sat = t - sat_hs
            pb = _leg_pnl(base, entry_base, t)
            ps = _leg_pnl(sat, entry_sat, t)
            accel_b = (cfg.accel_enabled
                       and not np.isnan(accel_th_mat[base, t])
                       and mom10_mat[base, t] > accel_th_mat[base, t])
            accel_s = (cfg.accel_enabled
                       and not np.isnan(accel_th_mat[sat, t])
                       and mom10_mat[sat, t] > accel_th_mat[sat, t])
            stop_b = t >= crash_held_until and pb <= stop_line(base, t)
            stop_s = t >= crash_held_until and ps <= stop_line(sat, t)

            if stop_b and stop_s:
                nav *= (1 - cost)                      # 双腿全卖 → 空仓
                actions.append((t, base, -1, "组合止损"))
                pos = -1; sat_pos = -1; sat_alloc = 0.0; sat_hs = -1
                streak = 0
            elif stop_b:
                nav *= (1 - cost * sat_alloc)          # 卖核心半仓 → 全持卫星
                actions.append((t, base, sat, "核心止损"))
                pos = sat; sat_pos = -1; sat_alloc = 0.0; sat_hs = -1
                hn = _resolve_combo_to(sat, entry_sat, t, nav); hs = entry_sat
                streak = 0
            elif stop_s:
                nav *= (1 - cost * sat_alloc)          # 卖卫星半仓 → 全持核心
                actions.append((t, sat, base, "卫星止损"))
                sat_pos = -1; sat_alloc = 0.0; sat_hs = -1
                hn = _resolve_combo_to(base, entry_base, t, nav)
                streak = 0
            elif accel_b and accel_s:
                nav *= (1 - cost)                      # 双腿止盈 → 空仓
                actions.append((t, base, -1, "组合止盈"))
                pos = -1; sat_pos = -1; sat_alloc = 0.0; sat_hs = -1
                streak = 0
            elif accel_b:
                nav *= (1 - cost * sat_alloc)          # 核心止盈 → 全持卫星
                actions.append((t, base, sat, "核心止盈"))
                pos = sat; sat_pos = -1; sat_alloc = 0.0; sat_hs = -1
                hn = _resolve_combo_to(sat, entry_sat, t, nav); hs = entry_sat
                streak = 0
            elif accel_s:
                nav *= (1 - cost * sat_alloc)          # 卫星止盈 → 全持核心
                actions.append((t, sat, base, "卫星止盈"))
                sat_pos = -1; sat_alloc = 0.0; sat_hs = -1
                hn = _resolve_combo_to(base, entry_base, t, nav)
                streak = 0
            elif g_sb >= cfg.fast_full_gap:
                nav *= (1 - cost * sat_alloc) ** 2     # 卖核心半仓+买卫星半仓 → 全切卫星
                actions.append((t, base, sat, "分档升级全切"))
                pos = sat; sat_pos = -1; sat_alloc = 0.0; sat_hs = -1
                hn = _resolve_combo_to(sat, entry_sat, t, nav); hs = entry_sat
                streak = 0
            elif g_bs >= cfg.min_gap and hd_sat >= cfg.min_hold:
                nav *= (1 - cost * sat_alloc)          # 核心重新领先: 卖卫星半仓 → 全持核心
                actions.append((t, sat, base, "卫星回撤"))
                sat_pos = -1; sat_alloc = 0.0; sat_hs = -1
                hn = _resolve_combo_to(base, entry_base, t, nav)
                streak = 0
            elif (ti != base and ti != sat and not ti_blocked
                    and t >= crash_held_until and streak >= cfg.confirm_days):
                gap_ts = ms[ti] - ms[sat]
                fast_ts = gap_ts >= cfg.exception_gap
                if fast_ts and cfg.exception_gap_max > 0 and gap_ts >= cfg.exception_gap_max:
                    fast_ts = False
                if ((fast_ts and hd_sat >= cfg.fast_min_hold)
                        or (hd_sat >= cfg.min_hold and gap_ts >= cfg.min_gap)):
                    nav *= (1 - cost * sat_alloc) ** 2 # 卫星腿追新第一名
                    actions.append((t, sat, ti, "卫星追新"))
                    sat_pos = ti; sat_hs = t
                    streak = 0
        elif pos == -1:
            if t < sh_cash_block_until:
                pass  # 空仓冷却期内, 禁止再入场 (恐慌抄底走独立分支, 不受影响)
            elif ti_blocked:
                pass  # 唯一候选在止损冷却期, 不入场
            else:
                lm = ms[ti]
                if entry_blocked(ti, t):
                    pass  # 动量耗尽门: 候选已耗尽/减速/超买, 保持空仓等待
                elif not trend_ok(ti, t):
                    pass  # 趋势门: 候选低于自身长均线, 保持空仓
                elif (etf_names[ti] == "纳指" and (
                        _panic_block(ti, t, vix_arr, cfg.vix_gate)
                        or _panic_block(ti, t, skew_arr, cfg.skew_gate))):
                    pass  # VIX/SKEW恐慌门: 美股恐慌期不新买纳指
                elif lm >= cfg.cash_skip_gt and not np.isnan(lm):
                    sw = True; np_pos = ti; streak = 0; reason = ">3%即买"
                elif streak >= cfg.confirm_days:
                    sw = True; np_pos = ti; streak = 0; reason = "确认买入"
        elif ti != pos and not ti_blocked and t >= crash_held_until:
            # 注意: 止损只在存在更优候选(ti != pos)时检查——持仓仍是动量第一名时
            # 永不止损 (相对动量取舍: 全池齐跌且持仓最抗跌时无处可切, 保持持有)。
            # 这意味着实际亏损可能显著超过 stop_loss 线 (如 2026-06 纳指 -9.79%)。
            # 止损卖出后该标的进入 stop_cooldown 冷却, N天内不会被买回。
            gap = ms[ti] - ms[pos]; met = (streak >= cfg.confirm_days)
            # 快速通道门槛: 下限=max(固定3%, k×滚动gapσ), 上限=exception_gap_max
            # (大gap=被卖者超跌/挑战者超涨, 过度延伸易反转 → 不走快速通道改走慢通道)
            fast_gap = gap >= cfg.exception_gap
            if fast_gap and cfg.exception_gap_sigma > 0 and len(gap_hist) >= 20:
                fast_gap = gap >= max(cfg.exception_gap,
                                      cfg.exception_gap_sigma * float(np.nanstd(gap_hist)))
            if fast_gap and cfg.exception_gap_max > 0 and gap >= cfg.exception_gap_max:
                fast_gap = False
            if fast_gap and cfg.fast_lane_ma > 0:
                # 追涨闸门: 挑战者现价须高于自身MA(N), 下跌反弹中继不追 (被拦则落回慢通道)
                _px = price_mat[ti, t]
                _ma = ma_gate_mat[ti, t]
                if not np.isnan(_px) and not np.isnan(_ma) and _px <= _ma:
                    fast_gap = False
            if pnl <= stop_line(pos, t):
                sw = True; np_pos = ti; streak = 0; reason = "止损"
            elif cfg.rank_sticky > 0 and int(np.sum(ms > ms[pos])) + 1 <= cfg.rank_sticky:
                pass  # 排名缓冲: 持仓仍在前N名, 不切换
            elif not trend_ok(ti, t):
                pass  # 趋势门: 候选低于自身长均线, 不切换
            elif (etf_names[ti] == "纳指" and (
                    _panic_block(ti, t, vix_arr, cfg.vix_gate)
                    or _panic_block(ti, t, skew_arr, cfg.skew_gate))):
                pass  # VIX/SKEW恐慌门: 美股恐慌期不切向纳指
            elif fast_gap and met and hd >= cfg.fast_min_hold:
                if (cfg.fast_lane_staged and cfg.fast_full_gap > cfg.exception_gap
                        and gap < cfg.fast_full_gap):
                    # 分档切换: gap∈[exception_gap, fast_full_gap) 切50% → 核心+卫星组合
                    # (卫星=新第一名; 核心保留原仓, 双腿各自独立风控)
                    nav *= (1 - (cfg.commission + cfg.slippage) * 0.5) ** 2
                    actions.append((t, pos, ti, ">3%切换(50%)"))
                    sat_pos = ti; sat_alloc = 0.5; sat_hs = t
                    streak = 0
                else:
                    sw = True; np_pos = ti; streak = 0; reason = ">3%切换"
            elif hd >= cfg.min_hold and gap >= cfg.min_gap and met:
                sw = True; np_pos = ti; streak = 0; reason = "正常切换"
        elif (cfg.abs_stop_loss and pos >= 0 and pnl <= stop_line(pos, t)
                and t >= crash_held_until):
            # 绝对止损: 持仓仍是动量第一(ti==pos)时的盲区补丁——
            # 有更优候选时走上面的"止损"切换; 无候选可切时清到空仓,
            # 不再硬扛 (2026-06 纳指 -9.79% 远超 -5% 线的成因)
            sw = True; np_pos = -1; streak = 0; reason = "绝对止损"
        elif (cfg.accel_enabled and pos >= 0
                and etf_names[pos] not in {x.strip() for x in cfg.accel_exempt_names.split(",") if x.strip()}
                and not np.isnan(accel_th_mat[pos, t])
                and mom10_mat[pos, t] > accel_th_mat[pos, t]):
            sw = True; np_pos = -1; streak = 0; reason = "加速止盈"
            accel_exit = True

        if sw:
            cost = cfg.commission + cfg.slippage
            crash_alloc_now = 0.0   # 任何正常动作清理抄底叠加仓, 归并为新主仓全仓
            sat_pos = -1; sat_alloc = 0.0; sat_hs = -1   # 任何全仓动作清理分档组合双腿
            if pos >= 0: nav *= (1 - cost)
            elif cfg.cash_park_code: nav *= (1 - cost) # 空仓停泊: 卖出停泊标的计费
            if np_pos >= 0: nav *= (1 - cost); hs = t; hn = nav
            old_pos = pos; pos = np_pos
            if (accel_exit and old_pos >= 0 and cfg.accel_cooldown > 0
                    and (not cfg.accel_cooldown_gold_only or etf_names[old_pos] == "黄金")):
                accel_block_idx = old_pos
                accel_block_until = t + cfg.accel_cooldown
            if accel_exit and old_pos >= 0 and cfg.accel_reentry_gate:
                accel_gate[old_pos] = True
            if (reason in ("止损", "绝对止损") and old_pos >= 0
                    and cfg.stop_cooldown > 0):
                stop_block_until[old_pos] = t + cfg.stop_cooldown
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
    """从信号序列和价格矩阵重建净值。手续费已在compute_signals中处理, 此处不再重复扣除。
    注意: 不支持 fast_lane_staged 组合仓语义 (仅 backtest/runner.py 旧路径使用;
    研究口径一律以 compute_signals 返回的 nav_log 为准)。"""
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
