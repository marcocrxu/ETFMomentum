"""核心信号函数 — 回测和实盘共用同一套逻辑"""
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


def _rolling_mean_valid(arr: np.ndarray, w: int) -> np.ndarray:
    """按**该标的自身有效报价**计算 w 期均线, NaN 空洞不再向内扩散。

    口径: 第 i 位的均线 = 截至 i 的最后 w 个有效报价之均值 (i 有效时即通常
    rolling(w) 窗口)。无空洞时与 pd.Series(arr).rolling(w).mean() 逐点相同,
    仅在存在缺行/多行空洞时填补 NaN。"""
    a = np.asarray(arr, dtype=float)
    out = pd.Series(a).rolling(w).mean().values
    if not np.isnan(out).any():
        return out
    valid = ~np.isnan(a)
    if valid.sum() < w:
        return out
    vpos = np.flatnonzero(valid)
    cv = np.concatenate([[0.0], np.cumsum(a[vpos])])   # cv[k] = 前 k 个有效报价之和
    rank = np.cumsum(valid) - 1          # 每个位置"最后一个有效报价"的序号
    fill = np.full(len(a), np.nan)
    ok = rank >= w - 1
    r = rank[ok]
    fill[ok] = (cv[r + 1] - cv[r + 1 - w]) / w
    return np.where(np.isnan(out) & ~np.isnan(fill), fill, out)


def _ma_safe(price: np.ndarray, w: int, ma: np.ndarray, tag: str) -> np.ndarray:
    """长均线防空洞: MA 出现 NaN(行情缺行/多余行导致)时按有效报价回填并告警。

    此前 NaN MA 会让 `现价 < MA` 恒为 False, 静默否决恐慌抄底等依赖 MA 的分支
    (2026-10-15 事故: 000001.SH 一条周六脏行 → 创业板指 MA200 全 NaN 一个月,
    10-14 该出的恐慌抄底被吞掉)。"""
    base = pd.Series(price).rolling(w).mean().values if ma is None else np.asarray(ma, dtype=float)
    holes = np.isnan(base)
    if not holes.any():
        return base
    fill = _rolling_mean_valid(price, w)
    usable = holes & ~np.isnan(fill)
    if usable.any():
        _tail = np.flatnonzero(usable)[-1]
        warnings.warn("数据告警: %s MA%d 有 %d 处 NaN 空洞(行情缺行/多余行), "
                      "已按有效报价回填, 最近一处 index=%d; 请检查数据完整性"
                      % (tag, w, int(usable.sum()), int(_tail)), RuntimeWarning)
        return np.where(usable, fill, base)
    return base


def _roll_hilo(arr: np.ndarray, w: int):
    """(截至当日的) w 日最高/最低价 —— N-30 恐慌抄底高低点结构门。

    与 `_rolling_mean_valid` 同口径: 第 i 位的 HH/LL = 截至 i 的最后 w 个**有效报价**
    的最高/最低 (i 有效时即通常 rolling(w) 窗口)。无空洞时与 pandas rolling 逐点相同,
    仅在缺行/多行空洞时按有效报价回填, 避免 NaN 让闸门静默失效。
    """
    a = np.asarray(arr, dtype=float)
    hh = pd.Series(a).rolling(w).max().values
    ll = pd.Series(a).rolling(w).min().values
    if not (np.isnan(hh).any() or np.isnan(ll).any()):
        return hh, ll
    valid = ~np.isnan(a)
    if valid.sum() < w:
        return hh, ll
    vals = a[valid]
    hit = np.cumsum(valid) - 1                 # 每个位置"最后一个有效报价"的序号
    fill_h = np.full(len(a), np.nan)
    fill_l = np.full(len(a), np.nan)
    for i in np.flatnonzero(hit >= w - 1):
        k = hit[i]
        seg = vals[k - w + 1:k + 1]
        fill_h[i] = seg.max(); fill_l[i] = seg.min()
    return (np.where(np.isnan(hh) & ~np.isnan(fill_h), fill_h, hh),
            np.where(np.isnan(ll) & ~np.isnan(fill_l), fill_l, ll))


# 恐慌抄底"无法判定"事件: 每次 compute_signals 调用清空, 元素 (tf, label, 缺什么)。
# 实盘 runner 只取当天(t == n_days-1)的事件, 据此在推送正文里告警。
CRASH_DIAG = []


def _make_mom5_series(cfg, cyb_idx_p, sh_idx_p):
    """N-38: 生成 (tf)->(mom5_pct, mom5_z) 求值器所需的 5 日动量序列。

    口径约定 (与 mom10/MA200 一致, 全部无前视):
      - mom5[t] = px[t]/px[t-5]-1, 单位 %
      - z[t]    = mom5[t] / σ5[t], σ5 = mom5 **当日之前**的样本标准差 (排除第 t 日,
                  与 calc_threshold 的 arr[:t] 口径一致); 样本 <60 时 z 记 None
    返回 dict: {"cyb": fn, "sh": fn, "mom5_cyb": arr, "z_cyb": arr, ...} 供两个分支共用。
    """
    span = max(5, int(getattr(cfg, "crash_mom5_span", 5) or 5))

    def _mom5_z(px):
        px = np.asarray(px, dtype=float)
        n = len(px)
        m5 = np.full(n, np.nan)
        if n > span:
            m5[span:] = (px[span:] / px[:-span] - 1) * 100
        z = np.full(n, np.nan)
        s = pd.Series(m5)
        # shift(1): 只用截至 t-1 的样本估 σ, 第 t 日自身不参与, 避免同日后视
        sd = s.rolling(60, min_periods=60).std().shift(1).values
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where((~np.isnan(sd)) & (sd > 1e-12), m5 / sd, np.nan)
        return m5, z

    m5_c, z_c = _mom5_z(cyb_idx_p)
    m5_s, z_s = _mom5_z(sh_idx_p)

    def _mk(m5, z):
        def fn(tf):
            if tf >= len(m5) or np.isnan(m5[tf]):
                return None, None
            zv = None if (tf >= len(z) or np.isnan(z[tf])) else float(z[tf])
            return float(m5[tf]), zv
        return fn

    return dict(cyb=_mk(m5_c, z_c), sh=_mk(m5_s, z_s),
                mom5_cyb=m5_c, z_cyb=z_c, mom5_sh=m5_s, z_sh=z_s)


def _make_crash_eval(cfg, calc_threshold, diag, _moments=None):
    """恐慌抄底判定的**三态**求值器 (2026-10-15 事故防复发)。

    旧写法 `mom < th and px < ma` 在任一项为 NaN 时整体为 False, 与"条件不成立"
    完全无法区分 → 脏数据导致的失效看起来就是一次普通 HOLD, 异常/warning/FP flag
    全都不响 (NaN 比较在 IEEE 754 下合法且静默)。这里把三态显式化:
    返回 True=触发 / False=不触发 / None=**无法判定**(缺数据, 已记入 diag)。
    引擎对 None 的行为与旧版一致(不触发, 保证基线逐分不变), 但调用方可据此告警。

    N-38: cfg.crash_mom5_mode 非空时, 把「5 日跌幅」接进判据 (默认空 = 老行为逐分不变)。
    `_moments` 是 (apply(cfg)->evaluator) 的延迟绑定器: 6 日动量与 σ5 都需要完整价格序列,
    而 800 / 1400 两个 compute_signals 分支各自持有 cyb/sh 指数序列, 故按分支注入。
    """
    _hl_on = float(getattr(cfg, "crash_hl_win", 0) or 0) > 0
    _mode = str(getattr(cfg, "crash_mom5_mode", "") or "")
    _pct = float(getattr(cfg, "crash_mom5_pct", -8.0) or -8.0)
    _z = float(getattr(cfg, "crash_mom5_z", -2.0) or -2.0)
    _mom5_fn = None
    if _mode and _moments is not None:
        _mom5_fn = _moments(cfg)          # 分支内绑定: (tf)->(mom5%, mom5_z) 或 (None,None)
    elif _mode:
        # 调用方未注入序列 → 视为无法判定, 显式记 diag 而不是静默返回不触发
        _mom5_fn = lambda tf: (None, None)
    _NOM5 = "缺5日动量序列(N-38 未注入)"

    def _eval(mom10, ma200, px, tf, label, hl=None):
        if tf >= len(mom10) or tf >= len(px) or tf >= len(ma200):
            diag.append((tf, label, "索引越界"))
            return None
        mv, pv, av = mom10[tf], px[tf], ma200[tf]
        if not (np.isnan(mv) or np.isnan(pv) or np.isnan(av)):
            th = calc_threshold(mom10, tf, cfg.crash_sigma)
            if th is None:
                return None                      # 阈值样本<60: 预热期, 非异常
            # ── N-38 5 日跌幅接入 (AND 语义: 收紧) ──
            if _mode in ("add", "repl", "zs"):
                m5, m5z = _mom5_fn(tf, label)
                if m5 is None:
                    diag.append((tf, label, _NOM5))
                    return None
                if _mode == "add":
                    if not (mv < th and pv < av):
                        return False
                    if m5 > _pct:                # 5 日跌幅不够深
                        return False
                elif _mode == "repl":
                    if not (m5 <= _pct and pv < av):
                        return False
                elif _mode == "zs":
                    if m5z is None:
                        diag.append((tf, label, _NOM5))
                        return None
                    if not (m5z <= _z and pv < av):
                        return False
            # ── N-39 5 日跌幅 OR 接入 (放宽: 任一通道成立即抄底) ──
            elif _mode in ("or_deep", "or_ma", "or_z"):
                m5, m5z = _mom5_fn(tf, label)
                if m5 is None:
                    diag.append((tf, label, _NOM5))
                    return None
                _orig = bool(mv < th and pv < av)
                if _mode == "or_deep":
                    _alt = bool(m5 <= _pct and pv < av)
                elif _mode == "or_ma":
                    _alt = bool(m5 <= _pct)
                else:  # or_z
                    if m5z is None:
                        diag.append((tf, label, _NOM5))
                        return None
                    _alt = bool(m5z <= _z and pv < av)
                if not (_orig or _alt):
                    return False
            elif not (mv < th and pv < av):
                return False
            # ── N-30 高低点结构门 (闸门只在"动量+年线"已成立后才判定, 不改变原语义) ──
            if hl is None or not _hl_on:
                return True
            hh_a, ll_a = hl
            hh, ll = hh_a[tf], ll_a[tf]
            if np.isnan(hh) or np.isnan(ll):
                diag.append((tf, label, "缺%d日高低点" % int(cfg.crash_hl_win)))
                return None
            if getattr(cfg, "crash_dd_min", 0.0) > 0:
                dd = (pv / hh - 1) * 100 if hh > 0 else 0.0
                if dd > -abs(cfg.crash_dd_min):
                    return False                 # 回撤不够深
            if getattr(cfg, "crash_pos_max", 1.0) < 1.0:
                pos = (pv - ll) / (hh - ll) if hh > ll else 1.0
                if pos > cfg.crash_pos_max:
                    return False                 # 不在区间下沿
            if getattr(cfg, "crash_off_low_max", 0.0) > 0:
                off = (pv / ll - 1) * 100 if ll > 0 else 0.0
                if off > abs(cfg.crash_off_low_max):
                    return False                 # 离最低点太远
            return True
        # ── 含 NaN: 只在"本该有数据却缺"时记为异常 ──
        # 预热期(有效价不足 200 / 不足 10 日)属正常, 不记, 免得淹没有效告警
        missing = []
        valid_px = int(np.count_nonzero(~np.isnan(px[:tf + 1])))
        if np.isnan(pv):
            missing.append("指数价")
        if np.isnan(mv) and tf >= 10:
            missing.append("指数10日动量")
        if np.isnan(av) and valid_px >= 200:
            missing.append("MA200(近200日窗口内有空洞)")
        if missing:
            diag.append((tf, label, "缺" + "/".join(missing)))
        return None
    return _eval


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
    nz_vol_gate: float = 0.0      # 纳指放量门: 513100 量比(VOL20/VOL120)>X 时禁止【新买】纳指 (0=关闭; 只拦入场, 不强制卖出)
    nz_vol_win: int = 20          # 量比分子窗口 (交易日)
    nz_vol_base: int = 120        # 量比分母窗口 (交易日)
    trail_stop: float = 0.0       # 移动止损(吊灯): 现价较【本次入场以来最高收盘】回撤≥X% 即离场 (0=关闭; 清到空仓)
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
    rate_shock_gate: float = 0.0     # 美债利率冲击门: |10Y较20个美股交易日前的Δ|>此值(pp)时禁止新买/切换纳指 (0=关闭; K-3: 0.4 三窗口正候选, 2026-08-27 研究)
    share_gate: float = 0.0          # 份额风险门: 宽基篮子(510300/510500/510050)N日净申购率<-X%时
                                     # 禁止新买/切换入A股标的 (0=关闭; 纳指/黄金天然豁免——QDII/商品份额
                                     # 不含国家队信息; 份额数据2016起, T+1发布已在数据层shift)
    share_window: int = 20           # 份额净申购率回看窗口(交易日)
    share_gate_names: str = ""       # 份额门受限标的(逗号分隔名, 空=除纳指/黄金外全部)
    share_codes: str = "510300,510500,510050"  # 份额信号源篮子 (沪市宽基, 数据层消费)
    fast_lane_ma: int = 0            # 快速通道追涨闸门: 挑战者现价须高于自身MA(N)才走快速通道 (0=关闭; 2026-08-17 消融为负优化默认关: 被拦切换恰是历史最优档83%胜率/均盈+4.5%; 被拦时落回慢通道=min_hold后正常切换)
    commission: float = 0.0001     # 手续费率(单边)
    slippage: float = 0.0          # 滑点/冲击成本(单边)
    accel_sigma: float = 3.0       # 加速止盈 σ
    accel_enabled: bool = True     # 加速止盈开关
    accel_exempt_names: str = ""    # 免加速止盈的标的(逗号分隔, 如 创业): 高波动标的的mom10尖峰是常态, 止盈=卖飞 (2026-08-18 研究)
    crash_sigma: float = -2.5      # 恐慌抄底 σ
    crash_enabled: bool = True     # 恐慌抄底开关
    crash_lock: int = 7            # 抄底锁仓天数 (与 web UI / bt_runner 默认一致)
    # ── N-30 高低点结构门 (2026-10-30, 用户提议; 默认全关 = 生产基线逐分不变) ──
    crash_hl_win: int = 0          # 高低点窗口(交易日): 60≈3个月 / 120≈6个月; 0=关闭结构门
    crash_pos_max: float = 1.0     # 位置门: 触发指数收盘价在[窗口低点,窗口高点]分位 ≤ 此值才抄底 (1.0=不限)
    crash_dd_min: float = 0.0      # 深度门: 距窗口高点回撤 ≥ 此值(%) 才抄底 (0=不限)
    crash_off_low_max: float = 0.0 # 离底门: 距窗口低点涨幅 ≤ 此值(%) 才抄底 (0=不限)
    # ── B4 抄底再入场闸门 (N-31, 默认关): 审计清单最高优先级未完成项 ──
    crash_reentry_gate: bool = False  # 抄底仓被止损后, 须等恐慌条件解除(crash_now 转 False)才准再抄底
    # ── N-38 抄底「5 日跌幅」接入 (2026-11-02, 用户提议; 默认全关 = 生产基线逐分不变) ──
    # 同行 V12-R2 用 min_hold=2 / switch_confirm=1 的"果断"抄底, 其门槛是绝对百分比
    # (MOM5<=-8% 且 距MA250<-20%)。我方是 mom10 z 值 + MA200 的统计相对口径。
    # 三种接入互斥, 一次只能启用一种:
    #   add   = 原条件 AND (MOM5 <= crash_mom5_pct)      [增量门]
    #   repl  = (MOM5 <= crash_mom5_pct) AND 收盘<MA200  [替代 mom10 条件]
    #   zs    = (MOM5/MOM5的滚动σ <= crash_mom5_z后保留) 见 _make_crash_eval
    crash_mom5_mode: str = ""      # ""=关闭 / "add" / "repl" / "zs"
    crash_mom5_span: int = 5       # 短窗口跨度(交易日), 固定 5 (同行 MOM5 口径)
    crash_mom5_pct: float = -8.0   # add/repl 的 5 日跌幅阈值(%), 如 -8 表示 MOM5<=-8%
    crash_mom5_z: float = -2.0     # zs 模式的 z 门槛 (MOM5 用当日之前的滚动 σ5 标准化)
    # ── N-40/41/42 mom20 斜率门 (2026-11-02): 【已转正, 默认开启】 ──
    # 机制: 动量仍在(mom20>=mom_min)但斜率(动量变化率)已转负 = 动量衰竭 → 提前离场。
    # 无前视: slope = mom20(t) - mom20(t-win), 两个 mom20 都只用 t 及之前的价格。
    # 上线依据 (audit/审计结果清单.md N-42):
    #   - 五起点(2016/18/20/22/24-01 起 → 2026-10-29): 年化中位 +8.92pp, 最差 +6.12pp, 5/5 胜;
    #     回撤中位改善 +1.02pp; Calmar 中位 +0.51, 5/5 胜。
    #   - 参数平台: persist=5 固定时 win 从 8 到 25 共 9 档全部 5/5 胜为正 (win=5 为负, 短窗口误伤)。
    #   - 成本档: 单边 6bp 仍全面优于基线。
    #   - 事件独立性: 2016~2022 四起点差异占比 28%~51%, 不集中; 2024 起点 64% 集中 2025 年(该格降级 n≈1)。
    #   - 跨策略互证: 同行 v9 同机制在创业板 5/5 胜 (+1.0~+2.2pp, 平台非尖峰), 见 N-41。
    # 只对【创业, 黄金】生效: 纳指实测 -5.76pp(0/5), 因其 mom20 混入 513100 的 QDII 溢价噪声 (N-23 记录
    # 溢价单日变化绝对均值 1.24% / 最大 13%), 斜率门无法区分"动量衰竭"与"溢价回落"。
    slope_gate_names: str = "创业,黄金"   # 被约束标的(逗号分隔); "" = 关闭整个斜率门
    slope_gate_win: int = 12       # slope = mom20(t) - mom20(t-12)
    slope_gate_mode: str = "exit"  # exit=斜率转负离场 / tight=加速止盈 σ 收紧
    slope_gate_sigma: float = 2.0  # tight 模式下使用的 σ
    slope_gate_mom_min: float = 0.0  # 仅当 mom20 >= 此值(%)才生效
    # 抗抖动: 无状态机时会出现"持有1天卖出/次日买回"反复 (N-40 首轮实测 58% 离场持有≤1天), p>=3 必须。
    slope_gate_persist: int = 5    # 斜率须连续 N 天为负才触发 (N-40: p=5 是稳定平台, p=2 整行为负)
    slope_gate_cooldown: int = 0   # 斜率离场后 N 天禁止买回 (N-42: 固定冷却无单调改进, 保持 0)
    # ── N-32 N 日高低点结构离场 (默认全关 = 生产基线逐分不变) ──
    exit_hl_win: int = 0           # 结构离场窗口(交易日): 60≈3个月 / 120≈6个月; 0=关闭
    exit_dd_high: float = 0.0      # 回撤离场: 现价距 N 日最高点回撤 ≥ 此值(%) 即离场 (0=关)
    exit_break_low: bool = False   # 破底离场: 现价跌破前 N 日最低点(不含当日) 即离场
    exit_hi_prox: float = 0.0      # 高抛离场: 现价距 N 日最高点 ≤ 此值(%) 即卖出 (0=关; "卖在区间上沿")
    # ── 震荡期收紧止盈 (N-33, 默认关): 趋势走坏时用更紧的 σ 止盈, 趋势正常仍用 accel_sigma ──
    accel_sigma_tight: float = 0.0  # 震荡 regime 下使用的止盈 σ (如 1.0/2.0); 0=关闭 (不改变 accel_sigma)
    accel_dd_win: int = 60          # regime 判定窗口(交易日): 30 / 60
    accel_dd_trig: float = 10.0     # 现价距窗口高点回撤 ≥ 此值(%) → 视为震荡/破位, 切紧 σ
    # N-34: "震荡"的替代定义 (空=沿用 dd; width=区间压缩; mom=动量贴零; momstd=动量横盘度)
    accel_regime: str = ""          # ""=dd(距高点回撤) / width / mom / momstd
    accel_regime_th: float = 0.0    # 阈值 (width/mom 为 %, momstd 为 %)
    # N-37: 震荡期取消 ">3%即买" 快速通道 (止盈后当天被原价买回 = 重置成本价, 2024-10 实证)
    chop_block_fast_buy: bool = False   # True=震荡 regime 下禁止候选走 ">3%即买", 必须走 confirm_days 确认
    chop_block_rebuy_days: int = 0      # >0=某标的刚被加速止盈后 N 天内, 震荡 regime 下禁止其 ">3%即买"
    method: str = "sigma"          # sigma / percentile / ewma_z
    method_pct: float = 2.0        # percentile: 使用P值 (仅method=percentile)
    method_halflife: int = 252     # ewma_z: 半衰期 (仅method=ewma_z)
    abs_mom_enabled: bool = False  # 绝对动量门: 全池动量为负时强制空仓 (抄底锁仓期内豁免)
    defensive_enabled: bool = False  # 防御切换: 空仓信号触发时改持防御资产(动量最强且为正)
    risk_adj_mom: bool = False     # 风险调整动量: 排名用 动量/20日波动率 (阈值仍用原始动量)
    accel_cooldown: int = 3        # 加速止盈后N天禁止买回同一标的 (默认3天, 2026-08-18 用户决议: 打断黄金blow-off循环; 0=关闭, 恐慌抄底豁免)
    accel_cooldown_gold_only: bool = True  # 止盈冷却仅对黄金生效 (默认开; 创业的止盈后买回历史上是正收益, 不误伤)
    accel_reentry_gate: bool = False  # 止盈再入场闸门: 加速止盈后须 mom10 回落到自身止盈阈值下方才允许买回 (0=关闭; 2026-08-18 研究, 打断黄金blow-off循环)
    stop_cooldown: int = 0         # 止损/绝对止损后N天禁止买回同一标的 (0=关闭, 恐慌抄底豁免; 按标的独立计时)
    sh_cash_cooldown: int = 3      # 上证空仓后再入场冷却天数 (0=关闭, 恐慌抄底豁免; 2026-08 起默认3天)
    sh_cash_exempt: bool = False   # 上证空仓仅对A股标的生效 (豁免纳指/黄金)
    # N-44 上证空仓豁免范围细化 (默认 "" = 沿用 SH_CASH_EXEMPT_NAMES 的固定集合):
    # 逗号分隔的标的名, 仅这些标的在上证空仓时不被清出 (如 "黄金" 只豁免黄金)。
    # 需配合 sh_cash_exempt=True 才生效。
    sh_cash_exempt_names: str = ""
    multi_mom: bool = False        # 多周期动量: 排名分 = 3×mom10+2×mom20+1×mom60
    fip_tiebreak: bool = False     # FIP动量质量: 动量差<min_gap时选上涨路径更平滑者
    abs_stop_loss: bool = True     # 绝对止损: 持仓仍动量第一且pnl<=stop_loss时清到空仓(补盲区, 2026-08 起默认开)
    crash_alloc: float = 1.0       # 恐慌抄底切换仓位比例 (0~1): 已持其他标的时切此比例到创业; 空仓则强制全仓
    crash_target_name: str = "创业"  # 恐慌抄底的买入目标标的名 (默认创业; I-18 换池实验用, 需在池内才生效)
    top_k: int = 1                 # 持仓数: 1=Top-1全仓(原行为); >1=Top-K等权轮动(独立引擎)
    mom_skip: int = 0              # 排名动量跳过最近N天(短期反转规避), 0=关闭
    confirm_bind_identity: bool = False  # 确认计数绑定候选身份(B1修复): 同一挑战者连续N天第一才计数;
                                         # False=旧语义(ti!=pos即累计, 每日轮换第一名也会计满)
    fast_min_hold: int = 0         # 快速通道(>3%切换)最短持有天数: 持仓不足N天不被例外动量差切走
                                   # (0=关闭=旧行为; 止损保护不受影响; E2候选开关, 先消融)
    panic_drop: float = 0.0        # 单日急跌离场: 持仓单日跌幅<=-X%当日清仓/切更强候选 (0=关闭; v9式左尾保险, 2026-08-25 消融)
    # ── 波动缩放急跌阈值 (N-7, 2026-09-29): 复刻同行 V12-C 口径, 默认全关 ──
    # 同行 panic_mode='volatility': thr = clip(K × VOL, 2%, 10%), VOL 取"上一实际报价日"的
    # VOL20/VOL60/二者较大值 (prior_clock: previous actual quoted row of that same asset)。
    # 我方 2026-08-25 只消融过"固定阈值"分支 (panic_drop=3/4/5), 未测过这一支。
    panic_vol_on: bool = False     # True=急跌阈值改按波动缩放 (与 panic_drop 互斥, 本开关优先)
    panic_vol_k: float = 1.4       # 风险倍数 (同行扫过 1/1.25/1.5/1.75/2/2.5)
    panic_vol_lo: float = 0.02     # 裁剪下限 (同行 2%)
    panic_vol_hi: float = 0.10     # 裁剪上限 (同行 10%)
    panic_vol_src: str = "current20"  # current20=当日VOL20; prior20/prior60/prior_max20_60=上一实际报价日
    # ── WLS 排名器 (N-14, 2026-09-29, 事前注册 audit/n_wls_protocol.md; 默认关) ──
    # score = (时间加权回归斜率/加权均价×250)/vol20, 再取近 smooth 个实际报价的分数均值。
    # 只换"选谁", 所有阈值(min_gap/入场/离场/上证比较)仍用原始 mom20。
    # 与已否决的 risk_adj_mom 不同: 那个分子是 20 日价差动量, 这个是回归斜率。
    score_wls: bool = False
    wls_window: int = 25           # 20/25/30 (对方 H=20, C/R2=25)
    wls_smooth: int = 1            # 1..4 (对方 H=3, R2=4)
    wls_ann: float = 250.0
    # ── 打分平滑 (N-20 P1-1, 2026-09-29, 事前注册 audit/n_smooth_stopvol_protocol.md; 默认关) ──
    # 排名分改为"最近 N 个交易日 mom20 的滚动均值"; **只影响选谁**, 阈值判断仍用原始 mom20。
    score_smooth: int = 1          # 1=关闭(默认), >1=近 N 日 mom20 均值
    # ── 波动化止损 (N-20 P1-3; 默认关) ──
    # stop_vol_on 时止损线 = -clip(K × VOL, lo, hi), 替换固定 stop_loss;
    # src=prior20 用"该标上一实际报价日"的 VOL20 (对方 prior_clock 语义), current20 用当日。
    stop_vol_on: bool = False
    stop_vol_k: float = 3.0
    stop_vol_lo: float = 0.015
    stop_vol_hi: float = 0.10
    stop_vol_src: str = "prior20"
    # ── 多周期动量加权 (N-24, 2026-09-30, 用户提议; 默认关) ──
    # 格式 "5:w5,10:w10,15:w15,20:w20" (逗号分隔 周期:权重, 权重和应为1)。
    # 非空时用加权动量**整体替换 ms**(排名/gap/入场动量比较); 默认 "" = 关闭=纯 mom20。
    score_mom_mix: str = ""
    bull_pool_extra: str = ""      # 牛市附加池(标的名,逗号分隔): 仅牛市参与排名与持仓, 熊市制度性离场 (""=关闭; 学v9 regime条件池, 2026-08-25 消融)
    bull_pool_rule: str = "sh_mom20"  # 牛市判定规则 (bull_pool_extra 配套): sh_mom20=上证20日动量>0
    never_empty: bool = False      # 永不空仓 (v9式避险接盘): 去现金决策改持避险池(纳指/黄金)score最强, 允许负动量(跌最少者); 默认关, 2026-08-28 消融


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
def calc_accel_thresholds(mom10_mat: np.ndarray, cfg: "SignalConfig",
                          sigma: float = None) -> np.ndarray:
    """加速止盈阈值矩阵。expanding 逐日计算, shift(1) 保证 t 日阈值只用 t-1 及之前的数据
    (与恐慌阈值 calc_threshold 的 arr[:t] 口径一致, 修复原全样本计算的前视偏差)"""
    _sig = cfg.accel_sigma if sigma is None else float(sigma)
    accel_th_mat = np.full(mom10_mat.shape, np.nan)
    for i in range(mom10_mat.shape[0]):
        s = pd.Series(mom10_mat[i])
        if cfg.method == "percentile":
            th = s.expanding(min_periods=60).quantile((100 - cfg.method_pct) / 100.0)
        elif cfg.method == "ewma_z":
            e = s.ewm(halflife=cfg.method_halflife, min_periods=60)
            th = e.mean() + _sig * np.sqrt(e.var())
        else:
            ex = s.expanding(min_periods=60)
            th = ex.mean() + _sig * ex.std()
        accel_th_mat[i] = th.shift(1).values
    return accel_th_mat


def calc_mom60(price_mat: np.ndarray) -> np.ndarray:
    """60日动量 (multi_mom 用)"""
    m = np.full(price_mat.shape, np.nan)
    m[:, 60:] = (price_mat[:, 60:] / price_mat[:, :-60] - 1) * 100
    return m


def calc_vol20(price_mat: np.ndarray, window: int = 20) -> np.ndarray:
    """日收益波动率 (risk_adj_mom 用; window=20 为默认; N-7 加 window 参数以算 VOL60)"""
    vol = np.full(price_mat.shape, np.nan)
    for i in range(price_mat.shape[0]):
        vol[i] = pd.Series(price_mat[i]).pct_change().rolling(window).std().values
    return vol


def calc_up_ratio20(price_mat: np.ndarray) -> np.ndarray:
    """20日上涨天数占比 (FIP 动量质量用)"""
    up = np.full(price_mat.shape, np.nan)
    for i in range(price_mat.shape[0]):
        up[i] = (pd.Series(price_mat[i]).pct_change() > 0).rolling(20).mean().values
    return up


def _prior_quote_shift(mat: np.ndarray, price_mat: np.ndarray) -> np.ndarray:
    """把 mat 平移成"该标的**上一实际报价日**的值" (prior_clock 语义; 无前视)。

    上一实际报价日 = 严格早于 t 的最近一个价格有限的日子。若此前无值则 NaN。
    """
    out = np.full(mat.shape, np.nan)
    for i in range(mat.shape[0]):
        prev = np.nan
        for t in range(mat.shape[1]):
            out[i, t] = prev
            if not np.isnan(price_mat[i, t]) and not np.isnan(mat[i, t]):
                prev = mat[i, t]
    return out


def calc_mom(price_mat: np.ndarray, window: int) -> np.ndarray:
    """w 日价差动量 (%), 供多周期加权使用 (N-24)。"""
    m = np.full(price_mat.shape, np.nan)
    if window <= 0 or price_mat.shape[1] <= window:
        return m
    m[:, window:] = (price_mat[:, window:] / price_mat[:, :-window] - 1) * 100
    return m


def calc_mom_mix(price_mat: np.ndarray, spec: str, blend=None) -> np.ndarray:
    """多周期动量加权 (N-24): spec="5:0.05,10:0.15,15:0.20,20:0.60"。

    blend = (row_idx, other_prices_1d, weight): 把该行的**每个周期动量**与另一序列
    同周期动量按 (1-w)·本序列 + w·另一序列 混合 —— 复刻生产的纳指 QQQ-CNY 混合口径
    (bt_runner: mom20_mat[纳指] = (1-w)·513100动量 + w·QQQ-CNY动量, I-11 默认 w=0.7)。
    **不传 blend 会丢掉 QQQ 混合, 使退化权重 20:1.00 不等于基线** (2026-09-30 修正的 bug)。

    任一**非零权重**分量为 NaN 时结果 NaN (保守: 预热不足不参与排名)。
    """
    parts = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        w_s, wt_s = tok.split(":")
        parts.append((int(w_s), float(wt_s)))
    if not parts:
        raise ValueError("score_mom_mix 为空")
    total = sum(w for _, w in parts)
    if abs(total - 1.0) > 1e-9:
        raise ValueError("score_mom_mix 权重和必须为 1, 当前 %.4f" % total)
    out = np.zeros(price_mat.shape)
    bad = np.zeros(price_mat.shape, dtype=bool)
    for w, wt in parts:
        m = calc_mom(price_mat, w)
        if blend is not None:
            _r, _other, _bw = blend
            mo = calc_mom(np.asarray(_other, dtype=float)[None, :], w)[0]
            m = m.copy()
            m[_r] = (1.0 - _bw) * m[_r] + _bw * mo
        if wt > 0:
            bad |= ~np.isfinite(m)
        out += wt * np.nan_to_num(m, nan=0.0)
    return np.where(bad, np.nan, out)


def calc_wls_score(price_mat: np.ndarray, window: int, smooth: int = 1,
                   ann: float = 250.0, vol_mat: np.ndarray = None, blend=None) -> np.ndarray:
    """时间加权回归斜率打分 (N-14, 复刻同行 SCORE_WLS 口径; 默认关).

    score = (斜率 / 加权均价 × ann) / vol20
      斜率 = Σw(x-x̄)(y-ȳ)/Σw(x-x̄)², 权重 w_i = i+1 (越近越大)
      smooth > 1 时对位移 k=1..smooth 的同样回归取均值 (近 N 个报价的分数均值)
    vol 用**当日** vol20 (对方 current20 口径); 未传则现场算。
    仅供排名, 阈值判断仍用原始 mom20。
    """
    n_etf, n = price_mat.shape
    raw = np.full((n_etf, n), np.nan)
    ks = range(smooth)
    xs = np.arange(window, dtype=float)
    wts = xs + 1.0
    wsum = wts.sum()
    wmx = float((wts * xs).sum() / wsum)
    dxx = wts * (xs - wmx) ** 2
    wsxx = float(dxx.sum())
    for i in range(n_etf):
        px = price_mat[i]
        for t in range(window + smooth - 2, n):
            vals = []
            for k in ks:
                end = t - k
                seg = px[end - window + 1:end + 1]
                if len(seg) < window or np.any(~np.isfinite(seg)):
                    continue
                wmy = float((wts * seg).sum() / wsum)
                if wmy <= 0:
                    continue
                wsxy = float((wts * (xs - wmx) * (seg - wmy)).sum())
                if wsxx <= 0:
                    continue
                vals.append((wsxy / wsxx) / wmy * ann)
            if vals:
                raw[i, t] = sum(vals) / len(vals)
    v = vol_mat if vol_mat is not None else calc_vol20(price_mat)
    with np.errstate(divide='ignore', invalid='ignore'):
        sc = np.where((v > 0) & np.isfinite(raw), raw / np.where(v > 0, v, 1.0), np.nan)
    if blend is not None:
        # N-14 修正: 该行的打分与另一序列(QQQ-CNY)的同样打分混合, 复刻生产纳指口径
        _r, _other, _bw = blend
        _op = np.asarray(_other, dtype=float)[None, :]
        _ov = calc_vol20(_op)
        _raw2 = _wls_raw(_op, window, smooth, ann)
        with np.errstate(divide='ignore', invalid='ignore'):
            _sc2 = np.where((_ov > 0) & np.isfinite(_raw2), _raw2 / np.where(_ov > 0, _ov, 1.0), np.nan)
        sc = sc.copy()
        sc[_r] = (1.0 - _bw) * sc[_r] + _bw * _sc2[0]
    return sc


def _wls_raw(price_mat: np.ndarray, window: int, smooth: int, ann: float) -> np.ndarray:
    """WLS 斜率/加权均价×ann (打分分子, 未除 vol) —— 供 calc_wls_score 与混合口径复用。"""
    n_etf, n = price_mat.shape
    raw = np.full((n_etf, n), np.nan)
    xs = np.arange(window, dtype=float)
    wts = xs + 1.0
    wsum = wts.sum()
    wmx = float((wts * xs).sum() / wsum)
    wsxx = float((wts * (xs - wmx) ** 2).sum())
    for i in range(n_etf):
        px = price_mat[i]
        for t in range(window + smooth - 2, n):
            vals = []
            for k in range(smooth):
                end = t - k
                seg = px[end - window + 1:end + 1]
                if len(seg) < window or np.any(~np.isfinite(seg)):
                    continue
                wmy = float((wts * seg).sum() / wsum)
                if wmy <= 0 or wsxx <= 0:
                    continue
                wsxy = float((wts * (xs - wmx) * (seg - wmy)).sum())
                vals.append((wsxy / wsxx) / wmy * ann)
            if vals:
                raw[i, t] = sum(vals) / len(vals)
    return raw


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
    if getattr(cfg, "crash_reentry_gate", False):
        # 显式告警而非静默分叉 (2026-09 副本分叉事故的教训): 该闸门只在 Top-1 引擎实现
        warnings.warn("crash_reentry_gate (B4) 仅实现于 Top-1 引擎; top_k>1 时不生效",
                      RuntimeWarning)
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

    _m5 = (_make_mom5_series(cfg, cyb_idx_p, sh_idx_p)
           if str(getattr(cfg, "crash_mom5_mode", "") or "") else None)

    def _moments(tag):
        """N-38: 把 5 日动量绑到当前分支的指数序列上 (cyb/sh 分别注入)。"""
        def binder(_cfg):
            if _m5 is None:
                return lambda tf, _lb=None: (None, None)
            # label 区分 "上证指数" / "创业板指": 5 日动量必须用各自指数的序列
            return lambda tf, _lb=None: _m5["sh" if (_lb and "上证" in _lb) else "cyb"](tf)
        return binder

    _crash_eval = _make_crash_eval(cfg, calc_threshold, CRASH_DIAG,
                                   _moments=_moments("cyb"))

    sw_ = 10
    _nf = len(cyb_idx_p)
    mom10_cyb = np.full(_nf, np.nan)
    mom10_sh_idx = np.full(_nf, np.nan)
    mom10_cyb[sw_:] = (cyb_idx_p[sw_:] / cyb_idx_p[:-sw_] - 1) * 100
    mom10_sh_idx[sw_:] = (sh_idx_p[sw_:] / sh_idx_p[:-sw_] - 1) * 100
    ma200_cyb = _ma_safe(cyb_idx_p, 200, ma200_cyb, "创业板指")
    ma200_sh = _ma_safe(sh_idx_p, 200, ma200_sh, "上证指数")

    # ── N-30 高低点结构门: 窗口 HH/LL (含当日收盘, 与 mom10/MA200 同口径, 无前视) ──
    _hl_cyb = _hl_sh = None
    if int(getattr(cfg, "crash_hl_win", 0) or 0) > 0:
        _hl_cyb = _roll_hilo(cyb_idx_p, int(cfg.crash_hl_win))
        _hl_sh = _roll_hilo(sh_idx_p, int(cfg.crash_hl_win))

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
        # 三态: None=无法判定(缺数据) → 记 CRASH_DIAG 供实盘告警, 行为同旧版(不触发)
        if _crash_eval(mom10_sh_idx, ma200_sh, sh_idx_p, tf, "上证指数", _hl_sh):
            crash_now = True
        if not crash_now and _crash_eval(mom10_cyb, ma200_cyb, cyb_idx_p, tf, "创业板指", _hl_cyb):
            crash_now = True

        held_idx = [i for i in range(n_etfs) if w[i] > 1e-9]
        buys = []; sells = []
        cyb_pool_idx = (etf_names.index(cfg.crash_target_name)
                        if cfg.crash_target_name in etf_names else -1)

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

    CRASH_DIAG.clear()
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

    # 纳指放量门: 量比矩阵由调用方(bt_runner/live runner)在全量历史上算好切片传入
    # (成交量不在 price_mat 里, 引擎自己算不出来; 缺省 = 门关闭)
    nz_vr_mat = (pre_mats or {}).get("nz_vr")
    if cfg.nz_vol_gate > 0 and nz_vr_mat is None:
        # 静默失效是 2026-10-15 事故的成因, 这里必须显式告警
        warnings.warn("nz_vol_gate>0 但未传入 pre_mats['nz_vr'](量比矩阵) → 放量门不生效; "
                      "调用方需补 VOL20/VOL120 矩阵", RuntimeWarning)

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

    _m5 = (_make_mom5_series(cfg, cyb_idx_p, sh_idx_p)
           if str(getattr(cfg, "crash_mom5_mode", "") or "") else None)

    def _moments(tag):
        """N-38: 把 5 日动量绑到当前分支的指数序列上 (cyb/sh 分别注入)。"""
        def binder(_cfg):
            if _m5 is None:
                return lambda tf, _lb=None: (None, None)
            # label 区分 "上证指数" / "创业板指": 5 日动量必须用各自指数的序列
            return lambda tf, _lb=None: _m5["sh" if (_lb and "上证" in _lb) else "cyb"](tf)
        return binder

    _crash_eval = _make_crash_eval(cfg, calc_threshold, CRASH_DIAG,
                                   _moments=_moments("cyb"))

    # ── 恐慌抄底: 10日动量 for 399006 + 000001 ──
    # cyb_idx_p/sh_idx_p 为完整历史序列(含预热), 决策时用 tf = t + pre_days 访问,
    # 保证短回测区间下阈值/MA200 依然用足量历史计算
    sw = 10
    _nf = len(cyb_idx_p)
    mom10_cyb = np.full(_nf, np.nan)
    mom10_sh_idx = np.full(_nf, np.nan)
    mom10_cyb[sw:] = (cyb_idx_p[sw:] / cyb_idx_p[:-sw] - 1) * 100
    mom10_sh_idx[sw:] = (sh_idx_p[sw:] / sh_idx_p[:-sw] - 1) * 100

    ma200_cyb = _ma_safe(cyb_idx_p, 200, ma200_cyb, "创业板指")
    ma200_sh = _ma_safe(sh_idx_p, 200, ma200_sh, "上证指数")

    # ── N-30 高低点结构门: 窗口 HH/LL (含当日收盘, 与 mom10/MA200 同口径, 无前视) ──
    _hl_cyb = _hl_sh = None
    if int(getattr(cfg, "crash_hl_win", 0) or 0) > 0:
        _hl_cyb = _roll_hilo(cyb_idx_p, int(cfg.crash_hl_win))
        _hl_sh = _roll_hilo(sh_idx_p, int(cfg.crash_hl_win))

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
    slope_block_until = np.full(n_etfs, -1)         # N-40 斜率离场冷却 (按标的独立计时)
    crash_alloc_now = 0.0                           # 恐慌抄底当前仓位比例 (0=无抄底叠加仓)
    crash_base_pos = -1                             # 抄底前的名义主仓 (用于组合净值计算; -1=空仓抄底)
    # ── B4 抄底再入场闸门 (N-31, 默认关): 抄底仓被止损后, 须等恐慌条件解除才准再抄底 ──
    pos_from_crash = False                          # 当前持仓是否由恐慌抄底建立
    crash_wait_clear = False                        # True=已止损, 等 crash_now 转 False 才解锁
    sat_pos = -1; sat_alloc = 0.0; sat_hs = -1      # 分档切换组合: 卫星腿(-1=无), 权重与入场日
    actions = []
    nav_log = []
    # N-44: 上证空仓的豁免标的集合。sh_cash_exempt_names 非空时按它, 否则沿用固定集合。
    _exempt_set = ({x.strip() for x in cfg.sh_cash_exempt_names.split(",") if x.strip()}
                   if cfg.sh_cash_exempt_names.strip() else SH_CASH_EXEMPT_NAMES)
    _extra_names = {x.strip() for x in cfg.bull_pool_extra.split(",") if x.strip()}
    extra_idx = np.array([i for i, nm in enumerate(etf_names) if nm in _extra_names], dtype=int)
    bull_rule = cfg.bull_pool_rule if extra_idx.size else ""
    _sh_ma250 = pd.Series(sh_idx_p).rolling(250).mean().values if extra_idx.size else None
    # 永不空仓避险池: 跨境+黄金 (v9 GLOBAL_POOL + GOLD)
    safe_idx = np.array([i for i, nm in enumerate(etf_names)
                         if nm in SH_CASH_EXEMPT_NAMES], dtype=int)

    def _safe_best(t):
        """避险池中 score 最强且数据有效者, 无则 -1 (允许负动量, 跌最少者; v9 NEVER_EMPTY 同义)"""
        best, best_s = -1, -np.inf
        for i in safe_idx:
            s = score[i]
            if not np.isnan(s) and s > best_s:
                best, best_s = i, s
        return best

    def is_bull(t):
        """牛市判定: sh_mom20=上证20日动量>0; sh_ma250=上证指数>其250日均线 (v9式)"""
        if not extra_idx.size:
            return True
        if bull_rule == "sh_mom20":
            return bool(sh_mom[t] > 0)
        if bull_rule == "sh_ma250":
            _tf = t + pre_days
            return bool(not np.isnan(_sh_ma250[_tf]) and sh_idx_p[_tf] > _sh_ma250[_tf])
        return True

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
    if (cfg.risk_adj_mom or cfg.stop_sigma > 0 or cfg.vol_scale_target > 0
            or cfg.panic_vol_on or cfg.stop_vol_on):
        vol_mat = (pre_mats or {}).get("vol")
        if vol_mat is None:
            vol_mat = calc_vol20(price_mat)

    # ── 波动缩放急跌阈值 (N-7): thr[i,t] = clip(K × VOL, lo, hi), VOL 口径见 cfg.panic_vol_src ──
    # "上一实际报价日" = 该标的最近一个有效报价行的 VOL (prior_clock, 绝不用当日/未来)。
    panic_thr_mat = None
    if cfg.panic_vol_on:
        _src = cfg.panic_vol_src
        if _src in ("prior20", "prior60", "prior_max20_60"):
            _v20 = vol_mat
            _v60 = (pre_mats or {}).get("vol60")
            if _v60 is None:
                _v60 = calc_vol20(price_mat, 60)
            if _src == "prior20":
                _base = _v20
            elif _src == "prior60":
                _base = _v60
            else:
                _base = np.fmax(_v20, _v60)
            _cur = np.clip(cfg.panic_vol_k * _base, cfg.panic_vol_lo, cfg.panic_vol_hi)
            panic_thr_mat = _prior_quote_shift(_cur, price_mat)
        else:  # current20: 当日 VOL20 (同行 current20 口径)
            panic_thr_mat = np.clip(cfg.panic_vol_k * vol_mat, cfg.panic_vol_lo, cfg.panic_vol_hi)

    # ── 波动化止损的波动尺度 (N-20 P1-3): prior20 用"上一实际报价日"的 VOL20 ──
    stop_vol_mat = None
    if cfg.stop_vol_on:
        if cfg.stop_vol_src in ("prior20", "prior60", "prior_max20_60"):
            _sb = vol_mat
            if cfg.stop_vol_src in ("prior60", "prior_max20_60"):
                _v60s = (pre_mats or {}).get("vol60")
                if _v60s is None:
                    _v60s = calc_vol20(price_mat, 60)
                _sb = _v60s if cfg.stop_vol_src == "prior60" else np.fmax(vol_mat, _v60s)
            stop_vol_mat = _prior_quote_shift(_sb, price_mat)
        else:
            stop_vol_mat = vol_mat

    def _panic_thr_at(i, t):
        """返回该标的当日急跌阈值(正数, 单位为%); None=不启用/无值。"""
        if cfg.panic_vol_on:
            if panic_thr_mat is None:
                return None
            v = panic_thr_mat[i, t]
            return float(v) * 100.0 if not np.isnan(v) else None
        if cfg.panic_drop > 0:
            return float(cfg.panic_drop)
        return None

    def _panic_hit(i, t):
        """持仓 i 在 t 日是否触发急跌离场。"""
        thr = _panic_thr_at(i, t)
        if thr is None:
            return False
        _p0 = price_mat[i, max(0, t - 1)]; _p1 = price_mat[i, t]
        if np.isnan(_p0) or np.isnan(_p1) or _p0 <= 0:
            return False
        return (_p1 / _p0 - 1) * 100 <= -thr

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
    rate_shock_arr = (pre_mats or {}).get("rate_shock")
    if rate_shock_arr is not None:
        rate_shock_arr = np.asarray(rate_shock_arr, dtype=float)

    def _panic_block(i, t, arr, gate):
        """恐慌门: 序列值>gate 时禁止买/切该标的"""
        return bool(gate > 0 and arr is not None and not np.isnan(arr[t]) and arr[t] > gate)

    # ── 份额风险门序列 (pre_mats 传入, 已按决策区对齐且 T+1 shift 无前视) ──
    share_arr = (pre_mats or {}).get("share_flow")
    if share_arr is not None:
        share_arr = np.asarray(share_arr, dtype=float)
    _share_names = {x.strip() for x in cfg.share_gate_names.split(",") if x.strip()}

    def _share_block(i, t):
        """份额风险门: 宽基篮子净申购率 < -share_gate 时禁止买/切 A 股标的。
        纳指/黄金天然豁免 (QDII额度/商品申赎不含国家队信息);
        序列 NaN(数据未覆盖, 2016年前) 时不拦, 优雅降级"""
        if cfg.share_gate <= 0 or share_arr is None or np.isnan(share_arr[t]):
            return False
        nm = etf_names[i]
        if _share_names:
            if nm not in _share_names:
                return False
        elif nm in SH_CASH_EXEMPT_NAMES:
            return False
        return bool(share_arr[t] < -cfg.share_gate)

    # ── 快速通道追涨闸门均线 (挑战者须高于自身MA(N)才走快速通道; 优先用全量历史预计算值防冷启动,
    #    内部冷启动兜底时窗口前N-1天为NaN→闸门放行, 仅影响极短回测窗口起点) ──
    ma_gate_mat = None
    if cfg.fast_lane_ma > 0:
        ma_gate_mat = (pre_mats or {}).get("ma_fast")
        if ma_gate_mat is None:
            ma_gate_mat = np.full(price_mat.shape, np.nan)
            for _i in range(price_mat.shape[0]):
                ma_gate_mat[_i] = pd.Series(price_mat[_i]).rolling(cfg.fast_lane_ma).mean().values

    # ── N-32 结构离场矩阵: HH(含当日) / LL(不含当日, 否则 px<LL 恒不成立) ──
    # 优先用调用方在全量历史上预计算并切片的矩阵 (防冷启动空白, 与 accel_th 同口径)
    _exit_hh = (pre_mats or {}).get("exit_hh")
    _exit_ll = (pre_mats or {}).get("exit_ll")
    if cfg.exit_hl_win > 0 and (_exit_hh is None or _exit_ll is None):
        _exit_hh = np.full(price_mat.shape, np.nan)
        _exit_ll = np.full(price_mat.shape, np.nan)
        for _i in range(price_mat.shape[0]):
            _h, _l = _roll_hilo(price_mat[_i], int(cfg.exit_hl_win))
            _exit_hh[_i] = _h
            _exit_ll[_i] = np.concatenate([[np.nan], _l[:-1]])

    def hl_exit(i, t):
        """N 日高低点结构离场 (N-32): 距 N 日最高点回撤 ≥ exit_dd_high% 或 跌破前 N 日最低点。
        与 trail_hit 的区别: 锚在【滚动窗口】结构位, 不是【本次入场以来】的最高点。"""
        if cfg.exit_hl_win <= 0 or i < 0 or t < 0:
            return False
        px = price_mat[i, t]
        if np.isnan(px):
            return False
        h = _exit_hh[i, t]
        if cfg.exit_dd_high > 0 and not np.isnan(h) and h > 0:
            if px <= h * (1 - cfg.exit_dd_high / 100.0):
                return True
        if cfg.exit_break_low:
            l = _exit_ll[i, t]
            if not np.isnan(l) and px < l:
                return True
        if cfg.exit_hi_prox > 0 and not np.isnan(h) and h > 0:
            if px >= h * (1 - cfg.exit_hi_prox / 100.0):
                return True      # 高抛: 已回到区间上沿 (卖在最强处)
        return False

    # ── N-33 震荡期收紧止盈: regime 判据用"距窗口高点回撤", 命中时改读紧 σ 阈值矩阵 ──
    _accel_tight = (pre_mats or {}).get("accel_th_tight")
    _accel_hh = (pre_mats or {}).get("accel_hh")
    _regime_ind = (pre_mats or {}).get("regime_ind")
    _need_regime = (cfg.accel_sigma_tight > 0 or cfg.chop_block_fast_buy
                    or cfg.chop_block_rebuy_days > 0)
    if _need_regime:
        if _accel_tight is None and cfg.accel_sigma_tight > 0:
            _accel_tight = calc_accel_thresholds(mom10_mat, cfg, sigma=cfg.accel_sigma_tight)
        if _accel_hh is None:
            _accel_hh = np.full(price_mat.shape, np.nan)
            for _i in range(price_mat.shape[0]):
                _accel_hh[_i] = _roll_hilo(price_mat[_i], int(cfg.accel_dd_win))[0]
        _mode = (cfg.accel_regime or "").strip().lower()
        if _mode in ("width", "width_pct") and _regime_ind is None:
            _regime_ind = np.full(price_mat.shape, np.nan)
            for _i in range(price_mat.shape[0]):
                _h, _l = _roll_hilo(price_mat[_i], 20)
                _regime_ind[_i] = (_h - _l) / price_mat[_i] * 100.0
        if _mode in ("momstd", "momstd_pct") and _regime_ind is None:
            _regime_ind = np.full(price_mat.shape, np.nan)
            for _i in range(price_mat.shape[0]):
                _regime_ind[_i] = pd.Series(mom10_mat[_i]).rolling(20).std().values
        if _mode.endswith("_pct") and _regime_ind is not None:
            # 转成"自身滚动 250 日分位"(0~1): 消除量纲/标的差异, 阈值=分位
            _regime_ind = np.array([
                pd.Series(_regime_ind[_i]).rolling(250, min_periods=120).rank(pct=True).values
                for _i in range(_regime_ind.shape[0])])
        if _mode == "combo":
            # N-35: 多定义组合(多数表决)。四票: dd(30d≥10%) / width_pct≤20% / momstd_pct≤20% / spread≤3
            _rw = (pre_mats or {}).get("regime_width_pct"); _rm = (pre_mats or {}).get("regime_momstd_pct")
            if _rw is None:
                _w0 = np.full(price_mat.shape, np.nan)
                for _i in range(price_mat.shape[0]):
                    _h, _l = _roll_hilo(price_mat[_i], 20)
                    _w0[_i] = (_h - _l) / price_mat[_i] * 100.0
                _rw = np.array([pd.Series(_w0[_i]).rolling(250, min_periods=120).rank(pct=True).values
                                for _i in range(_w0.shape[0])])
            if _rm is None:
                _m0 = np.array([pd.Series(mom10_mat[_i]).rolling(20).std().values
                                for _i in range(price_mat.shape[0])])
                _rm = np.array([pd.Series(_m0[_i]).rolling(250, min_periods=120).rank(pct=True).values
                                for _i in range(_m0.shape[0])])
            _regime_combo = (_rw, _rm)

    def _regime_chop(i, t):
        """N-33/N-34 "震荡" regime 判定 (只用当日可得的自身信息):
        dd=距窗口高点回撤≥trig% / width=区间宽度≤th% / mom=|mom20|≤th% / momstd=std(mom10,20)≤th"""
        mode = (cfg.accel_regime or "").strip().lower()
        if mode in ("", "dd"):
            if _accel_hh is None:
                return False
            h = _accel_hh[i, t]
            px = price_mat[i, t]
            if np.isnan(h) or np.isnan(px) or h <= 0:
                return False
            return bool(px <= h * (1 - cfg.accel_dd_trig / 100.0))
        if mode == "mom":
            v = abs(mom20_mat[i, t])
            return bool((not np.isnan(v)) and v <= cfg.accel_regime_th)
        if mode == "composite":
            # N-35: 三个"平坦度"分位的等权平均 (width/momstd/|mom20|), 越低=越像横盘
            _rw = (pre_mats or {}).get("regime_width_pct"); _rm = (pre_mats or {}).get("regime_momstd_pct")
            _rp = (pre_mats or {}).get("regime_mom_pct")
            if _rw is None or _rm is None or _rp is not None:
                if _rw is None or _rm is None:
                    return False
            vals = []
            for _mat in (_rw, _rm, _rp):
                if _mat is None:
                    continue
                v = _mat[i, t]
                if not np.isnan(v):
                    vals.append(float(v))
            if len(vals) < 2:
                return False
            return bool(float(np.mean(vals)) <= cfg.accel_regime_th)
        if mode == "combo":
            need = int(cfg.accel_regime_th) if cfg.accel_regime_th >= 1 else 2
            votes = []
            h = _accel_hh[i, t] if _accel_hh is not None else np.nan
            px = price_mat[i, t]
            votes.append(bool((not np.isnan(h)) and (not np.isnan(px)) and h > 0
                              and px <= h * 0.90))            # dd: 30d 高点回撤 ≥10%
            votes.append(bool((not np.isnan(_regime_combo[0][i, t])) and _regime_combo[0][i, t] <= 0.20))
            votes.append(bool((not np.isnan(_regime_combo[1][i, t])) and _regime_combo[1][i, t] <= 0.20))
            _row = mom20_mat[:, t]; _row = _row[~np.isnan(_row)]
            votes.append(bool(_row.size >= 2 and float(_row.max() - _row.min()) <= 3.0))
            return sum(votes) >= need
        if mode in ("spread", "weak"):
            # 横截面定义 (N-34 补充): spread=全池动量极差小(无领跑者); weak=全池动量最强也≤阈值(普跌)
            row = mom20_mat[:, t]
            row = row[~np.isnan(row)]
            if row.size < 2:
                return False
            v = float(row.max() - row.min()) if mode == "spread" else float(row.max())
            return bool(v <= cfg.accel_regime_th)
        _ind = _regime_ind
        if _ind is None:
            return False
        v = _ind[i, t]
        if np.isnan(v):
            return False
        return bool(v <= cfg.accel_regime_th)

    def accel_line(i, t):
        """加速止盈阈值: 震荡 regime 成立时改用 accel_sigma_tight 算出的更紧阈值,
        否则仍用正常 accel_sigma。N-40 tight 模式: 斜率转负时同样收紧。"""
        th = accel_th_mat[i, t]
        if i < 0 or t < 0:
            return th
        if cfg.accel_sigma_tight > 0 and _accel_tight is not None and _regime_chop(i, t):
            return _accel_tight[i, t]
        if _slope_gate_hit(i, t) and cfg.slope_gate_mode == "tight":
            return _accel_tight_slope[i, t] if _accel_tight_slope is not None else th
        return th

    # ── N-40 mom20 斜率门 (2026-11-02): 只作用于 slope_gate_names 内的标的 ──
    _slope_names = {x.strip() for x in cfg.slope_gate_names.split(",") if x.strip()}
    _slope_idx = {etf_names.index(n) for n in _slope_names if n in etf_names}
    _accel_tight_slope = None
    if _slope_idx and cfg.slope_gate_mode == "tight" and cfg.slope_gate_sigma > 0:
        _accel_tight_slope = calc_accel_thresholds(mom10_mat, cfg, sigma=cfg.slope_gate_sigma)

    def _slope_gate_hit(i, t):
        """N-40: 该标的 mom20 >= mom_min 且斜率(mom20(t)-mom20(t-win))转负 → True。
        只用 t 及之前数据, 无前视。slope_gate_persist>1 时要求连续 N 天为负。"""
        if not _slope_idx or i not in _slope_idx or t < 0:
            return False
        if i >= mom20_mat.shape[0] or t >= mom20_mat.shape[1]:
            return False
        _pw = max(int(cfg.slope_gate_persist), 1)
        for _k in range(_pw):
            _tk = t - _k
            if _tk < 0:
                return False
            _m = mom20_mat[i, _tk]
            if np.isnan(_m) or _m < cfg.slope_gate_mom_min:
                return False
            _tw = _tk - int(cfg.slope_gate_win)
            if _tw < 0:
                return False
            _old = mom20_mat[i, _tw]
            if np.isnan(_old) or not (_m - _old < 0):
                return False
        return True

    tp_recent = np.full(n_etfs, -10 ** 9)      # N-37: 各标的最近一次加速止盈日

    def fast_buy_blocked(i, t):
        """N-37 震荡期取消 ">3%即买": (a) 全局开关 chop_block_fast_buy;
        (b) 该标的刚被止盈后 chop_block_rebuy_days 天内, 且仍处震荡 regime。"""
        if i < 0:
            return False
        if cfg.chop_block_fast_buy and _regime_chop(i, t):
            return True
        if (cfg.chop_block_rebuy_days > 0 and t - tp_recent[i] <= cfg.chop_block_rebuy_days
                and _regime_chop(i, t)):
            return True
        return False

    def stop_line(i, t):
        """止损线: stop_sigma>0 时 = -k×标的自身20日波动率(不同标的不同线);
        stop_sigma_floor=True 时固定线为下限(只放宽不收紧);
        波动率缺失/非正时回退固定 stop_loss。
        N-20 P1-3: stop_vol_on 时改用 -clip(K×VOL, lo, hi), VOL 按 stop_vol_src 取当日/上一报价日。"""
        if cfg.stop_vol_on and stop_vol_mat is not None:
            _v = stop_vol_mat[i, t]
            if not np.isnan(_v) and _v > 0:
                _mag = min(max(cfg.stop_vol_k * _v * 100.0, cfg.stop_vol_lo * 100.0),
                           cfg.stop_vol_hi * 100.0)
                return -_mag
        if cfg.stop_sigma > 0 and not np.isnan(vol_mat[i, t]) and vol_mat[i, t] > 0:
            s = -cfg.stop_sigma * vol_mat[i, t] * 100.0
            return min(cfg.stop_loss, s) if cfg.stop_sigma_floor else s
        return cfg.stop_loss

    def nz_vol_block(i, t):
        """纳指放量门 (2026-10-22 体检): 513100 量比 VOL20/VOL120 高于阈值时【不新买纳指】。

        体检证据: 纳指ETF 量比与未来20日收益 IC -0.213 / -0.214 (2016-21 与 2021-26 两段
        一致), 前5档单调 -0.90/-0.80; 控制溢价档与波动档后依然成立; 落到引擎逐笔:
        高量比入场 21 笔平均 +0.99% vs 低量比 +5.30%。
        语义: 只拦"新买纳指"(空仓入场/切入纳指), 不影响卖出、止损、加速止盈与其它标的。
        """
        if cfg.nz_vol_gate <= 0 or nz_vr_mat is None:
            return False
        if etf_names[i] != "纳指":
            return False
        if t < 0 or t >= nz_vr_mat.shape[1]:
            return False
        v = nz_vr_mat[i, t]
        return (not np.isnan(v)) and v > cfg.nz_vol_gate

    def trail_hit(i, t):
        """移动止损(吊灯): 现价相对【本次入场以来最高收盘】回撤 ≥ trail_stop% 即触发。

        主流趋势跟踪的标准退出方式 (Chandelier Exit 的百分比简化版)。
        只做"利润保护", 与固定止损 -5% 并存; 触发后与止损同路: 有更优候选就切, 否则清空仓。
        """
        if cfg.trail_stop <= 0 or i < 0 or hs < 0 or t <= hs:
            return False
        px = price_mat[i, t]
        if np.isnan(px):
            return False
        w = price_mat[i, hs:t + 1]
        w = w[~np.isnan(w)]
        if len(w) < 2:
            return False
        peak = float(np.max(w))
        return peak > 0 and px <= peak * (1 - cfg.trail_stop / 100.0)

    def entry_blocked(i, t):
        """入场动量耗尽门 (空仓入场用, 恐慌抄底/切换买入豁免):
        entry_mom_cap: 候选20日动量>X% 视为耗尽; entry_decel_on: mom10<mom20 视为减速;
        entry_bias_cap: 候选现价高于20日均线X% 视为超买;
        nz_vol_gate: 纳指放量时不许新买"""
        if nz_vol_block(i, t):
            return True
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
        # N-24: 多周期动量加权 (用户提议) —— 用加权动量**整体替换动量口径 ms**
        # (排名 / 轮动 gap / 入场与空仓的动量比较全部随之改变;
        #  显式读 mom10_mat·mom20_mat 的判据 —— 止损/加速止盈/恐慌抄底 —— 不受影响)
        if cfg.score_mom_mix:
            _mm = (pre_mats or {}).get("mom_mix")
            if _mm is not None:
                ms = _mm[:, t]; vm = ~np.isnan(ms)

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
        if cfg.score_wls:
            # N-14: WLS 斜率打分 (全量历史预计算后切片, 防窗口起点冷启动)
            score = (pre_mats or {}).get("wls_score")
            if score is not None:
                score = np.where(vm, score[:, t], np.nan)
            else:
                score = np.where(vm, ms, np.nan)
        elif cfg.multi_mom:
            base = 3.0 * mom10_mat[:, t] + 2.0 * ms + mom60_mat[:, t]
            score = np.where(vm & ~np.isnan(mom10_mat[:, t]) & ~np.isnan(mom60_mat[:, t]), base, np.nan)
        else:
            # N-20 P1-1: 打分平滑 —— 排名分改用"最近 N 日 mom20 均值"(阈值仍用原始 ms)
            _sm = (pre_mats or {}).get("score_smooth") if cfg.score_smooth > 1 else None
            score = np.where(vm, (_sm[:, t] if _sm is not None else ms), np.nan)

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

        # ── N-40 斜率离场冷却: N天内禁止买回该标的 (防"卖出次日买回"抖动) ──
        if cfg.slope_gate_cooldown > 0:
            slb = t < slope_block_until
            if slb.any():
                score = np.where(slb, np.nan, score)

        # ── 止盈再入场闸门: mom10 回落到止盈阈值下方自动解除; 未解除者移出候选 ──
        if cfg.accel_reentry_gate and accel_gate.any():
            for _i in np.where(accel_gate)[0]:
                _th = accel_th_mat[_i, t]
                if np.isnan(_th) or (not np.isnan(mom10_mat[_i, t]) and mom10_mat[_i, t] < _th):
                    accel_gate[_i] = False
            if accel_gate.any():
                score = np.where(accel_gate, np.nan, score)

        # ── 牛市附加池: 熊市时移出排名; 持有中走"牛池退出"制度性离场 ──
        if extra_idx.size and not is_bull(t):
            score[extra_idx] = np.nan

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
                  if nm not in _exempt_set and vm[i]]
            sc = (sh_mom[t] > max(ne)) if ne else False
        elif cfg.sh_cash:
            _ms_c = ms.copy()
            if extra_idx.size and not is_bull(t):
                _ms_c[extra_idx] = np.nan
            sc = sh_mom[t] > np.nanmax(_ms_c)
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
        # 三态: None=无法判定(缺数据) → 记 CRASH_DIAG 供实盘告警, 行为同旧版(不触发)
        if _crash_eval(mom10_sh_idx, ma200_sh, sh_idx_p, tf, "上证指数", _hl_sh):
            crash_now = True
        if not crash_now and _crash_eval(mom10_cyb, ma200_cyb, cyb_idx_p, tf, "创业板指", _hl_cyb):
            crash_now = True

        # ── B4 再入场闸门: 恐慌条件解除(出现不恐慌的一天)才解锁二次抄底 ──
        if crash_wait_clear and not crash_now:
            crash_wait_clear = False
        _crash_blocked = bool(getattr(cfg, "crash_reentry_gate", False) and crash_wait_clear)

        cyb_pool_idx = (etf_names.index(cfg.crash_target_name)
                        if cfg.crash_target_name in etf_names else -1)
        if (cfg.crash_enabled and crash_now and t >= crash_held_until
                and cyb_pool_idx >= 0 and not _crash_blocked):
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
                pos_from_crash = True          # B4: 标记本仓来自抄底
                actions.append((t, crash_base_pos, cyb_pool_idx, "恐慌抄底"))
            # 已持创业: 仅刷新锁仓期, 不重复交易
        elif t < crash_held_until:
            # ── 抄底锁仓期: 只允许止损保护退出, 其他动作(轮动/空仓/止盈/动量门)全部锁住 ──
            # 符合"抄底=认准反弹拿住"的设计: 期间不被动量波动切走, 也不被上证空仓误伤
            if pos >= 0 and (pnl <= stop_line(pos, t) or trail_hit(pos, t) or hl_exit(pos, t)):
                np_pos = ti if (ti != pos and not ti_blocked) else -1   # 有更强候选切过去, 否则清仓
                if pnl <= stop_line(pos, t):
                    _why = "止损"
                elif trail_hit(pos, t):
                    _why = "移动止损"
                else:
                    _why = "结构离场"
                sw = True; streak = 0; reason = _why
        elif sc:
            if cfg.sh_cash_exempt and pos >= 0 and etf_names[pos] in _exempt_set:
                pass  # 持有豁免标的(纳指/黄金), 上证空仓不出场
            elif cfg.sh_cash_exempt and pos == -1:
                # 空仓时仍可按正常入场规则买豁免标的
                ex_idx = [i for i, nm in enumerate(etf_names)
                          if nm in _exempt_set and vm[i]]
                bought = False
                if ex_idx:
                    if cfg.accel_reentry_gate:
                        ex_idx = [i for i in ex_idx if not accel_gate[i]]
                    if ex_idx:
                        ei = max(ex_idx, key=lambda i: ms[i])
                        lm = ms[ei]
                        if (not np.isnan(lm) and not entry_blocked(ei, t) and trend_ok(ei, t)
                                and ((lm >= cfg.cash_skip_gt and not fast_buy_blocked(ei, t))
                                     or streak >= cfg.confirm_days)):
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
            _ab = accel_line(base, t); _as_ = accel_line(sat, t)
            accel_b = (cfg.accel_enabled and not np.isnan(_ab)
                       and mom10_mat[base, t] > _ab)
            accel_s = (cfg.accel_enabled and not np.isnan(_as_)
                       and mom10_mat[sat, t] > _as_)
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
        elif (extra_idx.size and pos >= 0 and pos in extra_idx
                and not is_bull(t) and t >= crash_held_until):
            # 熊市确立: 牛池标的制度性离场 (学v9: A股池受regime约束, 熊市无条件撤)
            sw = True; np_pos = -1; streak = 0; reason = "牛池退出"
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
                        or _panic_block(ti, t, skew_arr, cfg.skew_gate)
                        or _panic_block(ti, t, rate_shock_arr, cfg.rate_shock_gate))):
                    pass  # VIX/SKEW/美债利率冲击门: 美股恐慌期不新买纳指
                elif _share_block(ti, t):
                    pass  # 份额风险门: 宽基篮子净赎回期不新买A股标的
                elif (lm >= cfg.cash_skip_gt and not np.isnan(lm)
                        and not fast_buy_blocked(ti, t)):
                    sw = True; np_pos = ti; streak = 0; reason = ">3%即买"
                elif streak >= cfg.confirm_days:
                    sw = True; np_pos = ti; streak = 0; reason = "确认买入"
        elif ti != pos and not ti_blocked and t >= crash_held_until:
            # 注意: 止损只在存在更优候选(ti != pos)时检查——持仓仍是动量第一名时
            # 永不止损 (相对动量取舍: 全池齐跌且持仓最抗跌时无处可切, 保持持有)。
            # 这意味着实际亏损可能显著超过 stop_loss 线 (如 2026-06 纳指 -9.79%)。
            # 止损卖出后该标的进入 stop_cooldown 冷却, N天内不会被买回。
            gap = ms[ti] - ms[pos]; met = (streak >= cfg.confirm_days)
            # 纳指放量门: 今日不许【切进】纳指。止损/急跌仍可执行, 但落空仓而不是接纳指。
            _nz_gated = nz_vol_block(ti, t)
            _gated_pos = -1 if _nz_gated else ti
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
            if (cfg.panic_drop > 0 or cfg.panic_vol_on) and _panic_hit(pos, t):
                # 单日急跌 (固定阈值 或 波动缩放阈值, N-7): 切向当前更优候选
                sw = True; np_pos = _gated_pos; streak = 0; reason = "单日急跌"
            if not sw and (pnl <= stop_line(pos, t) or trail_hit(pos, t) or hl_exit(pos, t)):
                sw = True; np_pos = _gated_pos; streak = 0
                if pnl <= stop_line(pos, t):
                    reason = "止损"
                elif trail_hit(pos, t):
                    reason = "移动止损"
                else:
                    reason = "结构离场"
            elif not sw and cfg.slope_gate_mode == "exit" and _slope_gate_hit(pos, t):
                # N-40 斜率离场: 动量仍在但斜率转负 → 切向当前更优候选(无更优则清仓)
                # 注意: 仅在 ti != pos 分支; 持仓仍是第一时由下方"斜率补盲区"处理
                sw = True; np_pos = _gated_pos; streak = 0; reason = "斜率离场"
            elif _nz_gated:
                pass  # 纳指放量门: 今日不主动切进纳指 (止损/急跌出口已在上面按 _gated_pos 处理)
            elif cfg.rank_sticky > 0 and int(np.sum(ms > ms[pos])) + 1 <= cfg.rank_sticky:
                pass  # 排名缓冲: 持仓仍在前N名, 不切换
            elif not trend_ok(ti, t):
                pass  # 趋势门: 候选低于自身长均线, 不切换
            elif (etf_names[ti] == "纳指" and (
                    _panic_block(ti, t, vix_arr, cfg.vix_gate)
                    or _panic_block(ti, t, skew_arr, cfg.skew_gate)
                    or _panic_block(ti, t, rate_shock_arr, cfg.rate_shock_gate))):
                pass  # VIX/SKEW/美债利率冲击门: 美股恐慌期不切向纳指
            elif _share_block(ti, t):
                pass  # 份额风险门: 宽基篮子净赎回期不切向A股标的
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
        elif (cfg.trail_stop > 0 and pos >= 0 and t >= crash_held_until
                and trail_hit(pos, t)):
            # 移动止损补盲区: 持仓仍是动量第一(无处可切)时, 也从高点回撤线清到空仓
            sw = True; np_pos = -1; streak = 0; reason = "移动止损"
        elif (cfg.exit_hl_win > 0 and pos >= 0 and t >= crash_held_until
                and hl_exit(pos, t)):
            # N-32 结构离场补盲区: 持仓仍是动量第一(无处可切)时, 距N日高点回撤/破前低 → 清到空仓
            sw = True; np_pos = -1; streak = 0; reason = "结构离场"
        elif (cfg.slope_gate_mode == "exit" and pos >= 0 and t >= crash_held_until
                and _slope_gate_hit(pos, t)):
            # N-40 斜率离场补盲区: 持仓仍是动量第一(无处可切)时, 斜率转负 → 清到空仓
            sw = True; np_pos = -1; streak = 0; reason = "斜率离场"
        elif (cfg.abs_stop_loss and pos >= 0 and pnl <= stop_line(pos, t)
                and t >= crash_held_until):
            # 绝对止损: 持仓仍是动量第一(ti==pos)时的盲区补丁——
            # 有更优候选时走上面的"止损"切换; 无候选可切时清到空仓,
            # 不再硬扛 (2026-06 纳指 -9.79% 远超 -5% 线的成因)
            sw = True; np_pos = -1; streak = 0; reason = "绝对止损"
        elif (cfg.panic_drop > 0 or cfg.panic_vol_on) and pos >= 0 and t >= crash_held_until:
            if _panic_hit(pos, t):
                sw = True; np_pos = -1; streak = 0; reason = "单日急跌"
        elif (cfg.accel_enabled and pos >= 0
                and etf_names[pos] not in {x.strip() for x in cfg.accel_exempt_names.split(",") if x.strip()}
                and not np.isnan(accel_line(pos, t))
                and mom10_mat[pos, t] > accel_line(pos, t)):
            sw = True; np_pos = -1; streak = 0; reason = "加速止盈"
            accel_exit = True

        # ── 永不空仓 (v9式避险接盘, 默认关): 去现金决策 → 改持避险池 score 最强 ──
        # 覆盖: 上证空仓/绝对动量门/牛池退出/绝对止损/加速止盈/VIX异动/单日急跌 → 原为现金;
        # 空仓无候选时主动买入避险池; 抄底锁仓期内的止损退出不拦截 (保留锁仓语义)
        if cfg.never_empty and safe_idx.size and t >= crash_held_until:
            _si = _safe_best(t)
            if sw and np_pos == -1 and reason != "止损":
                if _si >= 0:
                    if _si == pos:
                        sw = False; reason = ""
                    else:
                        np_pos = _si; reason = "避险接盘(%s)" % reason
            elif not sw and pos == -1 and _si >= 0:
                sw = True; np_pos = _si; streak = 0; reason = "避险接盘"

        if sw:
            cost = cfg.commission + cfg.slippage
            crash_alloc_now = 0.0   # 任何正常动作清理抄底叠加仓, 归并为新主仓全仓
            sat_pos = -1; sat_alloc = 0.0; sat_hs = -1   # 任何全仓动作清理分档组合双腿
            if pos >= 0: nav *= (1 - cost)
            elif cfg.cash_park_code: nav *= (1 - cost) # 空仓停泊: 卖出停泊标的计费
            if np_pos >= 0: nav *= (1 - cost); hs = t; hn = nav
            old_pos = pos; pos = np_pos
            # ── B4: 抄底仓被止损 → 上锁, 直到恐慌条件解除(crash_now 转 False)才准再抄底 ──
            if (getattr(cfg, "crash_reentry_gate", False) and pos_from_crash
                    and old_pos >= 0 and reason in ("止损", "绝对止损", "移动止损")):
                crash_wait_clear = True
            pos_from_crash = False
            if (accel_exit and old_pos >= 0 and cfg.accel_cooldown > 0
                    and (not cfg.accel_cooldown_gold_only or etf_names[old_pos] == "黄金")):
                accel_block_idx = old_pos
                accel_block_until = t + cfg.accel_cooldown
            if accel_exit and old_pos >= 0:
                tp_recent[old_pos] = t          # N-37: 记录止盈日, 供震荡期禁止快买用
            if accel_exit and old_pos >= 0 and cfg.accel_reentry_gate:
                accel_gate[old_pos] = True
            if (reason in ("止损", "绝对止损") and old_pos >= 0
                    and cfg.stop_cooldown > 0):
                stop_block_until[old_pos] = t + cfg.stop_cooldown
            if (reason == "斜率离场" and old_pos >= 0
                    and cfg.slope_gate_cooldown > 0):
                slope_block_until[old_pos] = t + cfg.slope_gate_cooldown
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
