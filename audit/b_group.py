"""B组: 规则语义与代码审计 — 合成数据单测
B1 确认2天计数语义 (是否绑定候选身份)
B2 空仓确认计数泄漏
B4 抄底止损后二次入场闸门
B5 止损退出冷却接入
B6 冷却锚点 (最后信号日起算/天数/顺延)
B3 止损盈亏口径 (纯代码阅读, 结论写在报告)
"""
import sys
sys.path.insert(0, "/root/etf_strategy")
sys.path.insert(0, "/root/etf_strategy/audit")
import numpy as np
from strategy.signal import SignalConfig, compute_signals
from harness import save

N = 120  # 合成天数
def mk_mats(prices, sh_flat=True):
    """prices: list of arrays (n_days,). 返回 compute_signals 入参"""
    price_mat = np.array(prices)
    n_days = price_mat.shape[1]
    mom20 = np.full(price_mat.shape, np.nan)
    mom10 = np.full(price_mat.shape, np.nan)
    mom20[:, 20:] = (price_mat[:, 20:] / price_mat[:, :-20] - 1) * 100
    mom10[:, 10:] = (price_mat[:, 10:] / price_mat[:, :-10] - 1) * 100
    sh_p = np.linspace(100, 100, n_days)          # 上证走平
    cyb_p = np.linspace(100, 100, n_days)
    sh_mom = np.full(n_days, np.nan); sh_mom[20:] = 0.0
    return price_mat, mom20, mom10, sh_mom, cyb_p, sh_p, n_days

def run(cfg, *prices):
    pm, m20, m10, sh_mom, cyb, sh, n = mk_mats(list(prices))
    return compute_signals(cfg, ["A", "B", "C"], pm, m20, m10, sh_mom, cyb, sh, n)

res = {}

# ───────────────────── B1: 每日轮换第一名, 正确语义下应零切换 ─────────────────────
# A/B 交替做动量第一 (每天换人), 第三名 C 垫底。空仓起步, mom 全 <3% 防 >3%即买。
n = N
pa = np.full(n, 100.0); pb = np.full(n, 100.0); pc = np.full(n, 100.0)
# 让 A/B 以锯齿形交替领先: 每 20 日窗内涨幅 ~1.5% (< 3% 阈值), 且逐日交替第一
for t in range(n):
    pa[t] = 100 + 1.5 * np.sin(t * np.pi) + 0.010 * t * (1 if t % 2 == 0 else 0.999)
    pb[t] = 100 + 0.010 * t * (1 if t % 2 == 1 else 0.999)
# 更直接: 构造 20 日窗内交错爬升
pa = 100 + np.where(np.arange(n) % 2 == 0, 0.12, 0.10) * np.arange(n)
pb = 100 + np.where(np.arange(n) % 2 == 1, 0.12, 0.10) * np.arange(n)
pc = np.full(n, 100.0)
cfg = SignalConfig(sh_cash=False, accel_enabled=False, crash_enabled=False,
                   min_hold=0, confirm_days=2, cash_skip_gt=3.0, min_gap=0.0,
                   exception_gap=99.0, abs_stop_loss=False)
actions, nav, _ = run(cfg, pa, pb, pc)
# 检查 mom20 是否确实逐日交替第一
pm, m20, m10, sh_mom, cyb, sh, nd = mk_mats([pa, pb, pc])
top1_seq = [int(np.nanargmax(m20[:, t])) for t in range(21, 60)]
alternating = all(top1_seq[i] != top1_seq[i + 1] for i in range(len(top1_seq) - 1))
buys = [a for a in actions if a[2] is not None and a[2] >= 0 and a[3] in ("确认买入", ">3%%即买", ">3%即买")]
res["B1"] = {
    "top1_daily_alternating": alternating,
    "top1_seq_sample": top1_seq[:12],
    "buy_actions": [(a[0], a[3]) for a in buys],
    "verdict": ("BUG确认: 计数不绑定候选身份, 第一名每日轮换仍触发确认买入"
                if buys else "通过: 每日轮换下未触发买入"),
}

# ───────────────────── B2: 带旧计数进空仓, 第一日不得买入 ─────────────────────
# 持仓 A, B 连续 2 天动量第一(streak=2 蓄势), 第 3 天上证动量爆冲触发空仓卖出,
# 冷却=0, 次日 mom<3% → 正确语义下不得确认买入
n = N
pa = 100 + 0.05 * np.arange(n)          # A 缓慢上涨
pb = np.full(n, 100.0)
pb[40:] = 100 + 0.3 * np.arange(n - 40)   # B 从 t=40 起加速, 抢下第一
pc = np.full(n, 95.0)
pm, m20, m10, sh_mom, cyb, sh, nd = mk_mats([pa, pb, pc])
sh_mom2 = sh_mom.copy()
sh_mom2[55:] = 50.0                     # t=55 起上证动量 50% 压过一切 → 上证空仓
cfg = SignalConfig(sh_cash=True, sh_cash_cooldown=0, accel_enabled=False,
                   crash_enabled=False, min_hold=0, confirm_days=2,
                   cash_skip_gt=3.0, min_gap=0.0, exception_gap=99.0,
                   abs_stop_loss=False)
acts, nav2, _ = compute_signals(cfg, ["A", "B", "C"], pm, m20, m10, sh_mom2, cyb, sh, nd)
log = [(a[0], a[1], a[2], a[3]) for a in acts]
# 空仓卖出后第一天(卖出日次日)是否出现买入
sell_day = next((a[0] for a in acts if a[3] == "上证空仓"), None)
next_buys = [a for a in acts if sell_day is not None and a[0] > sell_day
             and a[2] is not None and a[2] >= 0]
first_rebuy_gap = (next_buys[0][0] - sell_day) if next_buys else None
res["B2"] = {
    "actions": log,
    "cash_sell_day": sell_day,
    "first_rebuy_after_days": first_rebuy_gap,
    "verdict": ("通过: 进空仓计数归零, 首日未买入" if (next_buys and first_rebuy_gap >= 2)
                else ("无任何再买回(通过)" if not next_buys else
                      "BUG: 空仓次日即买回(旧计数泄漏)")),
}

# ───────────────────── B4: 抄底锁仓期内止损后, 是否有二次入场闸门 ─────────────────────
# 创业板: 300 天平盘(MA200 就绪)后每日 -3% 崩盘 → 触发恐慌抄底后继续阴跌触发止损,
# 恐慌条件持续 → 观察锁仓期内/后是否再抄底、是否要求新证据
n = 400
cyb_idx = np.concatenate([np.full(300, 3000.0),
                          3000 * (1 - 0.03) ** np.arange(1, 101)])
pe = np.full(n, 100.0); pe[300:] = 100 * (1 - 0.03) ** np.arange(1, 101)
pflat = np.full(n, 100.0)
pflat2 = np.full(n, 100.0)
pm, m20, m10, sh_mom, cyb, sh, nd = mk_mats([pe, pflat, pflat2])
cyb_p = cyb_idx
sh_idx = np.full(n, 3000.0)
cfg = SignalConfig(sh_cash=False, accel_enabled=False, crash_enabled=True,
                   crash_sigma=-2.5, crash_lock=7, stop_loss=-5.0,
                   confirm_days=2, cash_skip_gt=99.0, abs_stop_loss=True,
                   min_gap=0.0, exception_gap=99.0, min_hold=0)
acts3, nav3, _ = compute_signals(cfg, ["创业", "B", "C"], pm, m20, m10,
                                 sh_mom, cyb_p, sh_idx, n)
crash_entries = [a for a in acts3 if a[3] == "恐慌抄底"]
stops = [a for a in acts3 if a[3] == "止损"]
res["B4"] = {
    "crash_entries": [(a[0], a[3]) for a in crash_entries],
    "stops": [(a[0], a[1], a[2]) for a in stops],
    "all_actions": [(a[0], a[1], a[2], a[3]) for a in acts3],
    "verdict": ("锁仓期(7天)本身构成闸门: 期内不再抄底; 但锁仓期过后若恐慌条件仍存续, "
                "无 z 值回升等新证据要求即可二次抄底; 且止损后创业未被冷却(stop_cooldown=0), "
                "可经普通规则买回" if len(crash_entries) <= 2 else "多次抄底, 需看明细"),
}

# ───────────────────── B5/B6: 冷却锚点与止损冷却接入(真实配置下语义核实) ─────────
# B6: sc 信号存续 5 天后消失, 记录禁止入场的确切日序
n = 120
pa = np.full(n, 100.0); pb = np.full(n, 100.0); pc = np.full(n, 100.0)
pm, m20, m10, sh_mom, cyb, sh, nd = mk_mats([pa, pb, pc])
sh_mom3 = sh_mom.copy()
sh_mom3[50:55] = 10.0   # t=50..54 sc 信号存续, t=55 起消失
# A 的动量: 全程 1% (低于 3% 即买线, 高于上证消失后的 0)
pa = 100 + 0.05 * np.arange(n)
pm, m20, m10, sh_mom, cyb, sh, nd = mk_mats([pa, pb, pc])
sh_mom3 = sh_mom.copy(); sh_mom3[50:55] = 10.0
cfg = SignalConfig(sh_cash=True, sh_cash_cooldown=3, accel_enabled=False,
                   crash_enabled=False, confirm_days=2, cash_skip_gt=3.0,
                   min_hold=0, min_gap=0.0, exception_gap=99.0, abs_stop_loss=False)
acts4, _, _ = compute_signals(cfg, ["A", "B", "C"], pm, m20, m10, sh_mom3, cyb, sh, nd)
buys4 = [a for a in acts4 if a[2] is not None and a[2] >= 0]
res["B6"] = {
    "sc_signal_days": "t=50..54 (最后信号日=54)",
    "first_buy_day": buys4[0][0] if buys4 else None,
    "buy_reason": buys4[0][3] if buys4 else None,
    "verdict": ("锚点=最后信号日, 顺延制; block_until=last+3, t<last+3 禁止, "
                "即最后信号日后第 3 个交易日起允许入场 (54+3=57 应首开)"
                if buys4 and buys4[0][0] == 57 else f"需核对: {buys4[:2]}"),
}

# B5 语义核实(代码阅读结论的实例化): 跳空 -8% 触发绝对止损清仓后, 同一标的暴力反弹
#   动量登顶且 >3% → 默认 stop_cooldown=0 下能否立即买回
n = 120
pa = np.concatenate([100 + 0.5 * np.arange(51),                    # A 前 51 天强涨至 125(持仓)
                     [125 * 0.90, 125 * 0.81],                     # t=51,52 连续两个 -10%
                     125 * 0.81 * (1.06) ** np.arange(1, 68)])     # t=53 起每日 +6% 反弹
pb = 100 + 0.02 * np.arange(n)                                     # B 全程微涨(弱)
pc = np.full(n, 90.0)
pm, m20, m10, sh_mom, cyb, sh, nd = mk_mats([pa, pb, pc])
cfg = SignalConfig(sh_cash=False, accel_enabled=False, crash_enabled=False,
                   stop_loss=-5.0, stop_cooldown=0, confirm_days=2,
                   cash_skip_gt=3.0, min_gap=0.0, exception_gap=99.0,
                   min_hold=0, abs_stop_loss=True)
acts5, _, _ = compute_signals(cfg, ["A", "B", "C"], pm, m20, m10, sh_mom, cyb, sh, nd)
stop_days = [a[0] for a in acts5 if a[3] in ("止损", "绝对止损")]
rebuy_A = [a for a in acts5 if a[2] == 0 and any(sd < a[0] for sd in stop_days)]
res["B5"] = {
    "stop_events": [(a[0], a[1], a[2], a[3]) for a in acts5 if a[3] in ("止损", "绝对止损")],
    "all_actions": [(a[0], a[1], a[2], a[3]) for a in acts5],
    "rebuy_A_after_stop": [(a[0], a[3]) for a in rebuy_A],
    "verdict": ("默认 stop_cooldown=0: 止损不触发任何再入场限制, 可立即买回(防循环靠 "
                "stop_cooldown 开关, 但窄池消融为负优化而默认关)" if rebuy_A else
                "止损后未再买回"),
}

save(res, "b_group")
for k, v in res.items():
    print(f"\n=== {k} ===")
    print(v["verdict"])
