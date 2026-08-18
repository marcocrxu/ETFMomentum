# AGENT.md — ETF 动量轮动策略项目：Agent 快速上手

> 写给接手的 AI Agent。目标：10 分钟内搞清架构、数据流、铁律与陷阱，避免踩已知坑、重做已否决的实验。

## 1. 项目定位

跨资产 ETF **20 日动量轮动**策略的回测/实盘平台。窄池 3 标的（**纳指 513100 / 黄金 518880 / 创业 159915**）每日排名，持有动量最强标的。回测与实盘**共用同一套信号函数** `strategy/signal.py`（唯一权威引擎）。

**基线成绩**（2016-07-07 ~ 2026-08-12，修正口径）：总收益 **3595%**、年化 **44.9%**、maxDD **-21.5%**、Calmar **2.09**、283 笔。任何改动都应与这个基线做十年消融对比。

## 2. ⛔ 铁律（违反即不可接受）

1. **禁止怀疑用户缓存**：用户报告 UI/回测结果不对时，**永远不要**建议"刷新浏览器缓存 / Ctrl+Shift+R / 无痕模式"。用 curl 查 API、查服务端代码和数据。历史上每次真正根因都是代码/数据 bug。
2. **禁止质疑用户记忆**：用户说"我记得 xx"时，先假设用户对，去验证，不要用"你可能记错了"开头。
3. **改动默认值/新参数必须过消融**：任何策略开关加入后先跑十年消融（vs 基线 3595/2.09），负优化则默认关闭并写入试验台账。本项目大量开关就是如此默认关掉的——**不要随手打开已否决的开关**（见 §7）。
4. **禁止扩池**：3→4→8→14 池单调恶化（3595/2.09 → 440/0.44），窄池 3 即最优，已写入纪律（F4）。

## 3. 目录结构与角色

| 路径 | 角色 | 备注 |
|---|---|---|
| `strategy/signal.py` | **核心信号引擎**（833 行） | 回测/实盘共用，唯一权威。含 `SignalConfig` 全部开关、`compute_signals`、派生矩阵预计算函数 |
| `strategy/momentum.py` | 动量计算方法类（Pct/OLS/EWMA） | OLS 口径与生产 cron 同 |
| `data/feed.py` | 数据层：Tushare 历史 + 腾讯实时 + 增量缓存 | 复权在读取时现场计算 |
| `web/bt_runner.py` | 回测编排（454 行） | `run_backtest(params)` → JSON（含图 base64、decisions、热力图） |
| `web/app.py` | Flask 路由 | 端口 **8082**（注意：CLAUDE.md 写的 8080 已过时）；`/mytrack/<key>` 私密对账页 |
| `web/mytrack.py` | 实盘对账：手工记账 → 实盘净值 vs 策略同窗口 | 数据在 `web/mytrack_data.json`（已 gitignore） |
| `live/runner.py` | 实盘 runner（cron 14:50） | 腾讯实时价写今日行 → 调 `compute_signals` → 钉钉/微信推送 |
| `backtest/runner.py` | 独立回测 runner | 同样调 `signal.py`，被 stock-bot 复用 |
| `strategy/engine.py` + `rules.py` + `core/` | **遗留 OO 架构，未被任何现役代码 import** | 不要改、不要依赖；改动 signal.py 不用同步它们 |
| `audit/` | 审计脚本 + 结论 | `harness.py` 复刻 bt_runner 数据准备；`[a-f]_group.py` 各审计组；结果 `audit/out/*.json` |
| `tests/` | pytest，**全合成数据、无需网络** | 29 个测试全绿；`integration` marker 留给需 tushare 的测试 |
| `web/config_local.py` | 本地私密配置（gitignore） | `TUSHARE_TOKEN`、`MYTRACK_KEY` |

**共享缓存目录**：`/root/backtest/cache/etf_momentum/`（web、mytrack、audit harness 都指向这里）。`/root/stock-bot` 是关联的实盘 bot 项目（live/runner.py 推送时依赖它）。

## 4. 核心数据流（改代码前必懂）

```
web/app.py (POST /api/bt/run)
  └─ web/bt_runner.py::run_backtest(params)
       1. 解析参数 → SignalConfig
       2. FEED.get_dataframe(codes, fetch_start, end)   # 起点往前预热 320 自然日(MA200)
       3. 全量时间轴 _all_full（含预热段）+ 用户窗口切片
          pre_days = 预热段天数
       4. 动量: mom20/mom10 在全量历史上算, 再切片到决策区 [pre_days:]
       5. pre_mats 派生矩阵(accel_th/mom60/vol/ma/park/up_ratio)在全量历史预计算后切片
       6. compute_signals(cfg, ..., pre_days=pre_days, pre_mats=pre_mats)
          → (actions, final_nav, nav_log)
       7. 决策记录/热力图/基准/图表/自检 checks → JSON
```

### 关键不变量（改坏会静默失真，不是报错）

- **防前视**：`compute_signals` 内 `t` 是决策区索引；恐慌抄底的阈值/MA200 用 `tf = t + pre_days` 访问**完整历史**序列（`cyb_idx_p`/`sh_idx_p` 传全量数组）。`pre_mats` 同理——在切片后数据上 expanding 会造成窗口前 60 天阈值 NaN、加速止盈/恐慌静默失效。
- **`actions` 语义**：`(day_idx, old_pos, new_pos, reason)`，`pos=-1` 为空仓；信号日**收盘**成交，当日收益归旧仓。**手续费已在 `compute_signals` 内部扣除**（切换双边各扣 `(1-commission-slippage)`，恐慌抄底部分仓位按 `(1-c·alloc)²`），外部不要再扣。
- **`nav_log` 逐日对齐**：无数据日也 append nav，长度 = 决策区天数；`bt_runner` 用 `_vs_user = (n_full - pre_days) - len(nav_log)` 回填前导 1.0。
- **有效起点 `vs`**：至少 2 个标的动量非 NaN 且 `sh_mom` 非 NaN 才开跑，不等最晚上市标的。
- **停牌处理**：价格 NaN → 动量 NaN → 不参与排名；持仓停牌时 nav 冻结、无成交（事实禁交易）。
- **复权**：`close_adj = close_raw × adj_factor / 最新factor`，缓存存原始价 + 因子，读取时现场算（C1 已验证）。

### Top-1 决策顺序（`compute_signals` 主循环，顺序即优先级）

1. 更新净值（含波动率缩放/空仓停泊/抄底组合仓计息）
2. 排名分数（可被 multi_mom/risk_adj/冷却期改写）→ top1 `ti`
3. **恐慌抄底**（mom10 指数 < μ-2.5σ 且 < MA200）→ 锁仓 7 天，期间只允许止损退出
4. 抄底锁仓期内 → 仅止损保护
5. **上证空仓**（上证 mom20 > 池内最强）→ 空仓 + 3 天冷却；豁免模式只对 A 股生效
6. 绝对动量门（全池负动量 → 空仓，默认关）
7. 空仓入场：>3% 即买 / 确认 2 天买入（经动量耗尽门/趋势门过滤）
8. 持仓切换：止损（有更优候选时）→ 排名缓冲 → 趋势门 → 快速通道(>3%) → 正常切换(≥1%, ≥10 天)
9. **绝对止损**（仍第一名但 pnl ≤ -5% → 清到空仓，默认**开**）
10. 加速止盈（mom10 > 自身历史 μ+3σ → 清仓，默认开）

## 5. 运行方式

```bash
cd /root/etf_strategy
PYTHONPATH=. python3 -m pytest tests/ -v        # 29 个测试, 全合成数据
PYTHONPATH=. python3 web/app.py                  # 端口 8082, http://127.0.0.1:8082
# 回测 API: curl -s -X POST http://127.0.0.1:8082/api/bt/run -H 'Content-Type: application/json' \
#   -d '{"universe":"纳指,黄金,创业","start_date":"2016-07-07","end_date":"2026-08-12"}'
# 审计快捷跑: PYTHONPATH=. python3 -c "from audit.harness import bt; import json; print(json.dumps({k:v for k,v in bt().items() if k in ('total_ret','ann_ret','max_dd','calmar','trade_count','checks')}, ensure_ascii=False))"
```

- 需要 `web/config_local.py`（Tushare token），否则数据层退化但测试不受影响。
- 改代码后必跑：**pytest 全绿 + 基线回测逐分不差**（基线 total_ret 应为 3595.13，Calmar 2.09）。

## 6. 已固化的结论与"别重做"清单

完整细节见 `audit/审计结果清单.md`（268 行，A–H 组）。核心条目：

| 开关/参数 | 默认值 | 状态 | 备注 |
|---|---|---|---|
| confirm_bind_identity | False | 负优化，留档关闭 | B1：确认计数不绑定候选身份的"bug"其实是更好语义，维持旧行为 |
| abs_stop_loss | True | 默认开 | 补"仍第一名时不止损"盲区（2026-06 纳指 -9.79% 成因） |
| stop_cooldown / accel_cooldown | 0 | 关闭 | 消融负优化，维持 |
| fast_min_hold | 0 | 否决 | E2 补遗：拖一天只亏更多 |
| fast_lane_staged (3%→50%/5%→100% 分档切换) | False | 否决 | 2026-08-17 补遗2：卖飞数据方向相反，3~5% 档是最优质档；全档位/双窗口大幅负优化 |
| fast_lane_ma (快速通道追涨闸门) | 0(关) | 否决默认开 | 2026-08-17 补遗4：被拦的低MA60追涨恰是历史最优档(83%胜率/+4.5%)，全档位负优化 |
| nz_mom_qqq_w (纳指动量混入QQQ-CNY) | 0(纯ETF=基线) | 候选 0.75 | 2026-08-17 补遗5：纯QQQ否决(2020-10 DD爆)，w=0.75 在5个窗口Calmar全胜/DD全降 |
| top_k / mom_skip | 1 / 0 | 开关留档 | 均实测负优化 |
| sh_cash_cooldown | 3 | 维持，但标记"孤峰" | A3：优势仅 +4.2%/+0.03 |
| accel_sigma | 3.0 | 峰，维持 | A1 平台可信 |
| 扩池 | — | **禁止** | F4 单调恶化 |

其他已否决：vol 归一化、regime 窗口解耦、EWMA/percentile 阈值口径、防御切换、多周期动量、风险调整动量、FIP、趋势门等——**未经用户同意不要重新启用**。

## 7. 已知陷阱（改动信号逻辑前必读）

- **B4 抄底死亡循环**（最高优先级立项，未完成）：连续暴跌下"抄底→止损→7 天后锁仓期满→再抄底→止损"循环出血（F1/F2 实证单轮 -5~10%）。立项方向：抄底止损后需 z 值回升等新证据才准再抄底。
- **止损是相对止损**：只有存在更优候选（ti≠pos）时才止损；绝对止损（默认开）覆盖 ti==pos 盲区。
- **reason 字符串 `%%` 显示 bug**：老 reason 字面量（">3%%切换"）在 Web UI 原样显示，顺手修单 %（立项 #5）。
- **空仓冷却锚点**：锚定最后信号日，信号存续期每天顺延 `t+3`；`t < block_until` 禁入场。
- **14:50 实盘口径**：实盘用腾讯实时价写今日行（含 MA200/动量），回测用收盘价——尾盘漂移已量化（C2/C3：0.3% 档贴线日 ~4.9%），对账时注意。
- **QDII 溢价**：纳指溢价/黄金价差无指数数据可 diff，实盘监控项（D2/G8），回测不建模。
- **git 工作区**：`strategy/signal.py`、`web/bt_runner.py` 有未提交改动；`audit/`、`.claude/` 未跟踪。改动前先 `git diff` 了解现状。

## 8. 协作习惯（与用户共事方式）

- 每次修改策略后：跑 pytest → 跑基线回测确认逐分一致 → 跑消融 → 在 `audit/审计结果清单.md` 或台账登记结论（哪怕是否决）。
- 用户报告"回测不对"：先 curl API + 查 `checks` 字段 + 查数据缓存，再下结论。
- 审计脚本风格：独立 `[a-z]_group.py` + `harness.py` 复用数据准备，输出 JSON 到 `audit/out/`，结论写回清单 md。
