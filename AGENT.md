# AGENT.md — ETF 动量轮动策略项目：Agent 快速上手

> 写给接手的 AI Agent。目标：10 分钟内搞清架构、数据流、铁律与陷阱，避免踩已知坑、重做已否决的实验。

## 1. 项目定位

跨资产 ETF **20 日动量轮动**策略的回测/实盘平台。窄池 3 标的（**纳指 513100 / 黄金 518880 / 创业 159915**）每日排名，持有动量最强标的。回测与实盘**共用同一套信号函数** `strategy/signal.py`（唯一权威引擎）。

**基线成绩**（2016-07-07 ~ 2026-08-12，QQQ 0.7 混合默认开）：总收益 **3729.95%**、maxDD **-18.08%**、Calmar **2.51**、夏普 1.73、253 笔（纯 ETF 口径 3595%/2.09/283 笔，显式 `nz_mom_qqq_on=False` 可复现）。任何改动都应与这个基线做十年消融对比。
（2026-08-21 重标定：修复 QQQ-CNY 数据源周五缺失后基线从 3683.00/2.43 → 3729.95/2.51，见 §7 已知陷阱。）

## 2. ⛔ 铁律（违反即不可接受）

1. **禁止怀疑用户缓存**：用户报告 UI/回测结果不对时，**永远不要**建议"刷新浏览器缓存 / Ctrl+Shift+R / 无痕模式"。用 curl 查 API、查服务端代码和数据。历史上每次真正根因都是代码/数据 bug。
2. **禁止质疑用户记忆**：用户说"我记得 xx"时，先假设用户对，去验证，不要用"你可能记错了"开头。
3. **改动默认值/新参数必须过消融**：任何策略开关加入后先跑十年消融（vs 基线 3595/2.09），负优化则默认关闭并写入试验台账。本项目大量开关就是如此默认关掉的——**不要随手打开已否决的开关**（见 §7）。
4. **禁止扩池**：3→4→8→14 池单调恶化（3595/2.09 → 440/0.44），窄池 3 即最优，已写入纪律（F4）。
5. **对账基准 = 当前默认口径**：实盘对账页（mytrack）的策略曲线一律用当前默认参数回放，**不做**"当时规则"的点时间回放（用户决议 2026-08-20：策略调整过就按调整过的来，不搞历史口径拼接自欺欺人）。改动默认值后曲线整体重算即可。

## 3. 目录结构与角色

| 路径 | 角色 | 备注 |
|---|---|---|
| `strategy/signal.py` | **核心信号引擎**（833 行） | 回测/实盘共用，唯一权威。含 `SignalConfig` 全部开关、`compute_signals`、派生矩阵预计算函数 |
| `strategy/momentum.py` | 动量计算方法类（Pct/OLS/EWMA） | OLS 口径与生产 cron 同 |
| `data/feed.py` | 数据层：Tushare 历史 + 腾讯实时 + 增量缓存 | 复权在读取时现场计算 |
| `web/bt_runner.py` | 回测编排（454 行） | `run_backtest(params)` → JSON（含图 base64、decisions、热力图） |
| `web/app.py` | Flask 路由 | 端口 **8082**（注意：CLAUDE.md 写的 8080 已过时）；`/mytrack/<key>` 私密对账页 |
| `web/mytrack.py` | 实盘对账：手工记账 → 实盘净值 vs 策略同窗口 | 数据在 `web/mytrack_data.json`（已 gitignore） |
| `web/pdfcompare.py` | 同行策略 PDF 对比（2026-08-25 新增） | 上传 PDF 落盘 `web/uploads/`(gitignore) → PyPDF2 提取 → claude(kimi k3) 结构化解析 → 对比表；「一键回测」复用 leaderboard 队列。任务存档 `web/pdfcompare_jobs.json`(gitignore) |
| `web/templates/peer_v9.html` + `/peer/v9` 页 | **同行深度对比（2026-09-29 重写）** | 对手已从 v9 迭代到 **V12-R2**（仓库 github.com/zzhu41/stock）。同窗重跑我方首次落后 ~10pp；对方本人撤回前复权口径并自审承认无干净样本外。档案见 `audit/审计结果清单.md` **N 组**；脚本 `audit/n_peer_window.py`；结果 `audit/out/n_peer_window.json`。**引用对方任何旧数字前先看 N-2（口径作废）** |
| `backtest/v9_official/` | 对方 v9 官方复现包（2026-08-25） | **已过时**，顶部 NOTE.md 有更新说明；旧数字（48.79% 等）属已撤回的前复权口径，勿再引用 |
| `live/daily_run.py` | **生产实盘唯一入口（cron 14:50）** | 刷新 QQQ → LiveRunner → 推送 → 完整落盘 → 更新实际持仓。生产口径 = `SignalConfig()` 全默认值；`--no-push` 干跑不写状态 |
| `live/runner.py` | 实盘 runner（被 daily_run 调用） | 腾讯实时价写今日行 → 调 `compute_signals`（320天预热+pre_days+pre_mats，与 bt_runner 逐位一致，2026-09-11 实测同输入同输出） |
| `live/notify.py` | 推送与归档 | 钉钉/微信发送 + **完整消息落盘** `live/notify_log.jsonl` |
| `live/query.py` | 只读查询视图（60s 缓存） | stock-bot 的 bot 应答/HTTP API 唯一数据源，**禁止另行实现策略判断** |
| `live/reconcile.py` | 每日对账（cron 21:30，见 §8） | 收盘回放仓位 vs 14:50 决策 vs 实际持仓 + QQQ 新鲜度，不一致即钉钉告警 |
| `web/live_page.py` + `/live` 页 | 实盘实况页 | 今日动量/信号/操作建议/仓位对比/对账状态；纯展示装配，数据来自 `live/query.py` |
| `state.json` | 实际持仓记录（含股数） | 从 stock-bot/momentum_state.json 迁移；成交明细记账在 `web/mytrack_data.json` |
| `backtest/runner.py` | 独立回测 runner | 同样调 `signal.py`，被 stock-bot 复用 |
| `strategy/engine.py` + `rules.py` + `core/` | **遗留 OO 架构，未被任何现役代码 import** | 不要改、不要依赖；改动 signal.py 不用同步它们 |
| `audit/` | 审计脚本 + 结论 | `harness.py` 复刻 bt_runner 数据准备；`[a-f]_group.py` 各审计组；结果 `audit/out/*.json` |
| `tests/` | pytest，**全合成数据、无需网络** | 29 个测试全绿；`integration` marker 留给需 tushare 的测试 |
| `web/config_local.py` | 本地私密配置（gitignore） | `TUSHARE_TOKEN`、`MYTRACK_KEY` |

**共享缓存目录**：`/home/ecs-user/backtest/cache/etf_momentum/`（web、mytrack、audit harness、live 都指向这里）。`/home/ecs-user/stock-bot` 是关联的实盘 bot 项目（`live/notify.py` 推送依赖它的钉钉/微信服务与 .env）。

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

### 5.1 全站门禁与 HTTPS（2026-08-21 上线）

- **门禁**：`web/app.py` 的 `before_request` — 除 `/mytrack/`、`/static/`、`/favicon.ico` 外全部路由要求 session cookie；只有访问 `/mytrack/<MYTRACK_KEY>`（或任意带正确密钥的 mytrack 路由）才种下 `authed` cookie。cookie：HttpOnly + SameSite=Lax + **Secure**（仅 HTTPS）。
- **HTTPS**：Let's Encrypt shortlived **IP 证书**（`acme.sh --issue -d 121.41.49.23 --webroot /home/ecs-user/acme_webroot --cert-profile shortlived`，约 6 天有效，ARI 自动续期）。正式入口 **`https://121.41.49.23/`**（本机 EIP，元数据可验；`8.219.57.219` 是**已停用的旧机器**，勿再引用），nginx 443 → 8082/8081；永久配置 `/etc/nginx/conf.d/etf_https.conf`。
- 注意：**HTTP 通道已禁用**。`web/app.py` 应用层对 http 请求一律 403「HTTP 通道已禁用」（钉钉回调走 8081 不受影响）；`setup_https_permanent.sh` 会在 nginx 层把 80 整体断连（return 444）、8080 仅保留 `/api/dingtalk/` 与 `/static/charts/`（钉钉机器人回调+回复图片），其余路径 444 断连。用户入口统一 `https://8.219.57.219`。
- 签发 IP 证书注意事项：ZeroSSL 已不支持 IP 标识；LE 需 `--cert-profile shortlived`（webroot 走 80 端口的 `/.well-known/acme-challenge/`）。
- **⛔ 钉钉回调 = HTTP + 8080（2026-09-30 定案，别再试 https）**：钉钉机器人的"出网/消息接收地址"实际是 `http://121.41.49.23:8080/api/dingtalk/callback` —— 走的是**群自定义机器人的"出网"老机制**（该能力在现行官方文档里**已无条目**：537 页开发文档 + 1165 页帮助文档索引搜 `出网`/`outgoing` 均 0 命中；文档只覆盖 Stream 模式与**企业机器人 HTTP 模式**，后者要求"公网可访问的**HTTPS** 服务地址"且**必须发布应用**）。因此 nginx 必须始终保留 **listen 8080 的 `/api/dingtalk/` 入口**（老机 8.219.57.219 就有这个块 —— `POST http://8.219.57.219:8080/api/dingtalk/callback` 实测 502，说明 nginx 在、后端停；本机重构后丢了这块 → 钉钉回调静默失效，改 IP/改 https 都不通）。自检：`curl -X POST http://127.0.0.1:8080/api/dingtalk/callback -d '{}'` 应回 `{"errcode":1,"验证失败"}`。
- **一键脚本（2026-09-30 新增）**：`bash live/setup_https_bot.sh` —— 自提权、幂等，6 步：证书现状 → 按需安装 → **回调入口加固（80/8080/443 全接；访问/错误日志进 journald；`nginx -t` 失败自动回滚；已加幂等保护，重复跑不会产生重复 location）** → 装 root cron 自动续期 → 端到端自检 → 结论。`--check` 只体检不改动，`--renew` 强制续期，`--restart` 顺带重启 web/bot。**任何重写 nginx 配置的操作（如旧的 `/root/setup_https_permanent.sh`）之后，都要重跑本脚本（或 `--check`），否则 8080 入口可能再次丢失、钉钉又哑。**
- **陷阱（2026-09-30 事故，静默 20 天）**：续期任务以 ecs-user 身份跑，`--install-cert` 要写 `/etc/nginx/ssl/`（root 所有）→ 必然 `Permission denied`；而 acme.sh **失败时退出码仍是 0**，cron 又把输出丢进 `/dev/null` → 证书 09-10 过期后无人知晓（浏览器证书警告、https 不可用）。自动化必须放 **root cron**（脚本第 3 步自动装：每天 03:17 `acme.sh --cron`，日志 `~/.acme.sh/cron.log`）。另注：nginx 单元带 `PrivateTmp=yes`，访问日志**不能**写 `/tmp`（会落进私有命名空间，外面看不见），统一走 `syslog:...tag=nginx_dingtalk` 与 `tag=nginx_err` → 用 `journalctl -t nginx_dingtalk` / `-t nginx_err` 查。

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
| vix_gate / vix_spike_exit (VIX恐慌门) | 0(关) | gate=25 候选 | 2026-08-18 I-6：gate=25 三窗口微正(+73/+59/+274pp)；异动离场否决(卖在恐慌底)；gate=20 阈值敏感 |
| accel_cooldown / gold_only (止盈冷却) | **3 / True (默认开)** | ✅ 已默认开 | 2026-08-18 用户决议 I-17：黄金止盈后3天不买回；新默认基线 3729.95/2.51 (2026-08-21 数据修复后重标定)，25/26 +16pp |
| skew_gate (SKEW尾部风险门) | 0(关) | 150 候选 | 2026-08-18 I-7：SKEW>150 三窗口正(+288/+233/+418pp, Calmar +0.04~0.05)，优于VIX门；10Y-3M倒挂门否决 |
| vix_gate+skew_gate 组合 | 0(关) | 候选降级 | 2026-08-27 K-4：纯ETF口径最强(2.19)，但 QQQ 0.7 默认口径下已失效(-46pp, Calmar 2.18<2.51, DD 反而 -20.75)；QQQ混合已吸收其择时信息 |
| rate_shock_gate (美债利率冲击门, |10Y 20dΔ|>0.4pp 禁新买/切纳指) | **0 (关, 已入引擎+UI)** | 候选留档 | 2026-08-27 K-3：三窗口正(+950/+833/+140pp, Calmar 2.70/2.24/2.59, DD不变, 门0.4经bt_runner=4680.17)；被拦7笔直接效应≈-10pp, 大头路径级联；25/26完全惰性；实盘未接入(需treasury.csv刷新cron)；10Y-1Y倒挂门/水平门否决(K-1/K-2) |
| nz_mom_qqq_on/w (纳指动量混入QQQ-CNY) | **on=True / w=0.7 (QQQ权重)** | ✅ 已默认开 | 2026-08-18 用户决议 I-11：混合 vs 纯ETF 3729.95/2.51 vs 3058.85/1.98 (2026-08-21 数据修复后口径)；w=0.7=70%QQQ+30%ETF；精确值不可迁移勿再调优 |
| top_k / mom_skip | 1 / 0 | 开关留档 | 均实测负优化 |
| sh_cash_cooldown | 3 | 维持，但标记"孤峰" | A3：优势仅 +4.2%/+0.03 |
| accel_sigma | 3.0 | 峰，维持 | A1 平台可信 |
| 扩池 | — | **禁止** | F4 单调恶化 |
| panic_drop（固定阈值急跌离场） | 0（关） | **已否决，默认关** | 2026-08-25：3/4/5% 全窗负优化 |
| panic_vol_on（波动缩放急跌离场，同行 V12-C 口径） | False（关） | **已否决，默认关** | 2026-09-29 N-7：补测当年没测的**波动缩放**分支（`clip(K×VOL,2%,10%)`，含 V12-C 精确口径 `prior_max20_60×1.4`），3 窗 21 档无一跑赢基线（最好仍 -2.84pp、回撤深 7pp）。原因：纳指/黄金被 2% 下限钉死（59%/87% 交易日），退化为比 3% 更紧的固定止损。**1.5%~5% 整条带全负** |
| nz_vol_gate（纳指放量门：513100 量比>X 禁止新买纳指） | **0（关，已入引擎+bt_runner）** | **已否决，默认关** | 2026-10-22 N-27：体检证据极强（IC -0.213/-0.214 两段一致，五分位单调 -0.90，控溢价/控波动后仍成立，引擎逐笔 高量比 +0.99% vs 低量比 +5.30%），但**正式消融四档全否**：六起点最高只 3/6 胜、中位全负、gate=1.0 在 2024 起点 **-15.0pp**。结论：**统计显著性 ≠ 可交易性**（IC 是全体日的时序关系，闸门只在入场那一刻生效且会拦掉好交易）。协议 `audit/n_vol_gate_protocol.md`，结果 `audit/out/n_vol_gate_ablation.json` |
| trail_stop（移动止损/吊灯：从入场以来最高收盘回撤 X% 离场） | **0（关，已入引擎+bt_runner）** | **已否决，默认关** | 2026-10-22 N-29：主流趋势跟踪标配的退出方式，但六起点全否 （5% 中位 **-6.48pp**；8% -1.14；10% -1.26；12% 几乎不触发）。机制同 N-9/N-20：已有日频"不在场"机制（动量掉出就换、全池转弱空仓），叠加移动止损=对同一风险重复收费。协议 `audit/n_trail_stop_protocol.md`，结果 `audit/out/n_trail_stop_ablation.json` |
| 加仓入场时机 (追加本金) | **立即追当前持仓** | ✅ 研究结论 (2026-08-21) | 立即追 ≈ 10日均分 > 等下次调仓; 四个口径一致, 尾部不对称全压在"等调仓"侧; 持黄金时差别最小。见 `backtest/加仓时机分析.md` + `backtest/add_capital_entry.py` |

其他已否决：vol 归一化、regime 窗口解耦、EWMA/percentile 阈值口径、防御切换、多周期动量、风险调整动量、FIP、趋势门等——**未经用户同意不要重新启用**。

## 7. 已知陷阱（改动信号逻辑前必读）

- **B4 抄底死亡循环**（最高优先级立项，未完成）：连续暴跌下"抄底→止损→7 天后锁仓期满→再抄底→止损"循环出血（F1/F2 实证单轮 -5~10%）。立项方向：抄底止损后需 z 值回升等新证据才准再抄底。
- **止损是相对止损**：只有存在更优候选（ti≠pos）时才止损；绝对止损（默认开）覆盖 ti==pos 盲区。
- **reason 字符串 `%%` 显示 bug**：老 reason 字面量（">3%%切换"）在 Web UI 原样显示，顺手修单 %（立项 #5）。
- **空仓冷却锚点**：锚定最后信号日，信号存续期每天顺延 `t+3`；`t < block_until` 禁入场。
- **14:50 实盘口径**：实盘用腾讯实时价写今日行（含 MA200/动量），回测用收盘价——尾盘漂移已量化（C2/C3：0.3% 档贴线日 ~4.9%），对账时注意。
- **QDII 溢价**：纳指溢价/黄金价差无指数数据可 diff，实盘监控项（D2/G8），回测不建模。
- **git 工作区**：`strategy/signal.py`、`web/bt_runner.py` 有未提交改动；`audit/`、`.claude/` 未跟踪。改动前先 `git diff` 了解现状。
- **QQQ-CNY 数据源周五缺失（2026-08-21 已修）**：`data/usfeed.py` 旧 inner join 让每周五的 QQQ 收盘无对应汇率行被丢弃（~20% 交易日缺失），且刷新永远落后一天 → 实盘推送持续"数据未就绪"回退纯ETF。已改 `merge_asof(backward)`；**基线因此重标定 3683.00/2.43 → 3729.95/2.51**，纯ETF口径(3595.1/2.09)不受影响。改数据源后必须重跑基线。
- **Yahoo 被墙的兜底（2026-09-11 已修）**：本机连不上 Yahoo，`refresh_qqq_cny()` 自动切 新浪美股(QQQ)+中国货币网(USDCNY中间价)。中间价与 Yahoo 现货有稳定价差，兜底汇率**按重叠期比值中位数重定基**后才追加，防止接缝假跳变污染动量。
- **实盘/回测口径一致性（2026-09-11 对齐）**：`live/runner.py` 必须与 `bt_runner` 同窗口语义：320 天预热 + `pre_days` + `accel_th` 全量预计算切片。此前实盘从决策区起点取数，expanding 阈值少一年历史，曾造成操作日期差 1 天。
- **对账回放窗口短窗陷阱（2026-10-12 已修）**：`live/reconcile.py` 曾用"今天-120天"跑回放，expanding σ 样本从 2686 个交易日缩到 274 个 → 恐慌抄底触发线抬高 2.47pp（-8.35% → -5.88%）→ 09-28 上证 mom10（-6.10%）假触发"恐慌抄底"买创业，09-29/09-30/10-09 连报 3 天假"口径分歧"，10-12 假腿止损回纳指后期末仓位撞回一致、对账转绿把分叉掩盖。现固定 `REPLAY_START = LiveRunner.DECISION_START` + `replay_window_problem()` 窗口体检（<2000 交易日即告警且不做仓位对比），`tests/test_production_link.py` 钉死。**任何回测/对账/审计的起点都不许按"最近 N 天"临时截，阈值类指标（恐慌抄底/加速止盈/自适应止损）全部吃样本长度。**

## 7b. 2026-09-10 实盘分叉事故（血泪教训，架构红线）

**事故**：stock-bot/cron_etf_momentum.py 是 8 月中旬从主策略手工抄的副本。之后主策略迭代（加速止盈/恐慌抄底/绝对止损/空仓冷却/QQQ 混合）均未同步，07 月起实盘通知与主策略分叉；09-02 Yahoo 断连后副本静默降级纯 ETF 口径，**09-09 丢失"上证空仓"信号**（主策略：卖出黄金锁定 +7.72%；副本通知：继续持有）。同时 stock-bot 的 bot 应答/API 还各有一套简化判断，群里发过互相矛盾的指令。

**红线（违反即不可接受）**：
1. 策略逻辑只允许存在一份：`strategy/signal.py`。任何"生产环境另写一份简化版"的做法都是事故温床。
2. 实盘决策唯一来源：`live/daily_run.py`（→ LiveRunner → compute_signals）。stock-bot/cron_etf_momentum.py 只是转发 shim（旧实现留档 `.deprecated-20260911`，禁止引用）。
3. 数据降级必须显式告警：QQQ-CNY 陈旧时消息红字标注，禁止静默回退。
4. 通知完整落盘 `live/notify_log.jsonl`，事后必须能还原用户收到的每一个字。
5. 每日对账 `live/reconcile.py`：收盘回放仓位 vs 实盘决策 vs 实际持仓，不一致当天告警。
6. `tests/test_production_link.py` 钉死上述结构，改动前先跑 pytest。
7. 对账回放窗口必须与实盘同源（`REPLAY_START = LiveRunner.DECISION_START`），禁止"今天-N天"临时截窗——短窗会把 expanding σ 阈值算歪，制造假信号和假分歧（2026-10-12 事故，见 §7）。

## 8. 协作习惯（与用户共事方式）

- 每次修改策略后：跑 pytest → 跑基线回测确认逐分一致 → 跑消融 → 在 `audit/审计结果清单.md` 或台账登记结论（哪怕是否决）。
- 用户报告"回测不对"：先 curl API + 查 `checks` 字段 + 查数据缓存，再下结论。
- 审计脚本风格：独立 `[a-z]_group.py` + `harness.py` 复用数据准备，输出 JSON 到 `audit/out/`，结论写回清单 md。

## yanyan 策略 (2026-08-17 新增, 用户备注 yanyan)

独立于主引擎的新策略, 单独页面 `/yanyan` + API `/api/yanyan/run`:
- `strategy/yanyan.py` 引擎 / `web/yanyan_runner.py` 回测编排 / `web/templates/yanyan.html` + `web/static/js/yanyan.js` 页面
- **体制层**: 沪深300 现价 < MA200 = 熊市 → A股池(7只)禁持仓, 只许 跨境(纳指/港股创新药)+黄金
- **动量层**: score = MOM20/VOL20; 进场=score第一且MOM20>0; 离场=持仓MOM20转负;
  轮动缓冲=挑战者MOM20超持仓2%; 过热=MOM20>40% 时离场收紧为 MOM5<0, 再进场须 MOM5>Q(3)
- 标的池 11 只: A股7 + 跨境2 + 黄金 + 货币511880(空仓停靠, 不入排名); 时变宇宙(NaN掩码)
- 基准成绩 (2016-07-07~2026-08-12): total +587.8% / 年化 21.9% / DD -26.6% / Calmar 0.82 / 仅 1 个负年
- 测试: tests/test_yanyan.py (5 个合成单测, 含牛熊开关/动量转负/轮动缓冲/过热闸门)
