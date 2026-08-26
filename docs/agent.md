# agent.md — `/root` 工作区 Agent 快速上手

> 面向后续接手此工作区的 AI Agent。目标：先分清“当前要操作哪个项目”，再遵守项目铁律与 DSH 运行环境约束，避免重复踩坑。

## 0. 最高优先级规则

1. **禁止怀疑用户浏览器缓存**：用户报告 UI/回测结果不对时，不要建议刷新缓存、Ctrl+Shift+R 或无痕模式。先用 `curl` 查 API、查服务端日志/代码/数据。该规则也固化在 `/root/.claude/projects/-root/memory/never-blame-cache.md` 与 `etf_strategy/CLAUDE.md`。
2. **禁止质疑用户记忆**：用户说“我记得 xx”时，先假设用户是对的，再验证；不要以“你可能记错了”开头。
3. **先定位项目**：`/root` 不是单一仓库，而是多项目工作区。开工前先判断用户说的是 `etf_strategy`、`stock-bot`、`backtest`、`pdfgen`、`gem_analysis`、音视频/ASR，还是 DSH 本身。
4. **保护私密信息**：`.bashrc`、`.dsh/settings.yaml`、`stock-bot/.env`、`etf_strategy/web/config_local.py`、`asr_transcribe.py` 等含密钥/token。不要打印、复制进日志或提交；引用时只说“包含密钥”。

## 1. 工作区地图

| 路径 | 定位 | 关键说明 |
|---|---|---|
| `/root/etf_strategy` | **主项目：ETF 动量轮动策略平台** | 已有更详细 `etf_strategy/AGENT.md`；核心信号 `strategy/signal.py`，Web/回测 `web/app.py`、`web/bt_runner.py`，数据 `data/feed.py`，实盘 `live/runner.py`，测试 `tests/`。 |
| `/root/stock-bot` | 股票/ETF Bot 与 Web API | Flask `app.py` 跑 **8081**；钉钉回调、股票查询、AI 对话、选股、ETF 动量接口；`cache.db` 约 1.6G，日志/缓存很大。 |
| `/root/backtest` | 旧版/独立回测与共享缓存 | `strategy_spec.md` 是旧 OO 架构说明；`cache/etf_momentum/` 被 `etf_strategy` Web/audit 复用，删除前必须确认。 |
| `/root/pdfgen` | Node 图表/PDF 工具 | `puppeteer`、`pdf-parse`、`pdfjs-dist`；用于图表截图、PDF 抽取/生成。 |
| `/root/gem_analysis` | 创业板分析 | `399006_daily.csv`、`next_day_hist.png`、`review/`。 |
| `/root/audio_chunks*`, `asr_transcribe.py`, `download_bilibili.js` | 音视频下载/转录 | B 站下载脚本、DashScope ASR 转录脚本；`asr_transcribe.py` 内含 API key，注意保密。 |
| `/root/.dsh` | DSH 运行数据 | `settings.yaml` 当前默认模型为 `kimi-coding/k3`；`storages/workspace.json` 记录 `/root` 与 `/root/etf_strategy` workspace；`sessions/**/session.jsonl.zstd` 是压缩会话日志。 |
| DSH checkout | `/root/.nvm/versions/node/v22.22.2/lib/node_modules/@deepseek-ai/dsh` | CLI/profile 启动器；不要把它误当当前项目源码。 |

## 2. DSH / 运行时记忆

- 当前 Web GUI：`http://127.0.0.1:3210`；这是已有服务，**不要启动替代服务器**来“更新这个页面”。DSH Web 由 `dsh --profile web --port 3210` 运行。
- DSH profile：`web`/`headless`/`tui`；`.bashrc` 有 `alias dshcli='dsh --profile tui'`，以及 `dsh-model kimi|deepseek` 切换函数。当前 `.dsh/settings.yaml` 指向 Kimi For Coding `k3`，API key 从 `KIMI_API_KEY` 环境变量读取。
- DSH 文件策略：`workspace-write`，可写 `/root`；审批策略 `ask`。仍应避免无谓改动系统路径；确需越权写 `/etc` 等位置时，要先确认。
- DSH 会话记忆不是纯文本：`.dsh/sessions/**/session.jsonl.zstd` 为 zstd 压缩；本机未发现 `zstd` CLI / Python `zstandard`。元数据可先看 `.dsh/storages/workspace.json` 与 `.dsh/storages/session_projcache.json`。
- 近期 DSH 会话标题显示主要上下文：ETF 动量策略回测/修复、QQQ 混合参数判断、钉钉机器人推送、SSH/免密、DSH TUI/源码阅读等。处理 ETF 问题时优先读取 `etf_strategy/AGENT.md` 与审计台账。

## 3. ETF 主项目必读摘要

完整规则见 `/root/etf_strategy/AGENT.md`；这里只列最容易踩坑的摘要。

- **唯一权威信号**：`strategy/signal.py`。回测与实盘共用；不要依赖/改动遗留 OO 架构 `strategy/engine.py`、`rules.py`、`core/`，它们未被现役代码 import。
- **端口以代码为准**：`etf_strategy/web/app.py` 监听 `127.0.0.1:8082`；`README.md`/`CLAUDE.md` 中 8080 是旧信息。生产入口统一 `https://8.219.57.219/`，HTTP 已禁用。
- **全站门禁**：除 `/mytrack/`、`/static/`、`/favicon.ico` 外都要 session；先访问正确 `/mytrack/<MYTRACK_KEY>` 种下 `authed` cookie。错误 key 一律 404，不暴露页面存在。
- **基线纪律**：任何策略默认值/新参数改动都要跑 pytest、基线回测、十年消融；负优化默认关闭并登记台账。禁止扩池；不要随手打开已否决开关。
- **防前视/手续费/对齐**：`pre_days`、全量历史派生矩阵、`actions` 语义、`nav_log` 对齐、信号日收盘成交、手续费已在 `compute_signals` 内扣，改动前先读 `etf_strategy/AGENT.md` §4。
- **数据源陷阱**：QQQ-CNY 周五缺失曾导致基线重标定；改 `data/usfeed.py` 或缓存后必须重跑基线。
- 常用命令：
  ```bash
  cd /root/etf_strategy
  PYTHONPATH=. python3 -m pytest tests/ -v
  PYTHONPATH=. python3 web/app.py   # 127.0.0.1:8082
  curl -s -X POST http://127.0.0.1:8082/api/bt/run -H 'Content-Type: application/json' \
    -d '{"universe":"纳指,黄金,创业","start_date":"2016-07-07","end_date":"2026-08-12"}'
  ```

## 4. stock-bot / 其他项目要点

### `/root/stock-bot`
- Flask 入口 `app.py`，监听 `0.0.0.0:8081`；启动时会拉起后台预扫描与阴跌放量扫描线程。
- 依赖很少：`flask==2.0.3`、`tushare==1.4.11`、`python-dotenv`、`markupsafe`；配置走 `.env`（`TUSHARE_TOKEN`、`SERVER_BASE_URL` 等）。
- 重要文件：`cache.db`（大约 1.6G SQLite）、`app.log`、`token_usage.jsonl`、`cron_etf_momentum.py`、`bot_wechat.py`、`qq_bot/`、`services/`。
- cron：工作日 14:50 运行 `cron_etf_momentum.py` 并写 `cron_etf.log`；另有 acme.sh 续期 cron。
- 对外路径：钉钉回调 `/api/dingtalk/callback`、图表 `/static/charts/`；ETF 回测桥接 `/etf-backtest` 与 `/api/bt/run` 会复用 `/root/etf_strategy`。

### `/root/backtest`
- 旧架构规格在 `strategy_spec.md`；更像历史/参考实现。注意 `cache/etf_momentum/` 与主项目共享，清理要谨慎。

### `/root/pdfgen`
- Node 项目，依赖 `puppeteer` 等；适合处理图表截图、PDF 解析、网页检查脚本。无测试脚本，`npm test` 会报错占位。

### HTTPS / nginx / 服务现状
- 观测到：主 nginx 监听 80/443/8080；另有一个临时 `nginx -c /root/nginx_https.conf` 监听 443；`etf_keepalive.sh` 正在运行；`etf_strategy` 的 `web/app.py` 监听 `127.0.0.1:8082`；`stock-bot/app.py` 监听 `8081`。
- `setup_https_permanent.sh` 声称会安装 `etf-web.service` 并固化 nginx，但当前观测 `systemctl status etf-web` 不存在；实际运行状态与脚本目标可能不一致。重启/固化前先 `ss -ltnp`、`ps`、`systemctl status`、`nginx -t` 核实。
- 证书：Let's Encrypt shortlived IP 证书，acme.sh cron 续期；ZeroSSL 已不支持 IP 标识。

## 5. 调试与操作习惯

- UI 不对：先 `curl` API 直接验证，再查服务端日志/数据/端口/反向代理；最后才考虑前端展示代码。不要说“刷新缓存”。
- 大目录防护：`/root` 下有超大文件与十万级文件（媒体、缓存、TinyTeX、node_modules、`.dsh` 等）。不要对整个 `/root` 做宽泛 `grep`/glob；始终限定目录。
- 修改前看状态：`etf_strategy` 是 git 仓库且有未提交改动；先 `git -C /root/etf_strategy status --short` / `git diff` 再改。
- 运行服务前先确认端口占用：`ss -ltnp | grep -E ':(8081|8082|3210|443|80|8080)'`。
- 写文件：优先小范围 `edit`；新建/整文件替换才用 `write`；读取现有文件后再覆盖。
- 回复风格：用户主要用中文；默认中文、直接、少客套。涉及风险/密钥/资金操作时，先说明影响再执行。

## 6. 快速选择入口

- 用户说“ETF / 动量 / 回测 / mytrack / yanyan / leaderboard / pdfcompare” → 去 `/root/etf_strategy`，先读 `AGENT.md`。
- 用户说“钉钉 / QQ / 微信 / 股票查询 / 选股 / token 用量 / 歌词游戏” → 去 `/root/stock-bot`。
- 用户说“旧回测架构 / 共享 ETF 缓存” → 看 `/root/backtest`。
- 用户说“PDF / 图表截图 / puppeteer / 页面检查” → 看 `/root/pdfgen`。
- 用户说“DSH / 这个 GUI / profile / agent 工具行为” → 看 `/root/.dsh` 与 DSH checkout；别重启 3210 替代服务。
