# CLAUDE.md — ETF 动量轮动策略项目

## ⛔ 最高优先级规则 — 违反即不可接受

1. **禁止怀疑用户缓存**：当用户报告 UI/回测结果不对时，**永远不要**建议"刷新浏览器缓存"、"Ctrl+Shift+R"、"开无痕模式"。用 curl 直接查 API 返回值，查服务端代码和数据。用户已多次确认刷新过，每次真正根因都是代码/数据 bug。这条规则写在了 `/root/.claude/projects/-root/memory/never-blame-cache.md`，每次对话前必须回忆。

2. **禁止质疑用户记忆**：用户说"我记得 xx"时，先假设用户是对的，去验证。不要用"你可能记错了"开头。

## 项目架构
- `strategy/signal.py` — 核心信号（回测实盘共用）
- `web/bt_runner.py` — 回测引擎
- `web/app.py` — Flask (8080)
- `web/templates/index.html` — 回测页
- `web/templates/analysis.html` — 分析页
- `data/feed.py` — Tushare 数据缓存层
- `live/runner.py` — 实盘 runner (cron 14:50)
- `tests/` — pytest 测试框架

## 运行
- 网站: `python3 web/app.py` (端口 8080)
- 测试: `python3 -m pytest tests/ -v`
