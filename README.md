# ETFMomentum

跨资产 ETF 动量轮动策略回测平台。

## 功能

- **动量轮动**：跨资产 ETF 池（纳指、黄金、A 股宽基等）日频动量轮动，持有最强标的
- **风控模块**：上证空仓过滤、止损、最短持有、确认天数、加速止盈、恐慌抄底
- **可视化回测**：Flask + ECharts，净值曲线 / 回撤 / 年度收益 / 持仓热力图，支持四图联动缩放
- **策略分析**：内置经典动量策略（GEM / DAA / VAA / TSMOM）对比与改进方向分析页
- **成本敏感性**：手续费 + 滑点独立参数，支持压力测试

## 目录结构

```
etf_strategy/
├── strategy/      # 核心信号逻辑 (signal.py)
├── data/          # Tushare 数据接口 (feed.py)
├── backtest/      # 回测引擎
├── live/          # 实盘
├── core/          # 状态与执行
└── web/           # Web 界面 (Flask + ECharts)
    ├── app.py         # 路由
    ├── bt_runner.py   # 回测编排
    ├── templates/     # 页面模板
    └── static/        # CSS / JS / ECharts
```

## 运行

```bash
# 配置 token
cp web/config_example.py web/config_local.py
# 编辑 web/config_local.py 填入 Tushare token

# 启动
PYTHONPATH=. python web/app.py
# 访问 http://localhost:8080/
```

## 依赖

Python 3.6+ / Flask / numpy / pandas / matplotlib / tushare
