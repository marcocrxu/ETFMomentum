#!/bin/bash
# 部署 2026-09-11 实盘链路重构: 重启服务 + 加每日对账 cron + 自检
# 用法: bash /home/ecs-user/etf_strategy/live/deploy_refactor.sh
# (脚本内部只对 systemctl 用 sudo, 其余均以当前用户 ecs-user 执行)
set -e

echo "== 1/4 重启 web/bot 服务 (加载新代码) =="
sudo systemctl restart etf-web stockbot-web wechat-bot
echo "   etf-web / stockbot-web / wechat-bot 已重启"

echo "== 2/4 重启 QQ bot (无 systemd 单元, 手动拉起) =="
QQ_PID=$(pgrep -f "services/qq_bot_service.py" || true)
if [ -n "$QQ_PID" ]; then
    kill "$QQ_PID" && sleep 2
    echo "   旧进程 $QQ_PID 已停止"
fi
cd /home/ecs-user/stock-bot
nohup ./.venv/bin/python services/qq_bot_service.py >> qq_bot.log 2>&1 &
echo "   新进程 $! 已启动"

echo "== 3/4 添加每日对账 cron (21:30, 幂等) =="
CRON_LINE='30 21 * * 1-5 cd /home/ecs-user/etf_strategy && /home/ecs-user/etf_strategy/.venv/bin/python -m live.reconcile >> /home/ecs-user/etf_strategy/live/reconcile.log 2>&1'
if crontab -l 2>/dev/null | grep -q "live.reconcile"; then
    echo "   已存在, 跳过"
else
    (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    echo "   已添加: $CRON_LINE"
fi

echo "== 4/4 自检 =="
sleep 3
for svc in etf-web stockbot-web wechat-bot; do
    systemctl is-active --quiet "$svc" && echo "   ✅ $svc active" || echo "   ❌ $svc 未运行!"
done
pgrep -f "services/qq_bot_service.py" > /dev/null && echo "   ✅ qq_bot_service 运行中" || echo "   ❌ qq_bot_service 未运行!"
crontab -l 2>/dev/null | grep -q "live.reconcile" && echo "   ✅ 对账 cron 已就位" || echo "   ❌ 对账 cron 缺失!"
crontab -l 2>/dev/null | grep -q "cron_etf_momentum.py" && echo "   ✅ 14:50 实盘 cron 在 (现在跑的是共享引擎 shim)"
code=$(curl -sk -o /dev/null -w "%{http_code}" https://127.0.0.1:8082/live)
[ "$code" = "200" ] || [ "$code" = "403" ] && echo "   ✅ /live 页面可访问 (http=$code; 403=需先解锁属正常)" || echo "   ❌ /live 异常 (http=$code)"

echo
echo "完成。今天 14:50 的实盘推送将走共享引擎 (strategy/signal.py) 口径。"
echo "实况页: https://121.41.49.23/live (需先用 mytrack 私密地址解锁一次)"
