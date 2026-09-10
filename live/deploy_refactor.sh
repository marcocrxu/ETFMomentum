#!/bin/bash
# 一键部署 2026-09-11 实盘链路重构 —— 所有需要手工的事项都在这
# 用法 (只需这一条命令, 不要加 sudo, 脚本会在需要处自行提权):
#   bash /home/ecs-user/etf_strategy/live/deploy_refactor.sh
set -e

if [ "$(id -u)" = "0" ]; then
    echo "❌ 请不要用 sudo/root 直接跑本脚本 (会把 bot 拉成 root 进程)"
    echo "   正确用法: bash $0"
    exit 1
fi

echo "== 1/5 重启 web/bot 服务 (加载最新代码) =="
sudo systemctl restart etf-web stockbot-web wechat-bot
echo "   etf-web / stockbot-web / wechat-bot 已重启"

echo "== 2/5 清理 QQ bot: 杀掉所有实例 (含 root 误启的), 以当前用户重启一个 =="
# root 启的实例需要 sudo 才能杀
ROOT_PIDS=$(ps -eo pid,user,args | grep "services/qq_bot_service.py" | grep -v grep | awk '$2=="root"{print $1}')
for p in $ROOT_PIDS; do sudo kill "$p" 2>/dev/null && echo "   已杀 root 实例 $p"; done
# 本用户实例直接杀
USER_PIDS=$(pgrep -u "$(id -u)" -f "services/qq_bot_service.py" || true)
for p in $USER_PIDS; do kill "$p" 2>/dev/null && echo "   已杀旧实例 $p"; done
sleep 2
cd /home/ecs-user/stock-bot
nohup ./.venv/bin/python services/qq_bot_service.py >> qq_bot.log 2>&1 &
echo "   新实例 $! 已启动 (用户: $(whoami))"

echo "== 3/5 每日对账 cron (21:30, 幂等) =="
CRON_LINE='30 21 * * 1-5 cd /home/ecs-user/etf_strategy && /home/ecs-user/etf_strategy/.venv/bin/python -m live.reconcile >> /home/ecs-user/etf_strategy/live/reconcile.log 2>&1'
if crontab -l 2>/dev/null | grep -q "live.reconcile"; then
    echo "   已存在, 跳过"
else
    (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    echo "   已添加"
fi

echo "== 4/5 自检 =="
sleep 3
FAIL=0
for svc in etf-web stockbot-web wechat-bot; do
    if systemctl is-active --quiet "$svc"; then echo "   ✅ $svc active"; else echo "   ❌ $svc 未运行!"; FAIL=1; fi
done
QQ_N=$(pgrep -fc "services/qq_bot_service.py" || true)
if [ "$QQ_N" = "1" ]; then echo "   ✅ qq_bot_service 单实例运行中"
else echo "   ❌ qq_bot_service 实例数=$QQ_N (应为1)!"; FAIL=1; fi
if ps -eo user,args | grep "services/qq_bot_service.py" | grep -v grep | grep -q "^root"; then
    echo "   ❌ 仍有 root 启的 qq_bot!"; FAIL=1
fi
crontab -l 2>/dev/null | grep -q "live.reconcile" && echo "   ✅ 对账 cron 已就位" || { echo "   ❌ 对账 cron 缺失!"; FAIL=1; }
crontab -l 2>/dev/null | grep -q "cron_etf_momentum.py" && echo "   ✅ 14:50 实盘 cron 在 (现为共享引擎 shim)"
sleep 2
code=$(curl -sk -o /dev/null -w "%{http_code}" https://127.0.0.1:8082/live)
if [ "$code" = "200" ] || [ "$code" = "403" ]; then echo "   ✅ /live 已上线 (http=$code; 403=需解锁属正常)"; else echo "   ❌ /live 异常 (http=$code)"; FAIL=1; fi
code2=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8081/api/health)
[ "$code2" = "200" ] && echo "   ✅ stockbot 8081 健康" || { echo "   ❌ stockbot 8081 异常 (http=$code2)"; FAIL=1; }

echo "== 5/5 结果 =="
if [ "$FAIL" = "0" ]; then
    echo "🎉 全部就位。今天 14:50 推送 = 主策略共享引擎口径; 21:30 起每日对账。"
    echo "   实况页: https://121.41.49.23/live (或 mytrack 页顶部'今日实况', 可折叠)"
else
    echo "⚠️ 有失败项 (见上面 ❌), 把输出发给我"
    exit 1
fi
