#!/bin/bash
# ============================================================================
# 一键: HTTPS 证书自动续期 + 机器人回调链路加固/自检   (etf_strategy 实盘)
#
#   bash /home/ecs-user/etf_strategy/live/setup_https_bot.sh            # 一键修复/加固
#   bash /home/ecs-user/etf_strategy/live/setup_https_bot.sh --check    # 只体检(只读, 不需要 root)
#   bash /home/ecs-user/etf_strategy/live/setup_https_bot.sh --renew    # 强制立刻续期一次
#   bash /home/ecs-user/etf_strategy/live/setup_https_bot.sh --restart  # 顺带重启 web/bot 服务
#
# 设计原则: 需要 root 的步骤全部在脚本内部完成 (自提权), 不把"你自己去执行 XX"甩给用户。
#
# 做三件事:
#   ① 证书: 按需签发/安装, 并确保续期挂在 root cron (ecs-user 跑必然 Permission denied)
#   ② 回调入口加固: 不管钉钉后台填的是 https://IP/... 、http://IP/... 还是 http://IP:8080/...,
#      都接住(80 不再 301 跳转, 8080 补专用入口), 并落访问日志 /tmp/dingtalk_access.log
#      -> "钉钉到底有没有来" 从此有据可查
#   ③ 端到端自检: 证书链 / TLS / 回调路由 / 服务状态
# ============================================================================
set -uo pipefail

DOMAIN="121.41.49.23"
OWNER="ecs-user"
ACME_HOME="/home/ecs-user/.acme.sh"
ACME_SH="$ACME_HOME/acme.sh"
ACME_CERT="$ACME_HOME/${DOMAIN}_ecc/fullchain.cer"
CERT_DIR="/etc/nginx/ssl/$DOMAIN"
SSL_CERT="$CERT_DIR/fullchain.cer"
SSL_KEY="$CERT_DIR/$DOMAIN.key"
RENEW_LOG="$ACME_HOME/cron.log"
NGX_CONF="/etc/nginx/conf.d/etf_https.conf"
DING_LOG="journalctl -t nginx_dingtalk"
CRON_LINE="17 3 * * * $ACME_SH --cron --home $ACME_HOME --log $RENEW_LOG >/dev/null 2>&1"
CALLBACK_PATH="/api/dingtalk/callback"
SERVICES="etf-web stockbot-web wechat-bot"

MODE="fix"; DO_RESTART=0; ELEVATED=0
for a in "$@"; do
  case "$a" in
    --check)    MODE="check" ;;
    --renew)    MODE="renew" ;;
    --restart)  DO_RESTART=1 ;;
    --elevated) ELEVATED=1 ;;
    *) echo "未知参数: $a"; exit 2 ;;
  esac
done

FAIL=0
ok()  { echo "   ✅ $*"; }
bad() { echo "   ❌ $*"; FAIL=1; }
warn(){ echo "   ⚠️  $*"; }
hdr() { echo; echo "── $* ────────────────────────────────────────"; }

# ── 自提权: 脚本自己变成 root, 不甩给用户 ──────────────────────────────────
if [ "$MODE" != "check" ] && [ "$(id -u)" != "0" ] && [ "$ELEVATED" != "1" ]; then
  echo "需要管理员权限(证书/nginx/root cron), 正在提权..."
  if sudo -n true 2>/dev/null; then exec sudo -n -E bash "$0" --elevated "$@"; fi
  if sudo -v 2>/dev/null;      then exec sudo -E  bash "$0" --elevated "$@"; fi
  echo
  echo "❌ 无法自动提权。本脚本必须有一次 root 才能完成下面全部动作;"
  echo "   请用任一 root 入口重跑【本脚本】(不需要你手写任何命令):"
  echo "     bash $0"
  exit 3
fi
AS_ROOT=0; [ "$(id -u)" = "0" ] && AS_ROOT=1

# ── 工具 ───────────────────────────────────────────────────────────────────
days_left() {
  local f="$1" end epoch
  [ -r "$f" ] || { echo 999; return; }
  end=$(openssl x509 -in "$f" -noout -enddate 2>/dev/null | cut -d= -f2)
  [ -n "$end" ] || { echo 999; return; }
  epoch=$(date -d "$end" +%s 2>/dev/null) || { echo 999; return; }
  echo $(( (epoch - $(date +%s)) / 86400 ))
}
cert_end_epoch() {
  local f="$1" end
  [ -r "$f" ] || { echo 0; return; }
  end=$(openssl x509 -in "$f" -noout -enddate 2>/dev/null | cut -d= -f2)
  date -d "$end" +%s 2>/dev/null || echo 0
}
own_back() { [ "$AS_ROOT" = "1" ] && [ -e "$1" ] && chown "$OWNER:$OWNER" "$1" 2>/dev/null; return 0; }

echo "=================================================================="
echo " ETF 实盘 · HTTPS 证书 + 机器人回调 一键自检/修复/加固"
echo " 模式: $MODE   身份: $(id -un)   $(date '+%F %T')"
echo "=================================================================="

# ── 1. 证书现状 ────────────────────────────────────────────────────────────
hdr "1/6 证书现状"
NGINX_DAYS=$(days_left "$SSL_CERT"; true); NGINX_DAYS=${NGINX_DAYS:-999}
ACME_DAYS=$(days_left "$ACME_CERT"; true);  ACME_DAYS=${ACME_DAYS:-999}
echo "   nginx 正在用: $SSL_CERT   剩余 ${NGINX_DAYS} 天"
echo "   acme  已签发: $ACME_CERT   剩余 ${ACME_DAYS} 天"
if [ "$NGINX_DAYS" = "999" ]; then bad "读不到 nginx 证书 ($SSL_CERT)"
elif [ "$NGINX_DAYS" -lt 0 ]; then bad "nginx 证书已过期 $(( -NGINX_DAYS )) 天 -> 钉钉回调/浏览器全部不可用"
elif [ "$NGINX_DAYS" -lt 3 ]; then warn "nginx 证书 ${NGINX_DAYS} 天后到期"
else ok "nginx 证书有效 (剩余 ${NGINX_DAYS} 天)"; fi
[ -x "$ACME_SH" ] && ok "acme.sh 就位: $("$ACME_SH" --version 2>/dev/null | tail -1)" || bad "acme.sh 缺失: $ACME_SH"

# ── 2. 按需签发/安装证书 ───────────────────────────────────────────────────
hdr "2/6 签发/安装证书"
ACME_EPOCH=$(cert_end_epoch "$ACME_CERT"); NGINX_EPOCH=$(cert_end_epoch "$SSL_CERT")
if [ "$ACME_EPOCH" -le "$(date +%s)" ]; then
  ACTION="renew";   echo "   acme 目录无可用证书(缺失或已过期) -> 需要先签发"
elif [ "$ACME_EPOCH" -gt "$NGINX_EPOCH" ]; then
  ACTION="install"; echo "   acme 目录的证书比 nginx 用的新 -> 需要安装"
else
  ACTION="none";    ok "nginx 已是最新证书, 无需签发/安装"
fi
[ "$MODE" = "renew" ] && ACTION="renew"
case "$ACTION" in
  none) : ;;
  renew)
    if [ "$MODE" = "check" ]; then warn "(--check 只读, 未签发; 去掉 --check 即自动完成)"
    else
      echo "   签发中 (acme.sh --renew --force)..."
      if "$ACME_SH" --renew -d "$DOMAIN" --force --home "$ACME_HOME" --log "$RENEW_LOG"; then
        own_back "$RENEW_LOG"; ok "签发+安装完成"
      else bad "签发失败, 细节见 $RENEW_LOG"; fi
    fi ;;
  install)
    if [ "$MODE" = "check" ]; then warn "(--check 只读, 未安装; 去掉 --check 即自动完成)"
    else
      if "$ACME_SH" --install-cert -d "$DOMAIN" \
           --key-file "$SSL_KEY" --fullchain-file "$SSL_CERT" \
           --reloadcmd "systemctl reload nginx" --home "$ACME_HOME" --log "$RENEW_LOG"; then
        own_back "$RENEW_LOG"; ok "证书已安装并 reload nginx"
      else bad "安装失败, 细节见 $RENEW_LOG"; fi
    fi ;;
esac

# ── 3. 回调入口加固: 80 / 8080 / 443 全接 + 访问日志 ───────────────────────
hdr "3/6 回调入口加固 (80 / 8080 / 443)"
if grep -q "nginx_dingtalk" "$NGX_CONF" 2>/dev/null; then
  ok "入口已加固; 访问日志 $DING_LOG"
elif [ "$MODE" = "check" ]; then
  warn "入口未加固 (--check 只读; 去掉 --check 会自动加固: 80 不再 301、8080 补入口、落访问日志)"
elif [ "$AS_ROOT" != "1" ]; then
  bad "需要 root 才能改 nginx 配置"
else
  PATCH_RC=0
  python3 - "$NGX_CONF" <<'PY' || PATCH_RC=$?
import shutil, subprocess, sys, time
conf = sys.argv[1]
bak = conf + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
shutil.copy2(conf, bak)
s = open(conf, encoding="utf-8").read()
LOG = "access_log syslog:server=unix:/dev/log,facility=local7,tag=nginx_dingtalk,severity=info;\n"
if "nginx_dingtalk" not in s:
    # 清掉旧版写入的 /tmp 日志行 (nginx PrivateTmp=yes 时写进私有命名空间, 看不见)
    s = s.replace("        access_log /tmp/dingtalk_access.log;\n", "")

    # (a) 443 上的 /api/dingtalk/ 落访问日志 (仅当该 location 还没有 access_log)
    if "    location /api/dingtalk/ {\n        proxy_pass http://127.0.0.1:8081;" in s:
        s = s.replace(
            "    location /api/dingtalk/ {\n        proxy_pass http://127.0.0.1:8081;",
            "    location /api/dingtalk/ {\n        " + LOG +
            "        proxy_pass http://127.0.0.1:8081;", 1)
    # (b) 80 端口放行 /api/dingtalk/ (原来 301 跳转, 钉钉 POST 会失败)
    #     幂等: 只有在 80 段还没有该 location 时才插入 (否则重复 location 会让 nginx -t 失败)
    if s.count("location /api/dingtalk/") < 2:
        s = s.replace(
            "    location / {\n        return 301 https://$host$request_uri;\n    }",
            "    location /api/dingtalk/ {\n        " + LOG +
            "        proxy_pass http://127.0.0.1:8081;\n        proxy_set_header Host $host;\n    }\n"
            "    location / {\n        return 301 https://$host$request_uri;\n    }", 1)
    # (c) 8080 专用回调入口 —— 钉钉出网实际能通的组合 (HTTP + 8080), 必须保留
    if "listen 8080;" not in s:
        s += ("\nserver {\n    listen 8080;\n    server_name _;\n"
              "    " + LOG +
              "    location /api/dingtalk/ { proxy_pass http://127.0.0.1:8081; proxy_set_header Host $host; }\n"
              "    location /static/charts/ { proxy_pass http://127.0.0.1:8081; }\n"
              "    location / { return 444; }\n}\n")
    # (e) nginx 错误日志也送 journald -> 以后 TLS 握手失败/上游报错都能查
    if "tag=nginx_err" not in s and "    listen 443 ssl;\n" in s:
        s = s.replace(
            "    listen 443 ssl;\n",
            "    error_log syslog:server=unix:/dev/log,facility=local7,tag=nginx_err warn;\n"
            "    listen 443 ssl;\n", 1)
    # (d) 8080 是钉钉实际在用的入口 -> 保证它有访问日志 (仅当该 server 块里还没有时插入)
    if "listen 8080;" in s:
        i = s.index("listen 8080;")
        j = s.find("server {", i)
        blk = s[i:j if j > 0 else len(s)]
        if "nginx_dingtalk" not in blk and "server_name _;" in blk:
            blk2 = blk.replace("server_name _;", "server_name _;\n    " + LOG.rstrip("\n"), 1)
            s = s[:i] + blk2 + (s[j:] if j > 0 else "")
    open(conf, "w", encoding="utf-8").write(s)
    # 兜底自检: 8080 入口与 80/443 入口都必须存在
    s2 = open(conf, encoding="utf-8").read()
    problems = []
    if "listen 8080;" not in s2:
        problems.append("缺少 8080 回调入口(钉钉就靠它)")
    if s2.count("location /api/dingtalk/") < 2:
        problems.append("80/443 回调入口不完整")
    if problems:
        shutil.copy2(bak, conf)
        print("   ❌ 加固结果不完整(%s), 已回滚。备份: %s" % ("; ".join(problems), bak))
        sys.exit(1)
t = subprocess.run(["nginx", "-t"], capture_output=True, text=True)
if t.returncode != 0:
    shutil.copy2(bak, conf)
    print("   ❌ nginx -t 失败, 已回滚。原因: %s" % (t.stderr or t.stdout).strip()[:300])
    sys.exit(1)
print("   (配置已改, 备份: %s)" % bak)
PY
  if [ "$PATCH_RC" != "0" ]; then bad "加固失败, 已回滚原配置"
  else
      if systemctl reload nginx; then
      ok "已加固: 80 / 8080 / 443 都能收回调; 回调访问日志: journalctl -t nginx_dingtalk"
    else bad "nginx reload 失败"; fi
  fi
fi

# ── 4. 续期自动化 (root cron, 幂等) ────────────────────────────────────────
hdr "4/6 自动续期"
if [ "$AS_ROOT" = "1" ]; then
  if crontab -l 2>/dev/null | grep -qF "acme.sh --cron"; then ok "root cron 已有续期任务"
  elif [ "$MODE" = "check" ]; then warn "root cron 里没有续期任务 (--check 未安装)"
  else
    { crontab -l 2>/dev/null; echo "$CRON_LINE"; } | crontab - \
      && ok "已写入 root cron: 每天 03:17 自动续期 (日志 $RENEW_LOG)" || bad "写入 root cron 失败"
  fi
elif sudo -n crontab -l 2>/dev/null | grep -qF "acme.sh --cron"; then
  ok "root cron 已有续期任务"
else
  warn "root cron 状态未知(需 root 才能读); 去掉 --check 跑一次即会自动装好"
fi
if [ -f "$RENEW_LOG" ] && tail -40 "$RENEW_LOG" 2>/dev/null | grep -qi "permission denied\|error renewing"; then
  bad "续期日志里有失败记录: $(tail -2 "$RENEW_LOG" | tr '\n' ' ')"
fi

# ── 5. 端到端链路自检 ──────────────────────────────────────────────────────
hdr "5/6 链路自检"
PROBE=$(echo | timeout 20 openssl s_client -connect "$DOMAIN:443" -servername "$DOMAIN" 2>/dev/null)
SERVED_END=$(echo "$PROBE" | openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2)
if echo "$PROBE" | grep -q "Verify return code: 0 (ok)"; then
  ok "TLS 握手 + 证书链校验通过 (服务端证书到期: ${SERVED_END:-未知})"
else
  bad "TLS 校验未通过(证书过期/链不受信任/443 不通) -> 钉钉回调必然连不上"
fi
ERRLOG=$(mktemp); CODE=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 "https://$DOMAIN/" 2>"$ERRLOG"); RC=$?
if [ "$RC" = "0" ]; then ok "HTTPS 可达 (HTTP $CODE; 403=门禁正常, 非故障)"
else bad "HTTPS 不可达: $(head -c 200 "$ERRLOG")"; fi
rm -f "$ERRLOG"
echo "   回调入口探针 (钉钉可能填的三种地址):"
for u in "https://$DOMAIN$CALLBACK_PATH" "http://$DOMAIN$CALLBACK_PATH" "http://$DOMAIN:8080$CALLBACK_PATH"; do
  c=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 -X POST "$u" \
      -H 'Content-Type: application/json' -d '{}' 2>/dev/null)
  if [ "$c" = "200" ]; then echo "     ✅ $u -> HTTP $c"; else echo "     ❌ $u -> HTTP ${c:-无响应}"; fi
done
echo "   钉钉后台'消息接收地址/出网地址'应为: https://$DOMAIN$CALLBACK_PATH"
for s in nginx $SERVICES; do
  if systemctl is-active --quiet "$s" 2>/dev/null; then ok "$s 运行中"
  elif [ "$MODE" != "check" ] && [ "$AS_ROOT" = "1" ]; then
    systemctl start "$s" && ok "$s 已启动" || bad "$s 启动失败"
  else warn "$s 未运行"; fi
done
if [ "$DO_RESTART" = "1" ] && [ "$AS_ROOT" = "1" ]; then
  systemctl restart $SERVICES && ok "已按 --restart 重启: $SERVICES" || bad "重启失败"
fi

# ── 6. 结论 ────────────────────────────────────────────────────────────────
hdr "6/6 结论"
if [ "$FAIL" = "0" ]; then
  cat <<EOF
   ✅ 全部通过。诊断"钉钉没反应"的三板斧:
      1) 在群里 @机器人 发一句
      2) 看回调日志: journalctl -t nginx_dingtalk -n 20   <- 钉钉有没有来、什么路径、回了几号
      3) 看服务端:   journalctl -u stockbot-web -n 30 | grep -E "收到消息|DingTalk"
   ✅ 续期已交给 root cron (每天 03:17), 以后无需手工执行任何东西。
EOF
elif [ "$MODE" = "check" ]; then
  echo "   ❌ 有未通过项。去掉 --check 再跑一次本脚本即可全部自动修复(不需要你手写任何命令)。"
else
  echo "   ❌ 有未通过项, 上面已打印失败原因; 重跑本脚本即可重试(幂等, root 身份跑亦可)。"
fi
echo
exit "$FAIL"
