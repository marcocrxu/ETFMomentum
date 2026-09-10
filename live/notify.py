"""推送与归档 — 实盘链路唯一出口

钉钉: 复用 stock-bot/services/dingtalk_service.send_text
微信: 复用 stock-bot 的 ilink 会话 (需用户最近与 bot 互动过)
归档: 每条消息的完整正文追加写入 live/notify_log.jsonl
      (2026-09-10 事故教训: 旧 cron 只落盘决策一句话, 事后无法还原用户实际收到的内容)
"""
import json
import os
import sys
from datetime import datetime

STOCK_BOT = "/home/ecs-user/stock-bot"
if STOCK_BOT not in sys.path:
    sys.path.insert(0, STOCK_BOT)

NOTIFY_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notify_log.jsonl")


def send_dingtalk(text):
    """返回 (ok, err)"""
    try:
        from services.dingtalk_service import send_text
        send_text(text)
        return True, ""
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)


def send_wechat(text):
    """返回 (ok, err)。需要 wechat_user.json 中有活跃的 context_token。"""
    try:
        import requests
        wu_file = os.path.join(STOCK_BOT, "wechat_user.json")
        bs_file = os.path.join(STOCK_BOT, "bot_session.json")
        if not (os.path.exists(wu_file) and os.path.exists(bs_file)):
            return False, "无 wechat_user/bot_session 文件"
        with open(wu_file) as f:
            wu = json.load(f)
        with open(bs_file) as f:
            bs = json.load(f)
        to_user = wu.get("last_user_id", "")
        ctx = wu.get("context_token", "")
        bot_token = bs.get("bot_token", "")
        baseurl = bs.get("baseurl", "https://ilinkai.weixin.qq.com")
        if not ctx:
            return False, "无 context_token (用户需先给 bot 发条消息)"
        if not (to_user and bot_token):
            return False, "无活跃用户或 token"
        import secrets
        import struct
        import base64 as _64
        import time as _t
        uin = _64.b64encode(struct.pack(">I", secrets.randbelow(2**31 - 1) + 1)).decode()
        cid = f"stock-bot:{int(_t.time() * 1000)}-{secrets.token_hex(4)}"
        resp = requests.post(
            f"{baseurl}/ilink/bot/sendmessage",
            headers={"Content-Type": "application/json",
                     "AuthorizationType": "ilink_bot_token",
                     "X-WECHAT-UIN": uin,
                     "Authorization": f"Bearer {bot_token}"},
            json={"msg": {"from_user_id": "", "to_user_id": to_user,
                          "client_id": cid, "message_type": 2, "message_state": 2,
                          "context_token": ctx,
                          "item_list": [{"type": 1, "text_item": {"text": text}}]},
                  "base_info": {"channel_version": "1.0.2"}},
            timeout=15)
        ret = resp.json().get("ret", -999)
        if ret == 0:
            return True, ""
        return False, "ret=%s %s" % (ret, resp.text[:150])
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)


def push_all(text):
    """钉钉 + 微信都推, 返回 {"dingtalk": (ok, err), "wechat": (ok, err)}"""
    return {"dingtalk": send_dingtalk(text), "wechat": send_wechat(text)}


def archive(kind, message, extra=None):
    """完整正文落盘 (jsonl), kind: daily / reconcile / alert"""
    rec = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "kind": kind, "message": message}
    if extra:
        rec["extra"] = extra
    with open(NOTIFY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
