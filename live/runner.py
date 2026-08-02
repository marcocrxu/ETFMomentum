"""实盘运行 — 每天14:50调共享信号函数"""
import json, os, requests
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from typing import Dict

from strategy.signal import SignalConfig, compute_signals
from data.feed import TushareFeed


class LiveRunner:
    def __init__(self, cfg: SignalConfig, feed: TushareFeed,
                 name_to_code: Dict[str, str],
                 state_file: str = "/root/etf_strategy/state.json"):
        self.cfg = cfg
        self.feed = feed
        self.name_to_code = name_to_code
        self.state_file = state_file
        self.tencent_map = {
            "创业": "s_sz159915", "纳指": "s_sh513100",
            "黄金": "s_sh518880", "上证": "s_sh000001",
        }

    def check(self) -> dict:
        if not self._is_trade_day():
            return {"signal": "非交易日"}

        today = datetime.now().strftime("%Y-%m-%d")
        end = today.replace("-", "")
        start_all = "20160707"

        etf_names = list(self.name_to_code.keys())
        codes = list(self.name_to_code.values()) + ["000001.SH", "399006.SZ"]

        # 加载历史数据（对齐日期）
        dfs = self.feed.get_dataframe(codes, start_all, end)
        if len(dfs) < 3:
            return {"error": "数据加载失败"}

        all_dates = sorted(set().union(*[set(df["trade_date"]) for df in dfs.values()]))
        n = len(all_dates)

        # 构建价格数组
        prices = {}
        for name, code in self.name_to_code.items():
            if code not in dfs: continue
            s = dfs[code].set_index("trade_date")["close"]
            prices[name] = np.array([s.get(d, np.nan) for d in all_dates])
        sh_s = dfs["000001.SH"].set_index("trade_date")["close"]
        sh_price = np.array([sh_s.get(d, np.nan) for d in all_dates])

        cyb_p = sh_p = np.full(n, np.nan)
        if "399006.SZ" in dfs:
            s = dfs["399006.SZ"].set_index("trade_date")["close"]
            cyb_p = np.array([s.get(d, np.nan) for d in all_dates])
        if "000001.SH" in dfs:
            sh_p = np.array([sh_s.get(d, np.nan) for d in all_dates])

        price_mat = np.array([prices.get(n, np.full(n, np.nan)) for n in etf_names])

        # 实时价格（补到数组最后）
        live = self._get_live_prices()
        for i, name in enumerate(etf_names):
            if name in live and live[name] > 0:
                price_mat[i, -1] = live[name]
        if "上证" in live and live["上证"] > 0:
            sh_price[-1] = live["上证"]

        # 动量
        def momentum(arr, w):
            m = np.full(n, np.nan)
            m[w:] = (arr[w:] / arr[:-w] - 1) * 100
            return m

        mom20_mat = np.array([momentum(price_mat[i], 20) for i in range(len(etf_names))])
        mom10_mat = np.array([momentum(price_mat[i], 10) for i in range(len(etf_names))])
        sh_mom = momentum(sh_price, 20)

        # 用实时价补最后一刻动量
        for i in range(len(etf_names)):
            if n >= 21 and price_mat[i, -1] > 0 and price_mat[i, -21] > 0:
                mom20_mat[i, -1] = (price_mat[i, -1] / price_mat[i, -21] - 1) * 100
            if n >= 11 and price_mat[i, -1] > 0 and price_mat[i, -11] > 0:
                mom10_mat[i, -1] = (price_mat[i, -1] / price_mat[i, -11] - 1) * 100
        if n >= 21 and sh_price[-1] > 0 and sh_price[-21] > 0:
            sh_mom[-1] = (sh_price[-1] / sh_price[-21] - 1) * 100

        # 调用共享信号函数
        actions, _ = compute_signals(
            self.cfg, etf_names, price_mat, mom20_mat, mom10_mat,
            sh_mom, cyb_p, sh_p, n
        )

        # 取最后一个操作
        last_action = actions[-1] if actions else None
        signal = "HOLD"
        target = None
        reason = ""
        if last_action:
            _, old_p, new_p, reason = last_action
            if new_p >= 0: signal = "BUY"; target = etf_names[new_p]
            elif old_p >= 0 and new_p == -1: signal = "CASH"

        # 持仓盈亏
        pnl = 0; hold_days = 0; position = ""
        try:
            state_file = self.state_file
            if os.path.exists(state_file):
                with open(state_file) as f:
                    st = json.load(f)
                position = st.get("position", "")
                entry_price = st.get("entry_price", 0)
                if position and entry_price > 0 and position in live:
                    pnl = (live[position] / entry_price - 1) * 100
                if st.get("entry_date"):
                    hold_days = (datetime.now().date() - datetime.strptime(st["entry_date"][:10], "%Y-%m-%d").date()).days
        except: pass

        return {
            "date": today,
            "signal": signal, "target": target, "reason": reason,
            "position": position, "pnl": round(pnl, 2), "hold_days": hold_days,
            "momentums": {
                name: round(mom20_mat[i, -1], 2) if not np.isnan(mom20_mat[i, -1]) else 0
                for i, name in enumerate(etf_names)
            },
        }

    def execute_and_notify(self) -> str:
        """cron 模式: 执行操作 + 推送钉钉/微信"""
        result = self.check()
        if "error" in result:
            return result["error"]

        # 构建消息
        lines = ["📊 ETF 20日动量轮动", f"⏰ {result['date']}", ""]
        ranked = sorted(result["momentums"].items(), key=lambda x: -x[1])
        for i, (name, mom) in enumerate(ranked):
            arrow = "🟢" if mom >= 0 else "🔴"
            icon = {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, "")
            holder = " ← 持有" if name == result["position"] else ""
            lines.append(f"{icon} {arrow} **{name}**: {mom:+.2f}%{holder}")
        lines.append(f"    📉 上证指数: {result['momentums'].get('上证指数', 0):+.2f}%")

        if result["position"]:
            lines.append(f"\n💼 持仓: {result['position']} | 盈亏{result['pnl']:+.1f}% | 第{result['hold_days']}天")
        lines.extend(["", f"📋 **{result['reason']}**"])

        msg = "\n".join(lines)

        # 钉钉
        try:
            import sys; sys.path.insert(0, "/root/stock-bot")
            from services.dingtalk_service import send_text
            send_text(msg)
            print("钉钉: OK")
        except Exception as e:
            print("钉钉: %s" % e)

        # 微信
        try:
            wu_file = "/root/stock-bot/wechat_user.json"
            bs_file = "/root/stock-bot/bot_session.json"
            if os.path.exists(wu_file) and os.path.exists(bs_file):
                with open(wu_file) as f: wu = json.load(f)
                with open(bs_file) as f: bs = json.load(f)
                to_user = wu.get("last_user_id", "")
                bot_token = bs.get("bot_token", "")
                baseurl = bs.get("baseurl", "https://ilinkai.weixin.qq.com")
                if to_user and bot_token:
                    import secrets, time as _t, struct, base64 as _64
                    uin = _64.b64encode(struct.pack(">I", __import__("random").randint(1, 2**31))).decode()
                    cid = f"stock-bot:{int(_t.time()*1000)}-{secrets.token_hex(4)}"
                    resp = requests.post(f"{baseurl}/ilink/bot/sendmessage",
                        headers={"Content-Type":"application/json","AuthorizationType":"ilink_bot_token",
                                 "X-WECHAT-UIN":uin,"Authorization":f"Bearer {bot_token}"},
                        json={"msg":{"from_user_id":"","to_user_id":to_user,"client_id":cid,
                                     "message_type":2,"message_state":2,"context_token":"",
                                     "item_list":[{"type":1,"text_item":{"text":msg}}]},
                               "base_info":{"channel_version":"1.0.2"}}, timeout=15)
                    print("微信: HTTP %d" % resp.status_code)
            else:
                print("微信: 未找到用户/会话文件")
        except Exception as e:
            print("微信: %s" % e)

        return msg

    def _get_live_prices(self) -> Dict[str, float]:
        url = "http://qt.gtimg.cn/q=" + ",".join(self.tencent_map.values())
        try:
            resp = requests.get(url, timeout=5); resp.encoding = "gbk"
        except:
            return {}
        tc_to = {v: k for k, v in self.tencent_map.items()}
        tc_names = {"创业板ETF易方达": "创业", "纳指ETF": "纳指", "纳指ETF国泰": "纳指",
                    "黄金ETF华安": "黄金", "上证指数": "上证"}
        result = {}
        for line in resp.text.strip().split("\n"):
            parts = line.split("~")
            if len(parts) < 10: continue
            tag = tc_names.get(parts[1], "")
            if tag:
                try: result[tag] = float(parts[3])
                except: pass
        return result

    def _is_trade_day(self) -> bool:
        today = datetime.now().strftime("%Y%m%d")
        try:
            cal = self.feed.pro.trade_cal(exchange="SSE", start_date=today, end_date=today)
            if cal is not None and len(cal) > 0:
                return cal.iloc[0]["is_open"] == 1
        except: pass
        return datetime.now().weekday() < 5
