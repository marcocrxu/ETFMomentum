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
                 state_file: str = "/home/ecs-user/etf_strategy/state.json"):
        self.cfg = cfg
        self.feed = feed
        self.name_to_code = name_to_code
        self.state_file = state_file
        self.tencent_map = {
            "创业": "s_sz159915", "纳指": "s_sh513100",
            "黄金": "s_sh518880", "上证": "s_sh000001",
            "创业指": "s_sz399006",
        }

    # 与 bt_runner 完全同口径的历史窗口: 决策区起点 + 320 日历天预热
    # (2026-09-11 实测: 此前实盘从决策区起点开始取数, 上证/创业指数历史比回测短,
    #  MA200/恐慌阈值/加速止盈阈值的 expanding 窗口不同 → 历史操作日期差 1 天)
    DECISION_START = "2016-07-07"
    WARMUP_DAYS = 320   # = bt_runner._warmup 下限 (保证 MA200 在决策起点就绪)

    def check(self, require_trade_day=True) -> dict:
        if require_trade_day and not self._is_trade_day():
            return {"signal": "非交易日"}

        today = datetime.now().strftime("%Y-%m-%d")
        end = today.replace("-", "")
        fetch_start = (pd.Timestamp(self.DECISION_START)
                       - pd.Timedelta(days=self.WARMUP_DAYS)).strftime("%Y%m%d")

        etf_names = list(self.name_to_code.keys())
        codes = list(self.name_to_code.values()) + ["000001.SH", "399006.SZ"]

        # 加载历史数据（对齐日期）
        dfs = self.feed.get_dataframe(codes, fetch_start, end)
        if len(dfs) < 3:
            return {"error": "数据加载失败"}

        # 截断到 [fetch_start, today] (缓存可能返回更早的数据, 与 bt_runner 同处理)
        _fs_ts = pd.Timestamp(fetch_start)
        today_ts = pd.Timestamp(today)
        for code in list(dfs.keys()):
            df = dfs[code]
            df = df[(df["trade_date"] >= _fs_ts) & (df["trade_date"] <= today_ts)]
            if len(df) == 0:
                del dfs[code]
            else:
                dfs[code] = df.reset_index(drop=True)

        # 全量时间轴 (预热段 + 决策区)
        all_full = sorted(set().union(*[set(df["trade_date"]) for df in dfs.values()]))

        # 盘中运行时缓存最后一行是昨天: 为今天追加新行
        # (此前用实时价直接覆盖最后一行, 导致昨收从序列中消失、动量窗口错位)
        if not all_full or all_full[-1] < today_ts:
            all_full.append(today_ts)
        n_full = len(all_full)
        pre_days = next((i for i, d in enumerate(all_full)
                         if d >= pd.Timestamp(self.DECISION_START)), 0)
        all_dates = all_full[pre_days:]      # 决策区
        n = len(all_dates)

        # 构建价格数组 (全量)
        prices = {}
        for name, code in self.name_to_code.items():
            if code not in dfs: continue
            s = dfs[code].set_index("trade_date")["close"]
            prices[name] = np.array([s.get(d, np.nan) for d in all_full])
        sh_s = dfs["000001.SH"].set_index("trade_date")["close"]
        sh_price = np.array([sh_s.get(d, np.nan) for d in all_full])

        cyb_p = np.full(n_full, np.nan)
        if "399006.SZ" in dfs:
            s = dfs["399006.SZ"].set_index("trade_date")["close"]
            cyb_p = np.array([s.get(d, np.nan) for d in all_full])
        sh_p = sh_price.copy()

        price_mat = np.array([prices.get(nm, np.full(n_full, np.nan)) for nm in etf_names])

        # 实时价格写入今日行
        live = self._get_live_prices()
        for i, name in enumerate(etf_names):
            if name in live and live[name] > 0:
                price_mat[i, -1] = live[name]
        if "上证" in live and live["上证"] > 0:
            sh_price[-1] = live["上证"]
            sh_p[-1] = live["上证"]
        if "创业指" in live and live["创业指"] > 0:
            cyb_p[-1] = live["创业指"]

        # 动量 (全量历史上计算, 切片进决策区 —— 与 bt_runner 同口径)
        def momentum(arr, w, s=0):
            m = np.full(n_full, np.nan)
            if s <= 0:
                m[w:] = (arr[w:] / arr[:-w] - 1) * 100
            else:
                m[w+s:] = (arr[w:n_full-s] / arr[:n_full-w-s] - 1) * 100
            return m

        _skip = self.cfg.mom_skip
        mom20_mat = np.array([momentum(price_mat[i], 20, _skip) for i in range(len(etf_names))])
        mom10_mat = np.array([momentum(price_mat[i], 10) for i in range(len(etf_names))])
        sh_mom = momentum(sh_price, 20)

        # ── 纳指动量混入 QQQ-CNY (默认开; w=QQQ权重, 0.7=70%QQQ+30%ETF; 与回测同口径) ──
        nz_mix_active = False
        nz_note = ""
        if getattr(self.cfg, "nz_mom_qqq_on", True) and "纳指" in etf_names:
            try:
                from data.usfeed import load_qqq_cny, align_to_cn, qqq_cny_freshness
                # 陈旧数据必须回退: ffilled 的陈旧 QQQ 会污染纳指动量
                # (2026-09-09 空仓信号丢失事故: CSV 断更后静默降级, 无人察觉)
                _last, _expect, _fresh = qqq_cny_freshness()
                if not _fresh:
                    raise RuntimeError("QQQ-CNY 数据陈旧: 最新%s, 期望≥%s"
                                       % (_last.date(), _expect.date()))
                _w = min(max(float(getattr(self.cfg, "nz_mom_qqq_w", 0.7)), 0.0), 1.0)
                _qq = align_to_cn(all_full, load_qqq_cny())
                _nz = etf_names.index("纳指")
                _qm20 = momentum(_qq, 20, _skip)
                _qm10 = momentum(_qq, 10)
                mom20_mat[_nz] = (1 - _w) * mom20_mat[_nz] + _w * _qm20
                mom10_mat[_nz] = (1 - _w) * mom10_mat[_nz] + _w * _qm10
                nz_mix_active = True
                nz_note = "%.1f×QQQ-CNY(%+.2f%%) + %.1f×513100" % (
                    _w, _qm20[-1] if not np.isnan(_qm20[-1]) else 0.0, 1 - _w)
            except Exception as e:
                nz_note = "回退纯ETF: %s" % e
                print("QQQ混合失败(回退ETF动量): %s" % e)

        # 调用共享信号函数 — 切片进决策区, cyb/sh 传全量历史 + pre_days,
        # accel 阈值全量预计算后切片 (与 bt_runner 完全同口径, 杜绝冷启动差异)
        from strategy.signal import calc_accel_thresholds, calc_vol20
        pre_mats = {"accel_th": calc_accel_thresholds(mom10_mat, self.cfg)[:, pre_days:]}
        if self.cfg.stop_sigma > 0 or self.cfg.risk_adj_mom or self.cfg.vol_scale_target > 0:
            pre_mats["vol"] = calc_vol20(price_mat)[:, pre_days:]
        actions, _, _ = compute_signals(
            self.cfg, etf_names,
            price_mat[:, pre_days:], mom20_mat[:, pre_days:], mom10_mat[:, pre_days:],
            sh_mom[pre_days:], cyb_p, sh_p, n,
            pre_days=pre_days, pre_mats=pre_mats
        )

        # 只有发生在今日(最后一行)的 action 才是今天的信号
        # (此前直接取 actions[-1], 会把几天前的历史操作当成今日信号推送)
        last_action = actions[-1] if actions else None
        signal = "HOLD"
        target = None
        reason = ""
        if last_action and last_action[0] == n - 1:
            _, old_p, new_p, reason = last_action
            if new_p >= 0: signal = "BUY"; target = etf_names[new_p]
            elif old_p >= 0: signal = "CASH"

        # ── 策略模拟仓位: 全量回放的权威结果 (与实际持仓 state 分开,
        #    两者不一致 = 用户没跟上信号, daily_run 要在推送里告警) ──
        sim_pos = -1; sim_entry_idx = -1; sim_entry_price = 0.0
        prev_pos = -1           # 今日信号执行前的仓位 (操作建议"卖出X"用)
        for (t, _o, new_p, _r) in actions:
            if t == n - 1:
                prev_pos = sim_pos
            sim_pos = new_p
            if new_p >= 0:
                sim_entry_idx = t
                _ep = price_mat[new_p, pre_days + t]   # t 是决策区索引, price_mat 是全量
                sim_entry_price = float(_ep) if not np.isnan(_ep) else 0.0
        if not actions or actions[-1][0] != n - 1:
            prev_pos = sim_pos
        sim_position = etf_names[sim_pos] if sim_pos >= 0 else ""
        prev_position = etf_names[prev_pos] if prev_pos >= 0 else ""
        sim_entry_date = str(all_dates[sim_entry_idx])[:10] if sim_entry_idx >= 0 else ""
        recent_actions = [
            {"date": str(all_dates[t])[:10],
             "old": etf_names[o] if o >= 0 else "空仓",
             "new": etf_names[nw] if nw >= 0 else "空仓",
             "reason": rr}
            for (t, o, nw, rr) in actions[-5:]
        ]

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
            "sh_momentum": round(float(sh_mom[-1]), 2) if not np.isnan(sh_mom[-1]) else 0,
            "momentums": {
                name: round(mom20_mat[i, -1], 2) if not np.isnan(mom20_mat[i, -1]) else 0
                for i, name in enumerate(etf_names)
            },
            # ── 以下为对账/告警新增字段 ──
            "sim_position": sim_position,          # 策略全量回放的当前应持仓位 (""=空仓)
            "prev_position": prev_position,        # 今日信号执行前的策略仓位
            "sim_entry_date": sim_entry_date,
            "sim_entry_price": sim_entry_price,
            "recent_actions": recent_actions,       # 最近5条回放操作 (对账用)
            "nz_mix_active": nz_mix_active,         # 纳指 QQQ 混合是否生效
            "nz_note": nz_note,
            "live_prices": {k: v for k, v in live.items()},
        }

    def execute_and_notify(self) -> str:
        """[遗留] cron 模式: 执行操作 + 推送钉钉/微信。
        注意: 生产入口已迁移到 live/daily_run.py (完整消息落盘+对账),
        本方法仅供旧调用方兼容, 新代码请用 daily_run.main()。"""
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
        lines.append(f"    📉 上证指数: {result.get('sh_momentum', 0):+.2f}%")

        if result["position"]:
            lines.append(f"\n💼 持仓: {result['position']} | 盈亏{result['pnl']:+.1f}% | 第{result['hold_days']}天")
        lines.extend(["", f"📋 **{result['reason']}**"])

        msg = "\n".join(lines)

        # 钉钉
        try:
            import sys; sys.path.insert(0, "/home/ecs-user/stock-bot")
            from services.dingtalk_service import send_text
            send_text(msg)
            print("钉钉: OK")
        except Exception as e:
            print("钉钉: %s" % e)

        # 微信
        try:
            wu_file = "/home/ecs-user/stock-bot/wechat_user.json"
            bs_file = "/home/ecs-user/stock-bot/bot_session.json"
            if os.path.exists(wu_file) and os.path.exists(bs_file):
                with open(wu_file) as f: wu = json.load(f)
                with open(bs_file) as f: bs = json.load(f)
                to_user = wu.get("last_user_id", "")
                ctx = wu.get("context_token", "")
                bot_token = bs.get("bot_token", "")
                baseurl = bs.get("baseurl", "https://ilinkai.weixin.qq.com")
                if not ctx:
                    print("微信: 无 context_token (用户需先给 bot 发条消息)")
                elif to_user and bot_token:
                    import secrets, time as _t, struct, base64 as _64
                    uin = _64.b64encode(struct.pack(">I", __import__("random").randint(1, 2**31))).decode()
                    cid = f"stock-bot:{int(_t.time()*1000)}-{secrets.token_hex(4)}"
                    resp = requests.post(f"{baseurl}/ilink/bot/sendmessage",
                        headers={"Content-Type":"application/json","AuthorizationType":"ilink_bot_token",
                                 "X-WECHAT-UIN":uin,"Authorization":f"Bearer {bot_token}"},
                        json={"msg":{"from_user_id":"","to_user_id":to_user,"client_id":cid,
                                     "message_type":2,"message_state":2,"context_token":ctx,
                                     "item_list":[{"type":1,"text_item":{"text":msg}}]},
                               "base_info":{"channel_version":"1.0.2"}}, timeout=15)
                    ret = resp.json().get("ret", -999)
                    if ret == 0:
                        print("微信: 发送成功")
                    else:
                        print("微信: 发送失败 ret=%s %s" % (ret, resp.text[:150]))
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
                    "黄金ETF华安": "黄金", "上证指数": "上证", "创业板指": "创业指"}
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
