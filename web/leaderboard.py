"""打榜 — 参数网格批量回测排行 + 自然语言自定义策略提交 (dsh headless 生成单文件代码并回测)

- 网格 (研究固化参数轴): 主引擎 16 组合 + yanyan 54 组合
- 自定义: 输入自然语言 → 后台 claude code + kimi k3 生成单一文件 lb_custom/<id>.py
  (契约: run(params)->dict) → 子进程执行回测 → 结果并入榜单
- 安全: 提交须 LEADERBOARD_TOKEN (web/config_local.py), 防外部刷 dsh/tushare token;
  队列上限 3, 串行执行, 生成超时 900s
- 窗口固定: 2016-01-01 ~ 2026-07-31
"""
import json
import os
import re
import subprocess
import threading
import time
import uuid

import numpy as np
import pandas as pd

from strategy.yanyan import YanyanConfig, UNIVERSE, compute_yanyan
from web.bt_runner import run_backtest, FEED

LAST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard_last.json")
QUEUE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lb_queue.json")
CUSTOM_DIR = "/root/etf_strategy/lb_custom"
REPO_DIR = "/root/etf_strategy"

# 固定打榜窗口 (用户要求)
LB_START = "2016-01-01"
LB_END = "2026-07-31"

METRICS = ("calmar", "total_ret", "ann_ret", "sharpe", "mdd")

# ── 主引擎打榜网格 ──
ETF_GRID = []
for gap in (2.5, 3.0, 3.5, 4.0):
    for cd in (0, 3):
        for qqq_on, qqq_w in ((False, 0.0), (True, 0.7)):
            p = {"universe": "纳指,黄金,创业",
                 "exception_gap": gap, "sh_cash_cooldown": cd}
            label = "gap=%.1f 冷却=%d QQQ=%s" % (gap, cd, "0.7" if qqq_on else "关")
            p["nz_mom_qqq_on"] = qqq_on
            if qqq_on:
                p["nz_mom_qqq_w"] = qqq_w
            ETF_GRID.append((label, p))

# ── yanyan 打榜网格 ──
YANYAN_GRID = []
for buf in (1.0, 2.0, 3.0):
    for oh_mom in (30.0, 40.0, 50.0):
        for oh_q in (0.0, 3.0):
            for rma in (150, 200, 250):
                YANYAN_GRID.append(
                    ("buf=%.0f 过热=%.0f Q=%.0f MA=%d" % (buf, oh_mom, oh_q, rma),
                     {"switch_buffer": buf, "overheat_mom": oh_mom,
                      "overheat_q": oh_q, "regime_ma": rma}))

YANYAN_UNIVERSES = {
    "全池10只": "",
    "精简TOP6": "沪深300,中证500,创业板,红利低波,纳指,港股创新药",
    "跨境黄金3只": "纳指,港股创新药,黄金",
}

_queue_lock = threading.Lock()
_worker_alive = False


# ── 密钥 ──
def get_leaderboard_token():
    try:
        from web.config_local import LEADERBOARD_TOKEN
        return LEADERBOARD_TOKEN
    except ImportError:
        return ""


def check_token(tok):
    want = get_leaderboard_token()
    if not want:
        return False, "服务器未配置 LEADERBOARD_TOKEN (web/config_local.py)"
    if tok != want:
        return False, "token 错误"
    return True, ""


def _row(strategy, label, m, extra=None):
    r = {"strategy": strategy, "params": label,
         "total_ret": m["total_ret"],
         "mdd": m.get("mdd", m.get("max_dd")),
         "calmar": m.get("calmar"), "ann_ret": m.get("ann_ret"),
         "sharpe": m.get("sharpe"), "trade_count": m.get("trade_count"),
         "neg_years": m.get("neg_years")}
    if extra:
        r.update(extra)
    for k in ("nav_log", "dates", "decisions", "yearly"):
        if m.get(k) is not None:
            r[k] = m[k]
    return r


def _metrics_from_log(nav_log, actions=None):
    nav = np.array(nav_log)
    cm = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - cm) / cm * 100))
    years = max(len(nav) / 252, 0.01)
    ann = (nav[-1] ** (1 / years) - 1) * 100 if nav[-1] > 0 else -100.0
    dr = np.diff(nav) / nav[:-1]
    dr = dr[~np.isnan(dr)]
    sharpe = round(float(np.mean(dr) / np.std(dr) * np.sqrt(252)), 2) \
        if len(dr) > 10 and np.std(dr) > 1e-9 else 0.0
    m = {"total_ret": round(float((nav[-1] - 1) * 100), 1),
         "ann_ret": round(float(ann), 2), "mdd": round(mdd, 2),
         "calmar": round(ann / abs(mdd), 2) if mdd else 0.0,
         "sharpe": sharpe}
    if actions is not None:
        m["trade_count"] = len(actions)
    return m


def _run_yanyan_grid(start, end, universe=""):
    names = [u[0] for u in UNIVERSE]
    groups = [u[2] for u in UNIVERSE]
    codes = [u[1] for u in UNIVERSE]
    dfs = FEED.get_dataframe(codes, "20150301", end.replace("-", ""))
    _all = sorted(set().union(*[set(df["trade_date"]) for df in dfs.values()]))
    _all = [d for d in _all if pd.Timestamp("20150301") <= d <= pd.Timestamp(end)]
    if not _all:
        return []

    def bp(code):
        s = dfs[code].set_index("trade_date")["close"]
        return np.array([s.get(d, np.nan) for d in _all])
    pm = np.array([bp(c) for c in codes])
    hs = pm[names.index("沪深300")]

    rank_all = [i for i, g in enumerate(groups) if g != "C"]
    uni_raw = [x.strip() for x in (universe or "").split(",") if x.strip()]
    sel = [names.index(n) for n in uni_raw if n in names and names.index(n) in rank_all]
    if len(sel) < 2:
        sel = rank_all
    cash_i = names.index("货币")
    sub_idx = sel + [cash_i]
    names_sub = [names[i] for i in sel] + ["货币"]
    groups_sub = [groups[i] for i in sel] + ["C"]

    day0 = next(i for i, d in enumerate(_all) if str(d)[:10] >= start)
    day1 = next(i for i, d in enumerate(_all) if str(d)[:10] >= end)
    lo = max(0, day0 - 220)
    pm_sub = pm[sub_idx, lo:day1 + 1]
    hs_sub = hs[lo:day1 + 1]

    rows = []
    for label, p in YANYAN_GRID:
        cfg = YanyanConfig(**p)
        actions, nav, log = compute_yanyan(cfg, names_sub, groups_sub, pm_sub, hs_sub,
                                           pm_sub.shape[1])
        rows.append(_row("yanyan", label, _metrics_from_log(log, actions)))
    return rows


def run_leaderboard(params):
    strategy = params.get("strategy", "all")       # all | etf | yanyan
    metric = params.get("metric", "calmar")
    if metric not in METRICS:
        metric = "calmar"
    universe = params.get("universe", "")
    start, end = LB_START, LB_END                  # 固定窗口 (用户要求)

    t0 = time.time()
    rows = []
    errors = []
    if strategy in ("all", "etf"):
        for label, p in ETF_GRID:
            p2 = dict(p)
            p2["start_date"] = start
            p2["end_date"] = end
            r = run_backtest(p2)
            if "error" in r:
                errors.append(f"ETF[{label}]: {r['error']}")
                continue
            rows.append(_row("etf", label, r))
    if strategy in ("all", "yanyan"):
        rows += _run_yanyan_grid(start, end, universe)

    rows.sort(key=lambda r: -(r[metric] if r[metric] is not None else -1e9))
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    result = {"rows": rows, "metric": metric, "errors": errors,
              "elapsed": round(time.time() - t0, 1),
              "window": f"{start} ~ {end}",
              "server_time": time.strftime("%H:%M:%S")}
    tmp = LAST_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    os.replace(tmp, LAST_FILE)
    return result


def load_last():
    if not os.path.exists(LAST_FILE):
        return None
    try:
        with open(LAST_FILE) as f:
            return json.load(f)
    except Exception:
        return None


# ── 自然语言自定义策略 ──
def _load_queue():
    if not os.path.exists(QUEUE_FILE):
        return []
    try:
        with open(QUEUE_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save_queue(q):
    tmp = QUEUE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(q, f, ensure_ascii=False, indent=1)
    os.replace(tmp, QUEUE_FILE)


def submit_custom(description):
    desc = (description or "").strip()
    if len(desc) < 10:
        return None, "策略描述太短 (至少10字)"
    with _queue_lock:
        q = _load_queue()
        if sum(1 for x in q if x["status"] in ("pending", "running")) >= 3:
            return None, "队列已满 (最多3个待处理), 稍后再试"
        item = {"id": uuid.uuid4().hex[:8], "description": desc,
                "status": "pending", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "result": None, "error": None, "file": None}
        q.append(item)
        _save_queue(q)
        _ensure_worker()
        return item, None


def _ensure_worker():
    global _worker_alive
    if _worker_alive:
        return
    _worker_alive = True
    t = threading.Thread(target=_worker_loop, daemon=True)
    t.start()


def _worker_loop():
    global _worker_alive
    while True:
        with _queue_lock:
            q = _load_queue()
            item = next((x for x in q if x["status"] == "pending"), None)
        if item is None:
            _worker_alive = False
            return
        _process(item)


def _update_item(item_id, patch):
    with _queue_lock:
        q = _load_queue()
        for x in q:
            if x["id"] == item_id:
                x.update(patch)
        _save_queue(q)


KIMI_ENV_FILE = os.path.expanduser("~/.claude/backups/bashrc.kimi.20260813.bak")

def _agent_env():
    """claude code + kimi k3 环境: 从 kimi 配置备份读 ANTHROPIC_* (密钥不落日志);
    备份缺失时回退当前 ~/.bashrc"""
    keys = {}
    for path in (KIMI_ENV_FILE, os.path.expanduser("~/.bashrc")):
        try:
            with open(path) as f:
                for line in f:
                    m = re.match(r'export\s+(\w+)="?([^"\n]*)"?', line.strip())
                    if m and m.group(1) not in keys:
                        keys[m.group(1)] = m.group(2)
        except Exception:
            pass
    env = dict(os.environ)
    for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL",
              "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
              "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
        if keys.get(k):
            env[k] = keys[k]
    env["ANTHROPIC_MODEL"] = "k3"
    return env


def _generate(item):
    """claude code + kimi k3 无头生成单文件策略代码"""
    desc = item["description"]
    code_path = os.path.join(CUSTOM_DIR, item["id"] + ".py")
    os.makedirs(CUSTOM_DIR, exist_ok=True)
    prompt = f"""你是量化策略实现助手。请把下面的策略描述实现成一个单一 Python 文件: {code_path}
文件契约 (必须严格遵守):
1. 文件必须定义函数 run(params: dict) -> dict, 返回字典必须含以下键:
   - nav_log: 净值序列 list[float], 逐交易日, 起点 1.0 (空仓日净值不变, 已扣手续费)
   - dates: 与 nav_log 等长的日期字符串 list[str], 格式 "YYYY-MM-DD"
   - decisions: 交易决策 list[dict], 每项至少含 date(YYYY-MM-DD), action(如 买入XX/卖出XX/XX→YY), reason(原因)
   - total_ret / max_dd / ann_ret / calmar / sharpe / trade_count 也可返回, 但服务端会按 nav_log 重算,
     务必保证 nav_log 长度与 dates 一致且逐日对齐
2. 数据来源必须用 (from data.feed import TushareFeed):
   token 从 web.config_local import TUSHARE_TOKEN, cache_dir="/root/backtest/cache/etf_momentum",
   用 get_dataframe 拉取已复权 close 日线。可用 web.bt_runner.ALL_CODES 查代码表。
3. 回测窗口固定 2016-01-01 ~ 2026-07-31 (params 会传入 start_date/end_date, 以 params 为准)。
4. 全部逻辑都在这一个文件里, 自包含、可 import、无交互、不要启动任何 web 服务、不修改任何其他文件。
5. 实现后自己执行一次 run({{}}) 验证不抛异常; 结果中 nav_log 与 dates 必须等长。
策略描述:
{desc}
"""
    try:
        r = subprocess.run(["claude", "-p", "--model", "k3",
                            "--permission-mode", "acceptEdits", prompt],
                           cwd=REPO_DIR, env=_agent_env(),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=900)
    except subprocess.TimeoutExpired:
        raise RuntimeError("claude 生成超时(900s)")
    if r.returncode != 0:
        raise RuntimeError("claude 生成失败: %s" % (r.stderr or r.stdout)[-300:])
    if not os.path.exists(code_path) or os.path.getsize(code_path) < 100:
        raise RuntimeError("未生成有效代码文件: %s" % code_path)
    return code_path


BANNED_PATTERNS = (
    # 危险调用 (合法 import os/sys 不受影响)
    "os.system", "os.popen", "os.remove", "os.rename", "os.unlink", "os.chmod",
    "os.chown", "os.kill", "os.execl", "os.execv", "os.startfile", "os.mkdir",
    "subprocess", "socket", "shutil", "import requests", "from requests",
    "urllib", "http.client", "ftplib", "telnetlib", "eval(", "exec(", "__import__",
    "open(", "remove(", "unlink", "rmtree", "chmod", "chown", "kill(", "signal",
    "multiprocessing", "threading", ".bashrc", "MYTRACK_KEY", "LEADERBOARD_TOKEN",
)


def _safe_scan(code):
    """生成代码静态安全检查: 命中危险模式即拒执行 (防注入式描述生成恶意代码)"""
    hits = [p for p in BANNED_PATTERNS if p in code]
    return hits


def _limit_resources():
    """子进程资源限制: CPU/内存/文件/进程数上限 (防死循环/刷盘/fork炸弹)"""
    import resource
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 ** 3, 4 * 1024 ** 3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 ** 2, 256 * 1024 ** 2))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
    except Exception:
        pass


def _execute(item):
    """子进程执行生成的 run(params), 返回指标 dict (静态扫描 + 资源限制 + 环境隔离)"""
    code_path = os.path.join(CUSTOM_DIR, item["id"] + ".py")
    with open(code_path) as f:
        code = f.read()
    hits = _safe_scan(code)
    if hits:
        raise RuntimeError("代码未通过安全扫描 (命中: %s)" % ",".join(hits[:5]))
    workdir = os.path.join(CUSTOM_DIR, item["id"])
    os.makedirs(workdir, exist_ok=True)
    # 隔离环境: 清除全部密钥类变量, 只留最小运行环境
    env = {k: v for k, v in os.environ.items()
           if not any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "CREDENTIAL", "PASSWORD"))}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env.setdefault("HOME", "/root")
    script = (
        "import sys, json, importlib.util\n"
        "sys.path.insert(0, '/root/etf_strategy')\n"
        "spec = importlib.util.spec_from_file_location('lb_%s', %r)\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "params = {'start_date': %r, 'end_date': %r}\n"
        "res = m.run(params)\n"
        "out = {k: res.get(k) for k in ('total_ret','max_dd','mdd','ann_ret','calmar','sharpe','trade_count',\n"
        "                               'nav_log','dates','decisions')}\n"
        "print(json.dumps(out, default=str))\n"
    ) % (item["id"], code_path, LB_START, LB_END)
    try:
        r = subprocess.run(["python3", "-c", script], cwd=workdir, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=600,
                           preexec_fn=_limit_resources)
    except subprocess.TimeoutExpired:
        raise RuntimeError("回测执行超时(600s)")
    if r.returncode != 0:
        raise RuntimeError("回测执行失败: %s" % (r.stderr or r.stdout)[-300:])
    try:
        m = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        raise RuntimeError("回测结果解析失败: %s" % r.stdout[-200:])
    nav_log = m.get("nav_log") or []
    dates = m.get("dates") or []
    decisions = m.get("decisions") or []
    if not nav_log or len(nav_log) < 30:
        raise RuntimeError("生成代码未返回有效 nav_log (契约要求逐日净值序列)")
    if len(dates) != len(nav_log):
        dates = [None] * len(nav_log)
    mm = _recompute_metrics(nav_log, dates, decisions)
    mm["nav_log"] = [round(float(v), 4) for v in nav_log]
    mm["dates"] = [str(d)[:10] for d in dates]
    mm["decisions"] = decisions
    mm["trade_count"] = len(decisions) if decisions else None
    return _row("custom", item["description"][:60],
                mm, extra={"id": item["id"], "file": "lb_custom/%s.py" % item["id"]})


def _process(item):
    _update_item(item["id"], {"status": "running"})
    try:
        _generate(item)
        _update_item(item["id"], {"status": "running", "file": "lb_custom/%s.py" % item["id"]})
        row = _execute(item)
        _update_item(item["id"], {"status": "done", "result": row})
    except Exception as e:
        _update_item(item["id"], {"status": "failed", "error": str(e)[:300]})


def _recompute_metrics(nav_log, dates=None, decisions=None):
    """按 nav_log 重算指标 (服务端统一口径); 缺失时返回 None 值"""
    nav = np.array([float(v) for v in nav_log if v is not None], dtype=float)
    if len(nav) == 0:
        return {"total_ret": None, "mdd": None, "ann_ret": None,
                "calmar": None, "sharpe": None}
    cm = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - cm) / cm * 100))
    years = max(len(nav) / 252, 0.01)
    ann = (nav[-1] ** (1 / years) - 1) * 100 if nav[-1] > 0 else -100.0
    dr = np.diff(nav) / nav[:-1]
    dr = dr[~np.isnan(dr)]
    sharpe = round(float(np.mean(dr) / np.std(dr) * np.sqrt(252)), 2) \
        if len(dr) > 10 and np.std(dr) > 1e-9 else 0.0
    yearly = {}
    if dates is not None and len(dates) == len(nav):
        by = {}
        for i, d in enumerate(dates):
            y = str(d)[:4]
            by[y] = nav[i]
        prev = 1.0
        for y in sorted(by):
            yearly[y] = round((by[y] / prev - 1) * 100, 2)
            prev = by[y]
    return {"total_ret": round(float((nav[-1] - 1) * 100), 1),
            "mdd": round(mdd, 2), "ann_ret": round(float(ann), 2),
            "calmar": round(ann / abs(mdd), 2) if mdd else 0.0,
            "sharpe": sharpe, "yearly": yearly,
            "trade_count": len(decisions) if decisions else None}


def custom_status():
    return _load_queue()


def custom_code(item_id):
    p = os.path.join(CUSTOM_DIR, item_id + ".py")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return f.read()


def custom_result(item_id):
    """从队列里取完整结果 (含 nav_log/dates/decisions), 供详情页渲染"""
    for it in _load_queue():
        if it["id"] == item_id and it.get("result"):
            return it["result"]
    return None
