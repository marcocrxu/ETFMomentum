"""ETF 动量策略回测 Web 界面"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, abort, session
from web.bt_runner import run_backtest

app = Flask(__name__)

try:
    from web.config_local import SESSION_SECRET
except ImportError:
    import secrets
    SESSION_SECRET = secrets.token_hex(32)
app.secret_key = SESSION_SECRET
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = True  # 仅经 HTTPS 发送, 防 HTTP 明文嗅探

@app.after_request
def _security_headers(resp):
    # 密钥在 URL 里, 禁止任何跨站 Referer 泄露
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Frame-Options"] = "DENY"
    return resp

# ── http 通道已禁用: 一律 403 (nginx 层还会直接断连; 此处在应用层兜底) ──
@app.before_request
def _force_https():
    if request.headers.get("X-Forwarded-Proto") == "http":
        return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<title>🚫 HTTP 已禁用</title></head><body style='font-family:sans-serif;padding:40px'>"
                "<h2>🚫 HTTP 通道已禁用</h2>"
                "<p>请使用 <b>https://8.219.57.219</b> 访问。</p></body></html>", 403)

# ── 全站门禁: 必须先访问 /mytrack/<正确密钥> 解锁, 其他页面/API 才可访问 ──
@app.before_request
def _gate():
    p = request.path
    if p.startswith("/mytrack/") or p.startswith("/static/") or p == "/favicon.ico":
        return None  # mytrack 本身靠路径密钥保护; /static 是 mytrack 页面的资源
    if session.get("authed"):
        return None
    app.logger.warning("gate 403: ip=%s path=%s", request.headers.get("X-Forwarded-For") or request.remote_addr, p)
    return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>🔒 需要解锁</title></head><body style='font-family:sans-serif;padding:40px'>"
            "<h2>🔒 需要先访问私密对账页</h2>"
            "<p>请用 <b>https://8.219.57.219</b> 打开私密对账页（mytrack 地址）解锁一次，"
            "再访问本页面；http 地址会因安全策略始终无法解锁。</p></body></html>", 403)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/yanyan")
def yanyan():
    return render_template("yanyan.html")

@app.route("/leaderboard")
def leaderboard():
    return render_template("leaderboard.html")

@app.route("/api/yanyan/run", methods=["POST"])
def api_yanyan_run():
    from web.yanyan_runner import run_yanyan
    params = request.get_json(silent=True) or {}
    result = run_yanyan(params)
    if "error" in result:
        return jsonify({"success": False, "error": result["error"]}), 400
    return jsonify({"success": True, "data": result})

@app.route("/api/leaderboard/run", methods=["POST"])
def api_leaderboard_run():
    from web.leaderboard import run_leaderboard, check_token
    params = request.get_json(silent=True) or {}
    ok, err = check_token(params.get("token"))
    if not ok:
        return jsonify({"success": False, "error": err}), 403
    return jsonify({"success": True, "data": run_leaderboard(params)})

@app.route("/api/leaderboard/custom", methods=["POST"])
def api_leaderboard_custom():
    from web.leaderboard import submit_custom, check_token
    params = request.get_json(silent=True) or {}
    ok, err = check_token(params.get("token"))
    if not ok:
        return jsonify({"success": False, "error": err}), 403
    item, err = submit_custom(params.get("description"))
    if err:
        return jsonify({"success": False, "error": err}), 400
    return jsonify({"success": True, "data": item})

@app.route("/api/leaderboard/custom/status")
def api_leaderboard_custom_status():
    from web.leaderboard import custom_status
    return jsonify({"success": True, "data": custom_status()})

@app.route("/lb/code/<item_id>")
def lb_code_view(item_id):
    from web.leaderboard import custom_code, check_token
    if not item_id.isalnum() or len(item_id) > 16:
        abort(404)
    ok, _ = check_token(request.args.get("token", ""))
    if not ok:
        abort(404)                      # 错误 token 一律 404, 不暴露页面存在
    code = custom_code(item_id)
    if code is None:
        abort(404)
    return "<pre style='padding:20px;line-height:1.5'>%s</pre>" % code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

@app.route("/lb/view/<item_id>")
def lb_view_page(item_id):
    from web.leaderboard import custom_result, check_token
    if not item_id.isalnum() or len(item_id) > 16:
        abort(404)
    ok, _ = check_token(request.args.get("token", ""))
    if not ok:
        abort(404)
    if custom_result(item_id) is None:
        abort(404)
    return render_template("lb_view.html", item_id=item_id,
                           token=request.args.get("token", ""))

@app.route("/api/leaderboard/custom/result", methods=["POST"])
def api_leaderboard_custom_result():
    from web.leaderboard import custom_result, check_token
    params = request.get_json(silent=True) or {}
    ok, _ = check_token(params.get("token", ""))
    if not ok:
        return jsonify({"success": False, "error": "token 错误"}), 403
    r = custom_result(params.get("id", ""))
    if r is None:
        return jsonify({"success": False, "error": "记录不存在"}), 404
    return jsonify({"success": True, "data": r})

@app.route("/api/leaderboard/custom/code", methods=["POST"])
def api_leaderboard_custom_code():
    from web.leaderboard import custom_code, check_token
    params = request.get_json(silent=True) or {}
    ok, _ = check_token(params.get("token", ""))
    if not ok:
        return jsonify({"success": False, "error": "token 错误"}), 403
    code = custom_code(params.get("id", ""))
    if code is None:
        return jsonify({"success": False, "error": "代码不存在"}), 404
    return jsonify({"success": True, "data": {"id": params.get("id"), "code": code}})

@app.route("/api/leaderboard/last")
def api_leaderboard_last():
    from web.leaderboard import load_last
    data = load_last()
    return jsonify({"success": data is not None, "data": data})

@app.route("/api/bt/run", methods=["POST"])
def api_bt_run():
    params = request.get_json(silent=True) or {}
    result = run_backtest(params)
    if "error" in result:
        return jsonify({"success": False, "error": result["error"]}), 400
    return jsonify({"success": True, "data": result})


# ── 私密实盘对账页: 路径即密钥, 错误密钥一律 404(不暴露页面存在) ──
def _check_key(key):
    from web.config_local import MYTRACK_KEY
    if key != MYTRACK_KEY:
        abort(404)
    session["authed"] = True   # 正确密钥 → 种下解锁 cookie

@app.route("/mytrack/<key>")
def mytrack(key):
    _check_key(key)
    from web.mytrack import compute_track
    resp = app.make_response(render_template("mytrack.html", key=key, data=compute_track()))
    resp.headers["X-Robots-Tag"] = "noindex, nofollow"
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.route("/mytrack/<key>/save", methods=["POST"])
def mytrack_save(key):
    _check_key(key)
    from web.mytrack import validate_records, save_records
    body = request.get_json(silent=True) or {}
    records = body.get("records", [])
    err, records = validate_records(records, body.get("initial_cash"))
    if err:
        return jsonify({"success": False, "error": err}), 400
    save_records(records, body.get("initial_cash"))
    return jsonify({"success": True})


@app.route("/mytrack/<key>/combos/save", methods=["POST"])
def mytrack_combos_save(key):
    _check_key(key)
    from web.mytrack import validate_combos, save_combos
    body = request.get_json(silent=True) or {}
    err, combos = validate_combos(body.get("combos", []))
    if err:
        return jsonify({"success": False, "error": err}), 400
    save_combos(combos)
    return jsonify({"success": True})


@app.route("/mytrack/<key>/live", methods=["GET"])
def mytrack_live(key):
    _check_key(key)
    from web.mytrack import live_quotes
    assets = [a for a in request.args.get("assets", "").split(",") if a]
    return jsonify({"quotes": live_quotes(assets)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8082, debug=False)
