"""ETF 动量策略回测 Web 界面"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, abort
from web.bt_runner import run_backtest

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8082, debug=False)
