"""ETF 动量策略回测 Web 界面"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
