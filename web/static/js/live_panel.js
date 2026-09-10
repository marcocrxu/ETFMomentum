/* 实盘实况面板 — /live 页与 mytrack 页共用的展示组件
   数据来自 /api/live/today (web/live_page.py 装配, 共享引擎口径)。
   本文件只做渲染, 不含任何策略逻辑 (AGENT.md §7b 红线)。 */
var LivePanel = (function () {
  var MEDALS = ["🥇", "🥈", "🥉"];

  function fmt(v, d) { return (v === null || v === undefined) ? "—" : Number(v).toFixed(d === undefined ? 2 : d); }
  function sign(v) { return v >= 0 ? "+" : ""; }
  function cls(v) { return v >= 0 ? "lp-pos" : "lp-neg"; }

  function render(el, d) {
    if (!d.ok) {
      el.innerHTML = '<div class="lp-panel">⚠️ ' + (d.error || "数据不可用") + "</div>";
      return;
    }
    var h = "";

    // 信号横幅 + 操作建议
    h += '<div class="lp-banner ' + d.suggestion_level + '">' +
      '<div class="lp-sig">' + d.date + " · 今日信号: <b>" + d.signal.signal +
      (d.signal.target ? " → " + d.signal.target : "") + "</b></div>" +
      '<div class="lp-act">' + d.suggestion + "</div>" +
      (d.signal.reason ? '<div class="lp-reason">依据: ' + d.signal.reason + "</div>" : "") +
      "</div>";

    // 持仓偏离告警
    if (d.diverged) {
      h += '<div class="lp-alert">🚨 <b>持仓偏离</b>：实际持仓 <b>' +
        (d.actual_position ? d.actual_position.position : "空仓") + "</b>" +
        " vs 策略仓位 <b>" + (d.strategy_position.position || "空仓") + "</b> 不一致</div>";
    }

    // 动量卡片
    h += '<div class="lp-grid">';
    d.cards.forEach(function (c) {
      h += '<div class="lp-card">' +
        '<span class="lp-medal">' + (MEDALS[c.rank] || "") + "</span>" +
        '<div class="lp-name">' + c.name + "</div>" +
        '<div class="lp-mom ' + cls(c.momentum) + '">' + sign(c.momentum) + fmt(c.momentum) + "%</div>" +
        '<div class="lp-price">现价 ¥' + fmt(c.price, 4) + "</div>" +
        (c.is_strategy_pos ? '<span class="lp-tag sim">策略仓位</span>' : "") +
        (c.is_actual_pos ? '<span class="lp-tag act">实际持有</span>' : "") +
        "</div>";
    });
    h += "</div>";

    // 市场环境
    h += '<div class="lp-panel"><h3>市场环境</h3>' +
      '<div class="lp-row"><span>上证指数 20日动量</span><b class="' + cls(d.sh_momentum) + '">' +
      sign(d.sh_momentum) + fmt(d.sh_momentum) + "%</b></div>" +
      '<div class="lp-row"><span>纳指动量口径</span><span>' +
      (d.nz_mix_active ? "🧮 " + d.nz_note
                       : '<span class="lp-bad">⚠️ 降级: ' + d.nz_note + "</span>") +
      "</span></div></div>";

    // 仓位对比
    var sp = d.strategy_position, ap = d.actual_position;
    h += '<div class="lp-panel"><h3>仓位</h3>' +
      '<div class="lp-row"><span>策略仓位</span><b>' + (sp.position || "空仓") +
      (sp.position ? "（自 " + sp.entry_date + " 起）" : "") + "</b></div>";
    if (ap) {
      h += '<div class="lp-row"><span>实际持仓</span><b>' + ap.position +
        (ap.shares ? " " + ap.shares + "股" : "") + "</b></div>" +
        '<div class="lp-row"><span>入场</span><span>' + ap.entry_date + " @ ¥" + fmt(ap.entry_price, 3) + "</span></div>" +
        '<div class="lp-row"><span>浮动盈亏</span><b class="' + cls(ap.pnl_pct || 0) + '">' +
        (ap.pnl_pct === null ? "—" : sign(ap.pnl_pct) + fmt(ap.pnl_pct) + "%") + "</b></div>";
    } else {
      h += '<div class="lp-row"><span>实际持仓</span><b>空仓</b></div>';
    }
    h += "</div>";

    // 策略最近操作
    if (d.recent_actions && d.recent_actions.length) {
      h += '<div class="lp-panel"><h3>策略最近操作（全量回放）</h3><table class="lp-table">' +
        "<tr><th>日期</th><th>操作</th><th>依据</th></tr>";
      d.recent_actions.slice().reverse().forEach(function (a) {
        h += "<tr><td>" + a.date + "</td><td>" + a.old + " → " + a.new + "</td><td>" + a.reason + "</td></tr>";
      });
      h += "</table></div>";
    }

    // 每日对账状态
    if (d.last_reconcile) {
      h += '<div class="lp-panel"><h3>每日对账</h3>' +
        '<div class="' + (d.last_reconcile.ok ? "lp-ok" : "lp-bad") +
        '" style="white-space:pre-line;font-size:12px">' + d.last_reconcile.message + "</div>" +
        '<div class="lp-note">' + d.last_reconcile.ts + "</div></div>";
    }

    h += '<div class="lp-note">更新于 ' + d.server_time + " · 每 60s 自动刷新</div>";
    el.innerHTML = h;
  }

  function mount(elId) {
    var el = document.getElementById(elId);
    if (!el) return;
    function load() {
      fetch("/api/live/today").then(function (r) { return r.json(); })
        .then(function (d) { render(el, d); })
        .catch(function (e) { el.innerHTML = '<div class="lp-panel">⚠️ 加载失败: ' + e + "</div>"; });
    }
    load();
    setInterval(load, 60000);
  }

  return { mount: mount };
})();
