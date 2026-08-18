/* yanyan.js — yanyan 策略页: 参数收集 + 回测请求 + 渲染 (复用 charts.js) */

let navChart = null, ddChart = null, yearlyChart = null, posChart = null;
let currentController = null;

function collectParams() {
  return {
    regime_ma: +document.getElementById('regime_ma').value || 200,
    switch_buffer: +document.getElementById('switch_buffer').value || 2.0,
    overheat_mom: +document.getElementById('overheat_mom').value || 40,
    overheat_q: +document.getElementById('overheat_q').value || 3.0,
    cash_park: document.getElementById('cash_park').checked,
    cash_skip_buffer: document.getElementById('cash_skip_buffer').checked,
    start_date: document.getElementById('start_date').value,
    end_date: document.getElementById('end_date').value,
    commission: +document.getElementById('commission').value / 100,
    slippage: +document.getElementById('slippage').value / 100,
  };
}

function showError(msg) {
  const b = document.getElementById('error-banner');
  b.textContent = msg;
  b.style.display = 'block';
}

function hideError() {
  const b = document.getElementById('error-banner');
  b.style.display = 'none';
  b.textContent = '';
}

function setLoading(show) {
  document.getElementById('loading').style.display = show ? 'flex' : 'none';
  document.getElementById('btn-run').disabled = show;
}

async function runYanyan() {
  hideError();
  const params = collectParams();
  if (!params.start_date || !params.end_date || params.start_date >= params.end_date) {
    showError('请设置正确的起止日期');
    return;
  }
  if (currentController) currentController.abort();
  currentController = new AbortController();
  setLoading(true);
  try {
    const resp = await fetch('/api/yanyan/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
      signal: currentController.signal
    });
    const json = await resp.json();
    if (!json.success) {
      showError(json.error || '回测失败');
      return;
    }
    render(json.data);
  } catch (e) {
    if (e.name !== 'AbortError') showError('请求失败: ' + e.message);
  } finally {
    setLoading(false);
    currentController = null;
  }
}

function render(data) {
  document.getElementById('kpi-grid').style.display = 'grid';
  document.getElementById('charts-toolbar').style.display = 'flex';
  document.getElementById('charts-grid').style.display = 'grid';
  document.getElementById('table-card').style.display = 'block';
  renderKPI(data);
  renderCharts(data);
  renderTrades(data);
  setTimeout(() => {
    navChart && navChart.resize();
    ddChart && ddChart.resize();
    yearlyChart && yearlyChart.resize();
    posChart && posChart.resize();
  }, 50);
}

function renderKPI(data) {
  const set = (id, val, cls) => {
    const el = document.getElementById(id);
    el.textContent = val;
    el.className = 'kpi-value ' + (cls || '');
  };
  const sign = v => v >= 0 ? 'positive' : 'negative';
  set('kpi-total', (data.total_ret >= 0 ? '+' : '') + data.total_ret + '%', sign(data.total_ret));
  set('kpi-ann', (data.ann_ret >= 0 ? '+' : '') + data.ann_ret + '%', sign(data.ann_ret));
  set('kpi-dd', data.max_dd + '%', 'negative');
  set('kpi-calmar', data.calmar);
  set('kpi-sharpe', data.sharpe != null ? data.sharpe : '--');
  set('kpi-vol', data.volatility != null ? data.volatility + '%' : '--');
  set('kpi-trades', data.trade_count);
  set('kpi-neg', data.neg_years);
  set('kpi-pos', data.last_position || '--');
  set('kpi-range', data.data_range || '--');
}

function renderCharts(data) {
  if (!navChart) navChart = ChartRenderers.initChart('nav-chart');
  if (!ddChart) ddChart = ChartRenderers.initChart('drawdown-chart');
  if (!yearlyChart) yearlyChart = ChartRenderers.initChart('yearly-chart');
  if (!posChart) posChart = ChartRenderers.initChart('position-chart');
  ChartRenderers.renderNavChart(navChart, data);
  ChartRenderers.renderDrawdownChart(ddChart, data);
  ChartRenderers.renderYearlyChart(yearlyChart, data);
  ChartRenderers.renderPositionChart(posChart, data);
  ChartRenderers.setChartsLinked([navChart, ddChart, yearlyChart, posChart],
    document.getElementById('link-zoom').checked);
}

function renderTrades(data) {
  const tbody = document.getElementById('trades-body');
  const decisions = data.decisions || [];
  if (decisions.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#8b92a8;padding:24px">暂无决策记录</td></tr>`;
    return;
  }
  const badge = r => {
    if (r === '牛熊开关') return '<span class="badge-tag badge-crash">熊市</span>';
    if (r === '过热离场') return '<span class="badge-tag" style="background:rgba(255,152,0,.15);color:#ff9800">过热</span>';
    return '';
  };
  tbody.innerHTML = decisions.map(d => {
    const cls = (d.pnl || 0) >= 0 ? 'positive' : 'negative';
    const pnlText = d.pnl != null ? `${d.pnl >= 0 ? '+' : ''}${d.pnl}%` : '-';
    return `
      <tr>
        <td>${d.date || '-'}</td>
        <td>${d.action || '-'} ${badge(d.reason)}</td>
        <td>${d.price != null ? d.price.toFixed(3) : '-'}</td>
        <td class="${cls}">${pnlText}</td>
        <td>${d.nav != null ? d.nav.toFixed(4) : '-'}</td>
        <td>${d.days != null ? d.days + '天' : '-'}</td>
        <td>${d.reason || '-'}</td>
      </tr>`;
  }).join('');
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btn-run').addEventListener('click', runYanyan);
});
