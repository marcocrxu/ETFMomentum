/* app.js - 页面交互与回测控制 */

const ALL_ETFS = [
  { name: '纳指', code: '513100' },
  { name: '黄金', code: '518880' },
  { name: '创业', code: '159915' },
  { name: '科创50', code: '588000' },
  { name: '沪深300', code: '510300' },
  { name: '上证50', code: '510050' },
  { name: '中证2000', code: '563300' },
  { name: '港股创新药', code: '513120' },
  { name: '有色', code: '512400' },
  { name: '通信', code: '515880' },
  { name: '半导体', code: '512480' },
  { name: '红利低波', code: '512890' },
  { name: '5年国债', code: '511010' },
  { name: '30年国债', code: '511090' }
];

const DEFAULT_SELECTED = ['纳指', '黄金', '创业'];

let navChart = null;
let ddChart = null;
let yearlyChart = null;
let posChart = null;
let currentController = null;

function initETFChips() {
  const container = document.getElementById('etf-chips');
  container.innerHTML = ALL_ETFS.map(etf => `
    <label class="etf-chip">
      <input type="checkbox" class="etf-cb" value="${etf.name}" ${DEFAULT_SELECTED.includes(etf.name) ? 'checked' : ''}>
      <span class="chip-label">${etf.name}<span class="code">${etf.code}</span></span>
    </label>
  `).join('');
}

function bindEvents() {
  document.getElementById('btn-run').addEventListener('click', runBacktest);
  document.getElementById('btn-export').addEventListener('click', exportConfig);
  document.getElementById('btn-import').addEventListener('click', () => document.getElementById('import-input').click());
  document.getElementById('import-input').addEventListener('change', importConfig);

  document.querySelectorAll('.shortcut-btns button').forEach(btn => {
    btn.addEventListener('click', () => setDateRange(parseInt(btn.dataset.days, 10)));
  });

  const momSel = document.getElementById('mom_period');
  const momCustom = document.getElementById('mom_period_custom');
  momSel.addEventListener('change', () => {
    momCustom.style.display = momSel.value === '-1' ? 'block' : 'none';
  });

  document.getElementById('link-zoom').addEventListener('change', (e) => {
    ChartRenderers.setChartsLinked([navChart, ddChart, yearlyChart, posChart], e.target.checked);
  });

  window.addEventListener('resize', () => {
    navChart && navChart.resize();
    ddChart && ddChart.resize();
    yearlyChart && yearlyChart.resize();
    posChart && posChart.resize();
  });
}

function setDateRange(days) {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - days);
  document.getElementById('end_date').value = formatDate(end);
  document.getElementById('start_date').value = formatDate(start);
}

function formatDate(d) {
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function collectParams() {
  const etfs = [...document.querySelectorAll('.etf-cb:checked')].map(c => c.value);
  const momSel = document.getElementById('mom_period');
  const momCustom = document.getElementById('mom_period_custom');
  const mom_period = momSel.value === '-1'
    ? (parseInt(momCustom.value, 10) || 20)
    : parseInt(momSel.value, 10);

  return {
    start_date: document.getElementById('start_date').value,
    end_date: document.getElementById('end_date').value,
    universe: etfs.join(','),
    sh_cash: document.getElementById('sh_cash').value,
    method: document.getElementById('method').value,
    mom_period: mom_period,
    accel_sigma: +document.getElementById('accel_sigma').value,
    accel_on: document.getElementById('accel_on').checked,
    crash_sigma: +document.getElementById('crash_sigma').value,
    crash_on: document.getElementById('crash_on').checked,
    crash_lock: +document.getElementById('crash_lock').value,
    crash_alloc: +document.getElementById('crash_alloc').value || 100,
    min_hold: +document.getElementById('min_hold').value,
    min_gap: +document.getElementById('min_gap').value,
    stop_loss: +document.getElementById('stop_loss').value,
    exception_gap: +document.getElementById('exception_gap').value,
    confirm_days: +document.getElementById('confirm_days').value,
    commission: +document.getElementById('commission').value / 100,
    slippage: +document.getElementById('slippage').value / 100,
    abs_mom_on: document.getElementById('abs_mom_on').checked,
    defensive_on: document.getElementById('defensive_on').checked,
    risk_adj_on: document.getElementById('risk_adj_on').checked,
    accel_cooldown: +document.getElementById('accel_cooldown').value || 0,
    accel_cooldown_gold_only: document.getElementById('accel_cooldown_gold_only').checked,
    stop_cooldown: +document.getElementById('stop_cooldown').value || 0,
    sh_cash_cooldown: +document.getElementById('sh_cash_cooldown').value || 0,
    multi_mom_on: document.getElementById('multi_mom_on').checked,
    fip_on: document.getElementById('fip_on').checked,
    top_k: +document.getElementById('top_k').value || 1,
    mom_skip: +document.getElementById('mom_skip').value || 0,
    vix_gate: +document.getElementById('vix_gate').value || 0,
    skew_gate: +document.getElementById('skew_gate').value || 0,
    nz_mom_qqq_on: document.getElementById('nz_mom_qqq_on').checked,
    nz_mom_qqq_w: +document.getElementById('nz_mom_qqq_w').value || 0.7,
    ridge_on: document.getElementById('ridge_on').checked,
    mom_type: document.getElementById('mom_type').value,
    abs_stop_on: document.getElementById('abs_stop_on').checked,
  };
}

function validateParams(params) {
  if (!params.universe || params.universe.split(',').length < 2) {
    return '至少选择 2 个标的';
  }
  if (!params.start_date || !params.end_date) {
    return '请设置起始和结束日期';
  }
  if (params.start_date >= params.end_date) {
    return '结束日期必须晚于起始日期';
  }
  if (!params.mom_period || params.mom_period < 1) {
    return '动量周期必须为正整数';
  }
  return null;
}

function showError(msg) {
  const banner = document.getElementById('error-banner');
  banner.textContent = msg;
  banner.style.display = 'block';
}

function hideError() {
  const banner = document.getElementById('error-banner');
  banner.style.display = 'none';
  banner.textContent = '';
}

function setLoading(show, text) {
  const overlay = document.getElementById('loading');
  const btn = document.getElementById('btn-run');
  document.getElementById('loading-text').textContent = text || '回测计算中...';
  overlay.style.display = show ? 'flex' : 'none';
  btn.disabled = show;
}

async function runBacktest() {
  hideError();
  const params = collectParams();
  const err = validateParams(params);
  if (err) {
    showError(err);
    return;
  }

  if (currentController) {
    currentController.abort();
  }
  currentController = new AbortController();

  setLoading(true, '正在拉取数据并计算信号...');

  try {
    const resp = await fetch('/api/bt/run', {
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
    if (e.name !== 'AbortError') {
      showError('请求失败: ' + e.message);
    }
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

  // 容器刚显示时 ECharts 尺寸可能为 0，延迟 resize 一次
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

  const signClass = v => v >= 0 ? 'positive' : 'negative';

  set('kpi-total', (data.total_ret >= 0 ? '+' : '') + data.total_ret + '%', signClass(data.total_ret));
  set('kpi-ann', (data.ann_ret >= 0 ? '+' : '') + data.ann_ret + '%', signClass(data.ann_ret));
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

  // 窗口 ≤2 年时标题同步改为月度收益
  const yt = document.getElementById('yearly-title');
  if (yt) yt.textContent = ((data.dates || []).length <= 504) ? '📅 月度收益' : '📅 年度收益';

  // 按开关状态设置四图联动/独立缩放
  const linked = document.getElementById('link-zoom').checked;
  ChartRenderers.setChartsLinked([navChart, ddChart, yearlyChart, posChart], linked);
}

// ── 决策记录多维度排序 ──
let tradeSort = { key: 'date', dir: -1 };   // 默认日期降序(最近操作在最前), 点列头切换
let lastTradeDecisions = [];

const TRADE_SORT_KEYS = {
  date:   d => d.date || '',
  action: d => d.action || '',
  price:  d => (d.price != null ? d.price : null),
  pnl:    d => (d.pnl != null ? d.pnl : null),
  nav:    d => (d.nav != null ? d.nav : null),
  days:   d => (d.days != null ? d.days : null),
  reason: d => d.reason || '',
};

function cmpTrades(a, b, key, dir) {
  const va = TRADE_SORT_KEYS[key](a), vb = TRADE_SORT_KEYS[key](b);
  const na = va == null, nb = vb == null;
  if (na && nb) return 0;
  if (na) return 1;          // 空值永远排最后
  if (nb) return -1;
  if (va < vb) return -dir;
  if (va > vb) return dir;
  const da = a.date || '', db = b.date || '';
  return da < db ? -1 : da > db ? 1 : 0;   // 同值按日期升序稳定
}

function bindTradeSort() {
  const thead = document.querySelector('#table-card thead');
  if (!thead) return;
  const setInd = () => {
    thead.querySelectorAll('th[data-sort] .sort-ind').forEach(s => s.textContent = '↕');
    const th = thead.querySelector('th[data-sort="' + tradeSort.key + '"]');
    if (th) th.querySelector('.sort-ind').textContent = tradeSort.dir === 1 ? '▲' : '▼';
  };
  thead.querySelectorAll('th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.getAttribute('data-sort');
      if (tradeSort.key === key) {
        tradeSort.dir *= -1;
      } else {
        tradeSort.key = key;
        tradeSort.dir = 1;   // 换新列默认升序
      }
      setInd();
      renderTrades({ decisions: lastTradeDecisions });
    });
  });
  setInd();
}

function renderTrades(data) {
  const tbody = document.getElementById('trades-body');
  const decisions = (data.decisions || []).slice();
  lastTradeDecisions = data.decisions || [];
  if (decisions.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#8b92a8;padding:24px">暂无决策记录</td></tr>`;
    return;
  }

  decisions.sort((a, b) => cmpTrades(a, b, tradeSort.key, tradeSort.dir));

  tbody.innerHTML = decisions.map(d => {
    const pnlClass = (d.pnl || 0) >= 0 ? 'positive' : 'negative';
    const pnlText = d.pnl != null ? `${d.pnl >= 0 ? '+' : ''}${d.pnl}%` : '-';
    const isCrash = d.reason === '恐慌抄底';
    const isPartial = isCrash && d.action.indexOf('%') >= 0;   // 部分仓位抄底
    const rowBg = isCrash ? 'background:rgba(230,126,34,.08)' : '';
    const crashBadge = isCrash
      ? `<span class="badge-tag badge-crash">${isPartial ? '部分抄底' : '抄底'}</span>`
      : '';
    return `
      <tr style="${rowBg}">
        <td>${d.date || '-'}</td>
        <td>${d.action || '-'} ${crashBadge}</td>
        <td>${d.price != null ? d.price.toFixed(3) : '-'}</td>
        <td class="${pnlClass}">${pnlText}</td>
        <td>${d.nav != null ? d.nav.toFixed(4) : '-'}</td>
        <td>${d.days != null ? d.days + '天' : '-'}</td>
        <td>${d.reason || '-'}</td>
      </tr>
    `;
  }).join('');
}

function exportConfig() {
  const cfg = collectParams();
  const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `etf_strategy_config_${formatDate(new Date())}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function importConfig(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(evt) {
    try {
      const cfg = JSON.parse(evt.target.result);
      applyConfig(cfg);
      hideError();
      runBacktest();
    } catch (err) {
      showError('配置文件解析失败: ' + err.message);
    }
  };
  reader.readAsText(file);
  e.target.value = '';
}

function applyConfig(cfg) {
  if (cfg.universe) {
    const selected = cfg.universe.split(',').map(s => s.trim());
    document.querySelectorAll('.etf-cb').forEach(cb => {
      cb.checked = selected.includes(cb.value);
    });
  }
  const setVal = (id, val) => {
    if (val != null) {
      const el = document.getElementById(id);
      if (el) el.value = val;
    }
  };
  setVal('start_date', cfg.start_date);
  setVal('end_date', cfg.end_date);
  const scv = String(cfg.sh_cash);
  setVal('sh_cash', scv === '2' ? '2' : ((cfg.sh_cash === true || scv === 'true' || scv === '1') ? '1' : '0'));
  setVal('method', cfg.method);
  if (cfg.mom_period != null) {
    const presets = ['5', '10', '20', '30'];
    const momSel = document.getElementById('mom_period');
    const momCustom = document.getElementById('mom_period_custom');
    if (presets.includes(String(cfg.mom_period))) {
      momSel.value = String(cfg.mom_period);
      momCustom.style.display = 'none';
    } else {
      momSel.value = '-1';
      momCustom.style.display = 'block';
      momCustom.value = cfg.mom_period;
    }
  }

  const setChecked = (id, val) => {
    if (val != null) document.getElementById(id).checked = Boolean(val);
  };
  setChecked('accel_on', cfg.accel_on);
  setChecked('crash_on', cfg.crash_on);
  setChecked('abs_mom_on', cfg.abs_mom_on);
  setChecked('defensive_on', cfg.defensive_on);
  setChecked('risk_adj_on', cfg.risk_adj_on);
  setChecked('multi_mom_on', cfg.multi_mom_on);
  setChecked('fip_on', cfg.fip_on);
  setVal('accel_cooldown', cfg.accel_cooldown);
  setChecked('accel_cooldown_gold_only', cfg.accel_cooldown_gold_only);
  setVal('stop_cooldown', cfg.stop_cooldown);
  setVal('sh_cash_cooldown', cfg.sh_cash_cooldown);
  setVal('top_k', cfg.top_k);
  setVal('mom_skip', cfg.mom_skip);
  setVal('vix_gate', cfg.vix_gate);
  setVal('skew_gate', cfg.skew_gate);
  setChecked('nz_mom_qqq_on', cfg.nz_mom_qqq_on);
  setVal('nz_mom_qqq_w', cfg.nz_mom_qqq_w);
  setChecked('ridge_on', cfg.ridge_on);
  setVal('mom_type', cfg.mom_type);
  setChecked('abs_stop_on', cfg.abs_stop_on);
  setVal('accel_sigma', cfg.accel_sigma);
  setVal('crash_sigma', cfg.crash_sigma);
  setVal('crash_lock', cfg.crash_lock);
  setVal('crash_alloc', cfg.crash_alloc != null ? cfg.crash_alloc : 100);
  setVal('min_hold', cfg.min_hold);
  setVal('min_gap', cfg.min_gap);
  setVal('stop_loss', cfg.stop_loss);
  setVal('exception_gap', cfg.exception_gap);
  setVal('confirm_days', cfg.confirm_days);
  setVal('commission', cfg.commission != null ? (cfg.commission * 100).toFixed(3) : null);
  setVal('slippage', cfg.slippage != null ? (cfg.slippage * 100).toFixed(2) : null);
}

document.addEventListener('DOMContentLoaded', () => {
  initETFChips();
  bindEvents();
  bindTradeSort();
  // 默认结束日期为今天
  document.getElementById('end_date').value = formatDate(new Date());
  runBacktest();
});
