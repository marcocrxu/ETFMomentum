/* leaderboard.js — 打榜页: 过滤 + 跑榜 + 自然语言提交 + 渲染 (固定窗口 2016-01-01~2026-07-31) */

let currentController = null;
let pollTimer = null;
let lastData = null;

function showError(msg) {
  const b = document.getElementById('error-banner');
  b.textContent = msg;
  b.style.display = 'block';
}

function setLoading(show) {
  document.getElementById('loading').style.display = show ? 'flex' : 'none';
  document.getElementById('btn-run').disabled = show;
}

function getToken() {
  return document.getElementById('lb-token').value.trim();
}

async function runLeaderboard() {
  const banner = document.getElementById('error-banner');
  banner.style.display = 'none';
  if (!getToken()) { showError('请先填写提交 token'); return; }
  if (currentController) currentController.abort();
  currentController = new AbortController();
  setLoading(true);
  try {
    const resp = await fetch('/api/leaderboard/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        token: getToken(),
        strategy: document.getElementById('lb-strategy').value,
        metric: document.getElementById('lb-metric').value,
        universe: document.getElementById('lb-universe').value,
      }),
      signal: currentController.signal
    });
    const json = await resp.json();
    if (!json.success) {
      showError(json.error || '打榜失败');
      return;
    }
    lastData = json.data;
    render();
  } catch (e) {
    if (e.name !== 'AbortError') showError('请求失败: ' + e.message);
  } finally {
    setLoading(false);
    currentController = null;
  }
}

async function submitCustom() {
  const desc = document.getElementById('nl-desc').value.trim();
  const msg = document.getElementById('nl-msg');
  if (!getToken()) { showError('请先填写提交 token'); return; }
  if (desc.length < 10) { msg.textContent = '策略描述太短 (至少10字)'; return; }
  const btn = document.getElementById('btn-submit');
  btn.disabled = true;
  msg.textContent = '提交中...';
  try {
    const resp = await fetch('/api/leaderboard/custom', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: getToken(), description: desc })
    });
    const json = await resp.json();
    if (!json.success) {
      msg.textContent = '提交失败: ' + (json.error || '');
      return;
    }
    msg.textContent = '已入队 #' + json.data.id + ', claude+kimi 生成中 (约 2~5 分钟)...';
    document.getElementById('nl-desc').value = '';
    refreshQueue();
  } catch (e) {
    msg.textContent = '请求失败: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function refreshQueue() {
  try {
    const resp = await fetch('/api/leaderboard/custom/status');
    const json = await resp.json();
    if (!json.success) return;
    const items = json.data || [];
    const box = document.getElementById('queue-list');
    if (items.length === 0) {
      box.innerHTML = '';
      return;
    }
    box.innerHTML = items.map(it => {
      let statusCls = 'status-pending';
      if (it.status === 'running') statusCls = 'status-running';
      if (it.status === 'done') statusCls = 'status-done';
      if (it.status === 'failed') statusCls = 'status-failed';
      let detail = it.status;
      if (it.status === 'done' && it.result) {
        const r = it.result;
        detail += ` · total ${r.total_ret >= 0 ? '+' : ''}${r.total_ret}% · mdd ${r.mdd}% · cal ${r.calmar}`;
      }
      if (it.status === 'failed' && it.error) detail += ' · ' + it.error;
      const codeLink = (it.status === 'done' && it.file)
        ? ` <a class="open-link" href="/lb/code/${it.id}?token=${encodeURIComponent(getToken())}" target="_blank">代码</a>` : '';
      return `<div class="queue-item">
        <span class="${statusCls}">[${detail}]</span> #${it.id} ${it.created} · ${it.description.slice(0, 50)}${codeLink}
      </div>`;
    }).join('');
    // 有未完成项时 10s 后轮询
    const pending = items.some(it => it.status === 'pending' || it.status === 'running');
    if (pending) {
      if (!pollTimer) pollTimer = setInterval(refreshQueue, 10000);
    } else if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  } catch (e) { /* 静默 */ }
}

function mergeCustom(customs) {
  if (!lastData) return null;
  const done = customs.filter(it => it.status === 'done' && it.result);
  const rows = [...(lastData.rows || [])];
  for (const it of done) {
    rows.push({ ...it.result, rank: 0 });
  }
  const metric = document.getElementById('lb-metric').value;
  rows.sort((a, b) => {
    const av = a[metric] == null ? -1e9 : a[metric];
    const bv = b[metric] == null ? -1e9 : b[metric];
    return bv - av;
  });
  rows.forEach((r, i) => { r.rank = i + 1; });
  return rows;
}

function render() {
  const meta = document.getElementById('lb-meta');
  meta.style.display = 'block';
  meta.textContent = `固定窗口 ${lastData.window} · ${lastData.rows.length} 个网格组合 · 耗时 ${lastData.elapsed}s · ${lastData.server_time}`;
  document.getElementById('table-card').style.display = 'block';
  fetch('/api/leaderboard/custom/status').then(r => r.json()).then(json => {
    const rows = mergeCustom(json.data || []);
    if (!rows) return;
    const tbody = document.getElementById('lb-body');
    const fmt = (v, suf = '', digits = 2) => v == null ? '-' :
      `${v >= 0 && suf !== '' ? '+' : ''}${Number(v).toFixed(digits)}${suf}`;
    tbody.innerHTML = rows.map(r => {
      const badge = r.strategy === 'yanyan'
        ? '<span class="strategy-badge badge-yanyan">yanyan</span>'
        : r.strategy === 'custom'
          ? '<span class="strategy-badge badge-custom">自定义</span>'
          : '<span class="strategy-badge badge-etf">主引擎</span>';
      const cls = v => v == null ? '' : (v >= 0 ? 'positive' : 'negative');
      let link = '';
      if (r.strategy === 'yanyan') link = `<a class="open-link" href="/yanyan" target="_blank">打开</a>`;
      else if (r.strategy === 'etf') link = `<a class="open-link" href="/" target="_blank">打开</a>`;
      else if (r.strategy === 'custom' && r.file)
        link = `<a class="open-link" href="/lb/code/${r.id}?token=${encodeURIComponent(getToken())}" target="_blank">代码</a>`;
      return `
      <tr>
        <td class="${r.rank <= 3 ? 'rank-top' : 'rank-num'}">${r.rank}</td>
        <td>${badge}</td>
        <td style="text-align:left;max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${r.params}">${r.params}</td>
        <td class="${cls(r.total_ret)}">${fmt(r.total_ret, '%', 1)}</td>
        <td class="${cls(r.ann_ret)}">${fmt(r.ann_ret, '%')}</td>
        <td class="negative">${fmt(r.mdd, '%')}</td>
        <td>${fmt(r.calmar, '', 2)}</td>
        <td>${fmt(r.sharpe, '', 2)}</td>
        <td>${r.trade_count != null ? r.trade_count : '-'}</td>
        <td>${link}</td>
      </tr>`;
    }).join('');
  });
}

function loadLast() {
  fetch('/api/leaderboard/last').then(r => r.json()).then(json => {
    if (json.success && json.data && json.data.rows) {
      lastData = json.data;
      render();
    }
  }).catch(() => {});
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btn-run').addEventListener('click', runLeaderboard);
  document.getElementById('btn-submit').addEventListener('click', submitCustom);
  loadLast();
  refreshQueue();
});
