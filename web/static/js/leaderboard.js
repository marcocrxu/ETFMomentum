/* leaderboard.js — 打榜页: 过滤 + 跑榜 + 渲染 */

let currentController = null;

function showError(msg) {
  const b = document.getElementById('error-banner');
  b.textContent = msg;
  b.style.display = 'block';
}

function setLoading(show) {
  document.getElementById('loading').style.display = show ? 'flex' : 'none';
  document.getElementById('btn-run').disabled = show;
}

async function runLeaderboard() {
  const banner = document.getElementById('error-banner');
  banner.style.display = 'none';
  if (currentController) currentController.abort();
  currentController = new AbortController();
  setLoading(true);
  try {
    const resp = await fetch('/api/leaderboard/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        strategy: document.getElementById('lb-strategy').value,
        metric: document.getElementById('lb-metric').value,
        universe: document.getElementById('lb-universe').value,
        start_date: document.getElementById('start_date').value,
        end_date: document.getElementById('end_date').value,
      }),
      signal: currentController.signal
    });
    const json = await resp.json();
    if (!json.success) {
      showError(json.error || '打榜失败');
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
  const meta = document.getElementById('lb-meta');
  meta.style.display = 'block';
  meta.textContent = `窗口 ${data.window} · ${data.rows.length} 个组合 · 耗时 ${data.elapsed}s · ${data.server_time}`;
  if (data.errors && data.errors.length) {
    meta.textContent += ` · 失败 ${data.errors.length} 个`;
  }
  document.getElementById('table-card').style.display = 'block';
  const tbody = document.getElementById('lb-body');
  const fmt = (v, suf = '', digits = 2) => v == null ? '-' :
    `${v >= 0 && suf !== '' ? '+' : ''}${Number(v).toFixed(digits)}${suf}`;
  tbody.innerHTML = data.rows.map(r => {
    const badge = r.strategy === 'yanyan'
      ? '<span class="strategy-badge badge-yanyan">yanyan</span>'
      : '<span class="strategy-badge badge-etf">主引擎</span>';
    const cls = v => v == null ? '' : (v >= 0 ? 'positive' : 'negative');
    return `
      <tr>
        <td class="${r.rank <= 3 ? 'rank-top' : 'rank-num'}">${r.rank}</td>
        <td>${badge}</td>
        <td style="text-align:left">${r.params}</td>
        <td class="${cls(r.total_ret)}">${fmt(r.total_ret, '%', 1)}</td>
        <td class="${cls(r.ann_ret)}">${fmt(r.ann_ret, '%')}</td>
        <td class="negative">${fmt(r.mdd, '%')}</td>
        <td>${fmt(r.calmar, '', 2)}</td>
        <td>${fmt(r.sharpe, '', 2)}</td>
        <td>${r.trade_count != null ? r.trade_count : '-'}</td>
      </tr>`;
  }).join('');
}

function loadLast() {
  fetch('/api/leaderboard/last').then(r => r.json()).then(json => {
    if (json.success && json.data && json.data.rows) {
      render(json.data);
      const strat = json.data.rows.length &&
        json.data.rows.every(x => x.strategy === 'yanyan') ? 'yanyan' :
        (json.data.rows.every(x => x.strategy === 'etf') ? 'etf' : 'all');
      document.getElementById('lb-strategy').value = strat;
    }
  }).catch(() => {});
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btn-run').addEventListener('click', runLeaderboard);
  loadLast();
});
