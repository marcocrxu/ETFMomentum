/* charts.js - ECharts 图表渲染 */

const CHART_COLORS = {
  accent: '#e94560',
  accent2: '#ff6b6b',
  muted: '#8b92a8',
  green: '#4caf50',
  red: '#f44336',
  grid: '#2a3142',
  text: '#e8eaf0',
  etfPalette: ['#f44336', '#ff9800', '#ffeb3b', '#4caf50', '#00bcd4', '#2196f3', '#9c27b0', '#e91e63', '#795548', '#607d8b', '#3f51b5', '#009688']
};

function getCommonOption() {
  return {
    backgroundColor: 'transparent',
    // 关闭动画与分段渲染: 快速滑动 dataZoom 时,
    // 动画插值和 progressive 分块绘制会产生瞬时的错误连线
    animation: false,
    progressive: 0,
    progressiveThreshold: 0,
    textStyle: { fontFamily: 'Microsoft YaHei, sans-serif' },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(20,24,36,.95)',
      borderColor: '#2a3142',
      textStyle: { color: '#e8eaf0', fontSize: 12 },
      axisPointer: { lineStyle: { color: '#8b92a8' } }
    },
    grid: { left: 48, right: 24, top: 32, bottom: 56 },
    xAxis: {
      type: 'category',
      axisLine: { lineStyle: { color: '#2a3142' } },
      axisLabel: { color: '#8b92a8' },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: 'rgba(42,49,66,.6)' } },
      axisLabel: { color: '#8b92a8' }
    },
    dataZoom: [
      { type: 'inside', start: 0, end: 100 },
      { type: 'slider', start: 0, end: 100, height: 18, bottom: 12, borderColor: '#2a3142', fillerColor: 'rgba(233,69,96,.2)', handleStyle: { color: '#e94560' }, textStyle: { color: '#8b92a8' } }
    ]
  };
}

function initChart(domId) {
  const dom = document.getElementById(domId);
  if (!dom) return null;
  const chart = echarts.init(dom, null, { renderer: 'canvas' });
  return chart;
}

function renderNavChart(chart, data) {
  const dates = data.dates;
  const nav = data.nav;
  const benchmark = data.benchmark;
  const etfNavs = data.etf_navs || {};
  const decisions = data.decisions || [];

  const series = [
    {
      name: '策略净值',
      type: 'line',
      data: nav,
      smooth: false,
      symbol: 'none',
      lineStyle: { color: CHART_COLORS.accent, width: 2 },
      itemStyle: { color: CHART_COLORS.accent },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(233,69,96,.25)' },
          { offset: 1, color: 'rgba(233,69,96,.02)' }
        ])
      }
    },
    {
      name: '等权基准',
      type: 'line',
      data: benchmark,
      smooth: false,
      symbol: 'none',
      lineStyle: { color: CHART_COLORS.muted, width: 1.5, type: 'dashed' },
      itemStyle: { color: CHART_COLORS.muted }
    }
  ];

  // 各 ETF 净值线 (默认不选中，可通过 legend 切换)
  let colorIdx = 0;
  for (const [name, vals] of Object.entries(etfNavs)) {
    series.push({
      name: name,
      type: 'line',
      data: vals,
      smooth: false,
      symbol: 'none',
      lineStyle: { color: CHART_COLORS.etfPalette[colorIdx % CHART_COLORS.etfPalette.length], width: 1 },
      itemStyle: { color: CHART_COLORS.etfPalette[colorIdx % CHART_COLORS.etfPalette.length] },
      emphasis: { focus: 'series' }
    });
    colorIdx++;
  }

  // 交易标记点
  const markData = decisions
    .filter(d => d.action && (d.action.includes('→') || d.action.includes('买入') || d.action.includes('卖出')))
    .map(d => {
      const idx = dates.indexOf(d.date);
      return {
        coord: [idx, idx >= 0 ? nav[idx] : null],
        value: d.action,
        itemStyle: { color: d.action.includes('买入') ? CHART_COLORS.green : (d.action.includes('卖出') ? CHART_COLORS.red : CHART_COLORS.accent2) }
      };
    })
    .filter(d => d.coord[0] >= 0 && d.coord[1] != null);

  if (markData.length > 0) {
    series.push({
      name: '交易标记',
      type: 'scatter',
      data: markData.map(d => [d.coord[0], d.coord[1], d.value]),
      symbolSize: 7,
      itemStyle: { color: CHART_COLORS.accent2, borderColor: '#fff', borderWidth: 1 },
      tooltip: {
        formatter: function(p) {
          return p.data[2] + '<br>' + p.name;
        }
      }
    });
  }

  const option = {
    ...getCommonOption(),
    legend: {
      data: series.map(s => s.name),
      top: 4,
      textStyle: { color: '#8b92a8', fontSize: 11 },
      selected: Object.fromEntries(
        series
          .filter(s => !['策略净值', '等权基准', '交易标记'].includes(s.name))
          .map(s => [s.name, false])
      )
    },
    tooltip: {
      ...getCommonOption().tooltip,
      formatter: function(params) {
        let html = `<div style="font-weight:600;margin-bottom:4px">${params[0].axisValue}</div>`;
        params.forEach(p => {
          if (p.seriesName === '交易标记') return;
          const v = typeof p.value === 'number' ? p.value.toFixed(4) : p.value;
          html += `<div style="display:flex;align-items:center;gap:6px">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${p.color}"></span>
            <span>${p.seriesName}: ${v}</span>
          </div>`;
        });
        return html;
      }
    },
    xAxis: { ...getCommonOption().xAxis, data: dates },
    yAxis: { ...getCommonOption().yAxis, axisLabel: { formatter: '{value}' } },
    series: series
  };
  chart.setOption(option, true);
}

function renderDrawdownChart(chart, data) {
  const dates = data.dates;
  const dd = data.drawdown;

  const option = {
    ...getCommonOption(),
    grid: { left: 48, right: 24, top: 24, bottom: 56 },
    tooltip: {
      ...getCommonOption().tooltip,
      formatter: function(p) {
        return `${p[0].axisValue}<br/>回撤: ${p[0].value}%`;
      }
    },
    xAxis: { ...getCommonOption().xAxis, data: dates },
    yAxis: { ...getCommonOption().yAxis, max: 0, axisLabel: { formatter: '{value}%' } },
    series: [{
      name: '回撤',
      type: 'line',
      data: dd,
      smooth: true,
      symbol: 'none',
      lineStyle: { color: CHART_COLORS.red, width: 1.5 },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(244,67,54,.5)' },
          { offset: 1, color: 'rgba(244,67,54,.05)' }
        ])
      }
    }]
  };
  chart.setOption(option, true);
}

function renderYearlyChart(chart, data) {
  const yearly = data.yearly || {};
  const years = Object.keys(yearly).sort();
  const values = years.map(y => yearly[y]);

  const option = {
    ...getCommonOption(),
    grid: { left: 48, right: 24, top: 24, bottom: 36 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(20,24,36,.95)',
      borderColor: '#2a3142',
      textStyle: { color: '#e8eaf0' },
      formatter: function(p) {
        const v = p[0].value;
        return `${p[0].axisValue}<br/>收益: ${v >= 0 ? '+' : ''}${v}%`;
      }
    },
    xAxis: { ...getCommonOption().xAxis, data: years, axisLabel: { rotate: 0 } },
    yAxis: { ...getCommonOption().yAxis, axisLabel: { formatter: '{value}%' } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 8, borderColor: '#2a3142', fillerColor: 'rgba(233,69,96,.2)' }],
    series: [{
      name: '年度收益',
      type: 'bar',
      data: values,
      itemStyle: {
        color: function(p) {
          return p.value >= 0 ? CHART_COLORS.green : CHART_COLORS.red;
        },
        borderRadius: [3, 3, 0, 0]
      },
      label: {
        show: true,
        position: 'top',
        formatter: function(p) {
          return (p.value >= 0 ? '+' : '') + p.value + '%';
        },
        color: '#e8eaf0',
        fontSize: 10
      }
    }]
  };
  chart.setOption(option, true);
}

function renderPositionChart(chart, data) {
  const dates = data.dates;
  const posSeries = data.position_series || [];
  const labels = data.position_labels || ['空仓'];

  // 对日期降采样，避免热力图点过多
  const step = Math.max(1, Math.floor(dates.length / 200));
  const sampledDates = [];
  const sampledPos = [];
  for (let i = 0; i < dates.length; i += step) {
    sampledDates.push(dates[i]);
    sampledPos.push(posSeries[i]);
  }

  // 每个持仓行一种颜色: data value = y 索引(p+1), visualMap 按索引取色
  const heatmapData = sampledPos.map((p, i) => [i, p + 1, p + 1]);

  // 构造色板: idx0=空仓(灰), 组合行(含'+', 部分抄底)=橙, 单标的=palette 平铺
  const posColors = ['#2a3142'];
  let palIdx = 0;
  for (let li = 1; li < labels.length; li++) {
    if (labels[li].indexOf('+') >= 0) {
      posColors.push('#e67e22');
    } else {
      posColors.push(CHART_COLORS.etfPalette[palIdx % CHART_COLORS.etfPalette.length]);
      palIdx++;
    }
  }

  const option = {
    backgroundColor: 'transparent',
    animation: false,
    progressive: 0,
    progressiveThreshold: 0,
    tooltip: {
      position: 'top',
      backgroundColor: 'rgba(20,24,36,.95)',
      borderColor: '#2a3142',
      textStyle: { color: '#e8eaf0' },
      formatter: function(p) {
        const lbl = labels[p.data[1]] || '未知';
        // 组合行(部分抄底)拆分展示, 如 "黄金(50%)+创业(50%)"
        const detail = lbl.indexOf('+') >= 0
          ? lbl.split('+').map(s => `<div style="color:#e67e22">${s}</div>`).join('')
          : `<div style="color:#8b92a8">${lbl}</div>`;
        return `<div style="font-weight:600">${sampledDates[p.data[0]]}</div><br/>持仓: ${detail}`;
      }
    },
    grid: { left: 72, right: 24, top: 12, bottom: 56 },
    xAxis: {
      type: 'category',
      data: sampledDates,
      axisLine: { lineStyle: { color: '#2a3142' } },
      axisLabel: { color: '#8b92a8', interval: Math.floor(sampledDates.length / 6) },
      axisTick: { show: false },
      splitArea: { show: false }
    },
    yAxis: {
      type: 'category',
      data: labels,
      axisLine: { lineStyle: { color: '#2a3142' } },
      axisLabel: { color: '#8b92a8' },
      splitArea: { show: false }
    },
    dataZoom: [
      { type: 'inside', start: 0, end: 100, xAxisIndex: 0 },
      { type: 'slider', start: 0, end: 100, height: 14, bottom: 12, borderColor: '#2a3142', fillerColor: 'rgba(233,69,96,.2)' }
    ],
    visualMap: {
      show: false,
      min: 0,
      max: labels.length - 1,
      calculable: false,
      inRange: { color: posColors }
    },
    series: [{
      name: '持仓',
      type: 'heatmap',
      data: heatmapData,
      label: { show: false },
      itemStyle: {
        borderColor: '#141824',
        borderWidth: 1
      },
      emphasis: {
        itemStyle: { shadowBlur: 6, shadowColor: 'rgba(233,69,96,.5)' }
      }
    }]
  };
  chart.setOption(option, true);
}

const CHART_GROUP = 'etf-rotation-charts';

// 联动/独立缩放切换：connect 后任一图的 dataZoom 会同步到组内所有图，
// 同时十字光标也会联动；disconnect 后各图恢复独立缩放。
function setChartsLinked(charts, linked) {
  charts.forEach(c => { if (c) c.group = CHART_GROUP; });
  if (linked) {
    echarts.connect(CHART_GROUP);
  } else {
    echarts.disconnect(CHART_GROUP);
  }
}

window.ChartRenderers = {
  initChart,
  renderNavChart,
  renderDrawdownChart,
  renderYearlyChart,
  renderPositionChart,
  setChartsLinked
};
