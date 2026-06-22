/**
 * app.js — LLM Sim Bench GUI: 8-chart dashboard with agentic traffic visualization.
 *
 * Charts:
 *  1. Per-Request E2E Latency (scatter, colored by traffic type)
 *  2. Cumulative Throughput (line, area fill)
 *  3. Phase Breakdown (stacked bar: prefill vs decode)
 *  4. Latency CDF with p50/p95/p99 vertical markers
 *  5. Roofline Analysis (scatter: operational intensity vs throughput)
 *  6. K-Means Traffic Clustering (scatter: prompt len vs gen len, 4 clusters)
 *  7. Queue Depth Timeline (multi-line: pending, decode-waiting, active-decode)
 *  8. Traffic Mix (doughnut: 6 traffic classes)
 */

// ═══════════════════════════════════════════════════════════════════════════════
//  Traffic type color map (must match sim_worker.js)
// ═══════════════════════════════════════════════════════════════════════════════

const TRAFFIC_COLORS = {
    SimpleQuery:    '#22d3ee',
    ToolCall:       '#f59e0b',
    MultiStepCoT:   '#a78bfa',
    RecursiveLoop:  '#ef4444',
    MultiRAG:       '#10b981',
    CodeGen:        '#3b82f6',
};

const TRAFFIC_LABELS = {
    SimpleQuery:    'Simple Query',
    ToolCall:       'Tool Call',
    MultiStepCoT:   'Multi-Step CoT',
    RecursiveLoop:  'Recursive Loop',
    MultiRAG:       'Multi-RAG',
    CodeGen:        'Code Generation',
};

const CLUSTER_COLORS = ['#440154', '#31688e', '#35b779', '#fde725'];
const CLUSTER_LABELS = ['Compute-Bound', 'Memory-Bound', 'Balanced', 'Latency-Sensitive'];

// ═══════════════════════════════════════════════════════════════════════════════
//  Global State
// ═══════════════════════════════════════════════════════════════════════════════

let worker = null;
let charts = {};
let lastSnapshot = null;
let runHistory = {}; // Maps scheduler name to snap data

const els = {
    statusPill: document.getElementById('sim-status'),
    statusText: document.querySelector('#sim-status .status-text'),
    timeDisplay: document.getElementById('sim-time-display'),
    progressDisplay: document.getElementById('sim-progress-display'),
    btnInit: document.getElementById('btn-init'),
    btnPlay: document.getElementById('btn-play'),
    btnPause: document.getElementById('btn-pause'),
    btnStep: document.getElementById('btn-step'),
    btnFast: document.getElementById('btn-fast'),

    schedulerType: document.getElementById('scheduler-type'),
    trafficProfile: document.getElementById('traffic-profile'),
    trafficHint: document.getElementById('traffic-hint'),
    priorityFields: document.querySelectorAll('.priority-field'),
    batchFields: document.querySelectorAll('.batch-field'),
    stepBatch: document.getElementById('step-batch'),
    stepBatchVal: document.getElementById('step-batch-val'),

    valCompleted: document.getElementById('val-completed'),
    subCompleted: document.getElementById('sub-completed'),
    valThroughput: document.getElementById('val-throughput'),
    valAvgLat: document.getElementById('val-avg-lat'),
    valP95: document.getElementById('val-p95'),
    valCache: document.getElementById('val-cache'),
    subCache: document.getElementById('sub-cache'),
    valQueue: document.getElementById('val-queue'),
    subQueue: document.getElementById('sub-queue'),

    prefillDot: document.getElementById('prefill-dot'),
    prefillState: document.getElementById('prefill-state'),
    decodeDot: document.getElementById('decode-dot'),
    decodeState: document.getElementById('decode-state'),
    decodeBatchSize: document.getElementById('decode-batch-size'),
    eventsRemaining: document.getElementById('events-remaining'),
    decodeWaiting: document.getElementById('decode-waiting'),
    activeScheduler: document.getElementById('active-scheduler'),
    activeEviction: document.getElementById('active-eviction'),

    latencyCount: document.getElementById('latency-count'),
    throughputTotal: document.getElementById('throughput-total'),
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Initialization & Themes
// ═══════════════════════════════════════════════════════════════════════════════

const THEMES = {
    dark: {
        gridColor: 'rgba(42, 46, 55, 0.8)',
        tickColor: '#6b7280',
        axisTitleColor: '#9ca3af',
        tooltipBg: '#1a1d24',
        tooltipBorder: '#3a3f4b',
        tooltipTitle: '#f3f4f6',
        tooltipBody: '#9ca3af',
        textColor: '#9ca3af',
    },
    light: {
        gridColor: 'rgba(226, 232, 240, 0.8)',
        tickColor: '#64748b',
        axisTitleColor: '#475569',
        tooltipBg: '#ffffff',
        tooltipBorder: '#e2e8f0',
        tooltipTitle: '#0f172a',
        tooltipBody: '#475569',
        textColor: '#475569',
    }
};

let currentTheme = 'dark';

const TICK_FONT = { family: "'JetBrains Mono', monospace", size: 10 };
const AXIS_TITLE_FONT = { size: 11, weight: '600' };

function getTooltipStyle(t) {
    return {
        backgroundColor: THEMES[t].tooltipBg,
        borderColor: THEMES[t].tooltipBorder,
        borderWidth: 1,
        titleColor: THEMES[t].tooltipTitle,
        bodyColor: THEMES[t].tooltipBody,
        titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
        bodyFont: { family: "'JetBrains Mono', monospace", size: 10 },
    };
}

function init() {
    // Remove native title tooltips (we use CSS ::after instead)
    document.querySelectorAll('.info-tip').forEach(tip => {
        tip.dataset.tooltip = tip.getAttribute('title');
    });

    initCharts();
    initTheme(); // Initialize theme after charts are setup
    bindEvents();
    els.schedulerType.dispatchEvent(new Event('change'));
    els.trafficProfile.dispatchEvent(new Event('change'));
}

const engFormatter = (v) => {
    if (v === 0) return '0';
    const abs = Math.abs(v);
    if (abs >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (abs >= 1e3) return (v / 1e3).toFixed(1) + 'k';
    return v % 1 !== 0 ? v.toFixed(1) : v;
};

function initCharts() {
    Chart.defaults.color = THEMES[currentTheme].textColor;
    Chart.defaults.font.family = "'Times New Roman', 'Computer Modern', serif";
    Chart.defaults.font.size = 12;

    // ── 0. Tradeoff Chart ──
    charts.tradeoff = new Chart(document.getElementById('chart-tradeoff'), {
        type: 'bar',
        data: { labels: [], datasets: [
            { label: 'Throughput (tok/s)', data: [], yAxisID: 'y', backgroundColor: '#31688e', order: 2 },
            { label: 'P95 Latency (s)', data: [], yAxisID: 'y1', type: 'line', borderColor: '#fde725', backgroundColor: '#fde725', borderWidth: 2, pointRadius: 5, pointStyle: 'rectRot', fill: false, order: 1 }
        ]},
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: true, position: 'top', labels: { usePointStyle: true, font: { family: "'Times New Roman', serif" } } } },
            scales: {
                x: { grid: { color: THEMES[currentTheme].gridColor }, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor }, title: { display: true, text: 'Scheduler Policy', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT } },
                y: { type: 'linear', position: 'left', grid: { color: THEMES[currentTheme].gridColor }, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor }, title: { display: true, text: 'Throughput (tok/s)', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT } },
                y1: { type: 'logarithmic', min: 0.1, position: 'right', grid: { drawOnChartArea: false }, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor, callback: engFormatter }, title: { display: true, text: 'P95 Latency (seconds)', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT } }
            }
        }
    });

    // ── 1. Queue Evacuation (Time vs Cumulative Completed) ──
    charts.latency = new Chart(document.getElementById('chart-latency'), {
        type: 'line',
        data: { datasets: [] }, // Populated dynamically in updateCharts
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
                legend: { display: true, position: 'top', labels: { usePointStyle: true, font: { family: "'Times New Roman', serif" } } },
                tooltip: {
                    ...getTooltipStyle(currentTheme),
                    callbacks: {
                        label: (item) => `${item.dataset.label}: ${item.raw.y} completed at ${item.raw.x.toFixed(2)}s`
                    }
                }
            },
            scales: {
                x: { type: 'linear', grid: { color: THEMES[currentTheme].gridColor }, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor, callback: engFormatter }, title: { display: true, text: 'Simulation Time (seconds)', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT } },
                y: { grid: { color: THEMES[currentTheme].gridColor }, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor }, title: { display: true, text: 'Completed Requests', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT }, beginAtZero: true }
            }
        }
    });

    // ── 2. Throughput Line ──
    charts.throughput = new Chart(document.getElementById('chart-throughput'), {
        type: 'line',
        data: {
            datasets: [{
                label: 'Cumulative Tokens',
                data: [],
                borderColor: '#818cf8',
                backgroundColor: 'rgba(129, 140, 248, 0.08)',
                fill: true,
                stepped: true,
                pointRadius: 0,
                borderWidth: 2,
                parsing: false,
                normalized: true,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...getTooltipStyle(currentTheme),
                    callbacks: {
                        title: (items) => `After request #${items[0].raw.x}`,
                        label: (item) => `${item.raw.y.toLocaleString()} tokens`
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    grid: { color: THEMES[currentTheme].gridColor },
                    ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor, maxTicksLimit: 10, callback: v => Math.round(v) },
                    title: { display: true, text: 'Completed Requests', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT }
                },
                y: {
                    grid: { color: THEMES[currentTheme].gridColor },
                    ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor, callback: v => v >= 1000 ? (v/1000).toFixed(1)+'k' : v },
                    title: { display: true, text: 'Cumulative Tokens', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT }
                }
            }
        }
    });

    // ── 3. Phase Breakdown (Saliency) ──
    charts.phase = new Chart(document.getElementById('chart-phase'), {
        type: 'bar',
        data: { labels: ['% Total Time'], datasets: [
            { label: 'Prefill Compute', data: [0], backgroundColor: '#440154', borderWidth: 1 },
            { label: 'Decode Memory', data: [0], backgroundColor: '#31688e', borderWidth: 1 },
            { label: 'Overhead', data: [0], backgroundColor: '#35b779', borderWidth: 1 },
        ]},
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            indexAxis: 'y',
            plugins: { legend: { display: true, position: 'top', labels: { font: { family: "'Times New Roman', serif" } } } },
            scales: {
                x: { type: 'linear', stacked: true, min: 0, max: 100, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor, callback: v => v + '%' }, title: { display: true, text: 'Execution Time Breakdown', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT } },
                y: { type: 'category', stacked: true, grid: { display: false }, ticks: { display: true } }
            }
        }
    });

    // ── 4. Overlaid CDF ──
    charts.cdf = new Chart(document.getElementById('chart-cdf'), {
        type: 'line',
        data: { datasets: [] }, // Populated dynamically
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
                legend: { display: true, position: 'top', labels: { usePointStyle: true, font: { family: "'Times New Roman', serif" } } },
                tooltip: { ...getTooltipStyle(currentTheme), callbacks: { label: (item) => `${item.dataset.label}: ${(item.raw.y * 100).toFixed(1)}% at ${item.raw.x.toFixed(2)}s` } }
            },
            scales: {
                x: { type: 'linear', grid: { color: THEMES[currentTheme].gridColor }, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor, callback: engFormatter }, title: { display: true, text: 'Latency (seconds)', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT } },
                y: { grid: { color: THEMES[currentTheme].gridColor }, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor, callback: v => (v * 100).toFixed(0) + '%' }, title: { display: true, text: 'P(X ≤ x)', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT }, min: 0, max: 1 }
            }
        }
    });

    // ── 5. Roofline (colored by traffic type) ──
    const rooflineDatasets = [
        { label: 'Hardware Roof', data: [], type: 'line', borderColor: THEMES[currentTheme].tickColor, borderWidth: 2, pointRadius: 0, fill: false, parsing: false },
        ...Object.entries(TRAFFIC_LABELS).map(([type, label]) => ({
            label: label, data: [], type: 'scatter', backgroundColor: (TRAFFIC_COLORS[type] || '#888') + '99', borderColor: TRAFFIC_COLORS[type] || '#888', borderWidth: 1, pointRadius: 3.5, pointHoverRadius: 6, parsing: false
        }))
    ];

    charts.roofline = new Chart(document.getElementById('chart-roofline'), {
        type: 'scatter',
        data: { datasets: rooflineDatasets },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: 'nearest', intersect: true },
            plugins: {
                legend: { display: true, position: 'bottom', labels: { boxWidth: 8, font: { size: 9 }, padding: 6, usePointStyle: true, pointStyle: 'circle' } },
                tooltip: { ...getTooltipStyle(currentTheme), callbacks: { title: (items) => `Request #${items[0].raw.id}`, label: (item) => `OI: ${item.raw.x.toFixed(2)} FLOP/byte  |  ${item.raw.y.toFixed(2)} tok/s` } }
            },
            scales: {
                x: { type: 'logarithmic', grid: { color: THEMES[currentTheme].gridColor }, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor }, title: { display: true, text: 'Operational Intensity (FLOP/byte)', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT } },
                y: { type: 'logarithmic', grid: { color: THEMES[currentTheme].gridColor }, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor }, title: { display: true, text: 'Throughput (tok/s)', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT } }
            }
        }
    });

    // ── 6. Clustering ──
    charts.cluster = new Chart(document.getElementById('chart-cluster'), {
        type: 'scatter',
        data: {
            datasets: CLUSTER_LABELS.map((label, i) => ({
                label: label,
                data: [],
                backgroundColor: CLUSTER_COLORS[i] + '88',
                borderColor: CLUSTER_COLORS[i],
                borderWidth: 1,
                pointRadius: 4,
                pointHoverRadius: 6,
                parsing: false,
            }))
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: 'nearest', intersect: true },
            plugins: {
                legend: { display: true, position: 'bottom', labels: { boxWidth: 8, font: { size: 9 }, padding: 6, usePointStyle: true, pointStyle: 'circle' } },
                tooltip: {
                    ...getTooltipStyle(currentTheme),
                    callbacks: {
                        title: (items) => `Request #${items[0].raw.id}`,
                        label: (item) => `Prompt: ${item.raw.prompt || '?'} | Gen: ${item.raw.gen || '?'} | E2E: ${item.raw.e2e?.toFixed(2) || '?'}s`
                    }
                }
            },
            scales: {
                x: { grid: { color: THEMES[currentTheme].gridColor }, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor }, title: { display: true, text: 'Prompt Length (tokens)', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT } },
                y: { grid: { color: THEMES[currentTheme].gridColor }, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor }, title: { display: true, text: 'Generation Length (tokens)', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT } }
            }
        }
    });

    // ── 7. Queue Depth ──
    charts.queue = new Chart(document.getElementById('chart-queue'), {
        type: 'line',
        data: { datasets: [
            { label: 'Pending (prefill queue)', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.08)', fill: true, tension: 0.2, pointRadius: 0, borderWidth: 1.5, parsing: false, normalized: true },
            { label: 'Decode Waiting', data: [], borderColor: '#a78bfa', backgroundColor: 'rgba(167, 139, 250, 0.08)', fill: true, tension: 0.2, pointRadius: 0, borderWidth: 1.5, parsing: false, normalized: true },
            { label: 'Active Decode Batch', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.08)', fill: true, tension: 0.2, pointRadius: 0, borderWidth: 1.5, parsing: false, normalized: true },
        ]},
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: true, position: 'top', labels: { boxWidth: 10, font: { size: 9 } } },
                tooltip: getTooltipStyle(currentTheme)
            },
            scales: {
                x: { type: 'linear', grid: { color: THEMES[currentTheme].gridColor }, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor, maxTicksLimit: 12 }, title: { display: true, text: 'Simulation Time (s)', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT } },
                y: { grid: { color: THEMES[currentTheme].gridColor }, ticks: { font: TICK_FONT, color: THEMES[currentTheme].tickColor }, title: { display: true, text: 'Queue Depth', color: THEMES[currentTheme].axisTitleColor, font: AXIS_TITLE_FONT }, beginAtZero: true }
            }
        }
    });

    // ── 8. Traffic Mix Doughnut ──
    charts.traffic = new Chart(document.getElementById('chart-traffic'), {
        type: 'doughnut',
        data: {
            labels: Object.values(TRAFFIC_LABELS),
            datasets: [{
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: Object.values(TRAFFIC_COLORS).map(c => c + 'cc'),
                borderColor: Object.values(TRAFFIC_COLORS),
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            cutout: '55%',
            plugins: {
                legend: { display: true, position: 'right', labels: { boxWidth: 10, font: { size: 9 }, padding: 6 } },
                tooltip: {
                    ...getTooltipStyle(currentTheme),
                    callbacks: { label: (item) => `${item.label}: ${item.raw} requests (${(item.raw / (item.dataset.data.reduce((a,b)=>a+b,0)||1) * 100).toFixed(1)}%)` }
                }
            }
        }
    });
}

function bindEvents() {
    els.schedulerType.addEventListener('change', (e) => {
        const val = e.target.value;
        els.priorityFields.forEach(f => f.style.display = val === 'Priority' ? 'block' : 'none');
        els.batchFields.forEach(f => f.style.display = val === 'ContinuousBatching' ? 'block' : 'none');
    });

    els.trafficProfile.addEventListener('change', (e) => {
        const isAgentic = e.target.value === 'Agentic';
        els.trafficHint.style.display = isAgentic ? 'block' : 'none';
        // Hide manual prompt/gen fields when agentic (those are auto-generated per traffic type)
        document.querySelectorAll('.standard-workload-field').forEach(f => f.style.display = isAgentic ? 'none' : 'block');
    });

    els.stepBatch.addEventListener('input', (e) => {
        els.stepBatchVal.textContent = e.target.value;
        if (worker) worker.postMessage({ type: 'SET_STEP_BATCH', payload: { stepBatch: parseInt(e.target.value, 10) } });
    });

    els.btnInit.addEventListener('click', initSimulation);
    els.btnPlay.addEventListener('click', () => { setStatus('running'); worker.postMessage({ type: 'START', payload: { stepBatch: parseInt(els.stepBatch.value, 10) } }); });
    els.btnPause.addEventListener('click', () => { setStatus('paused'); worker.postMessage({ type: 'PAUSE' }); });
    els.btnStep.addEventListener('click', () => worker.postMessage({ type: 'STEP' }));
    els.btnFast.addEventListener('click', () => { setStatus('running'); worker.postMessage({ type: 'RUN_TO_COMPLETION' }); });

    const btnTheme = document.getElementById('btn-theme-toggle');
    if (btnTheme) {
        btnTheme.addEventListener('click', toggleTheme);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Simulation Management
// ═══════════════════════════════════════════════════════════════════════════════

function getConfig() {
    return {
        modelParamsB: parseFloat(document.getElementById('model-params').value),
        prefillHw: document.getElementById('prefill-hw').value,
        decodeHw: document.getElementById('decode-hw').value,
        kvCacheCapacity: parseInt(document.getElementById('kv-capacity').value, 10),
        kvTransferLatencyMs: parseFloat(document.getElementById('kv-transfer').value),
        evictionPolicy: document.getElementById('eviction-policy').value,
        scheduler: els.schedulerType.value,
        trafficProfile: els.trafficProfile.value,
        priorityPromptWeight: parseFloat(document.getElementById('priority-pw').value),
        priorityGenWeight: parseFloat(document.getElementById('priority-gw').value),
        maxBatchSize: parseInt(document.getElementById('max-batch').value, 10),
    };
}

function getWorkload() {
    return {
        numRequests: parseInt(document.getElementById('num-requests').value, 10),
        promptMin: parseInt(document.getElementById('prompt-min').value, 10),
        promptMax: parseInt(document.getElementById('prompt-max').value, 10),
        genMin: parseInt(document.getElementById('gen-min').value, 10),
        genMax: parseInt(document.getElementById('gen-max').value, 10),
        arrivalRate: parseFloat(document.getElementById('arrival-rate').value),
        seed: parseInt(document.getElementById('seed').value, 10),
    };
}

function initSimulation() {
    if (worker) worker.terminate();
    worker = new Worker('sim_worker.js');
    worker.onmessage = handleWorkerMessage;
    worker.onerror = (err) => {
        console.error("Worker error:", err.message, "at", err.filename, ":", err.lineno);
    };

    // Clear all charts
    for (const key of Object.keys(charts)) {
        const c = charts[key];
        if (c.config.type === 'doughnut') {
            c.data.datasets[0].data = [0, 0, 0, 0, 0, 0];
        } else {
            if (c.data.labels) c.data.labels = [];
            c.data.datasets.forEach(ds => ds.data = []);
        }
        c.update('none');
    }

    if (els.progressDisplay) {
        els.progressDisplay.textContent = `Completed: 0 / ${getWorkload().numRequests}`;
    }

    setStatus('idle');
    worker.postMessage({ type: 'INIT', payload: { config: getConfig(), workload: getWorkload() } });
}

function handleWorkerMessage(e) {
    const { type, payload } = e.data;
    lastSnapshot = payload;
    updateDashboard(payload);
    if (type === 'READY') {
        els.btnPlay.disabled = false; els.btnStep.disabled = false; els.btnFast.disabled = false;
    } else if (type === 'DONE') {
        setStatus('done');
        runHistory[payload.scheduler || 'FCFS'] = payload;
        updateCharts(payload, true);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  UI Updates
// ═══════════════════════════════════════════════════════════════════════════════

function setStatus(state) {
    els.statusPill.className = 'status-pill status-' + state;
    els.statusText.textContent = state.charAt(0).toUpperCase() + state.slice(1);
    const isRunning = state === 'running', isDone = state === 'done';
    els.btnPlay.disabled = isRunning || isDone;
    els.btnPause.disabled = !isRunning;
    els.btnStep.disabled = isRunning || isDone;
    els.btnFast.disabled = isRunning || isDone;
}

function updateDashboard(snap) {
    els.timeDisplay.textContent = `t = ${snap.time.toFixed(4)}s`;

    if (els.progressDisplay) {
        els.progressDisplay.textContent = `Completed: ${snap.completed} / ${snap.total}`;
    }

    // KPIs
    if (els.valCompleted) els.valCompleted.textContent = `${snap.completed} / ${snap.total}`;
    if (els.subCompleted) els.subCompleted.textContent = `${snap.total > 0 ? (snap.completed / snap.total * 100).toFixed(1) : '0.0'}%`;
    if (els.valThroughput) els.valThroughput.textContent = snap.throughput.toFixed(1);
    if (els.valAvgLat) els.valAvgLat.textContent = snap.avgLatency.toFixed(2);
    if (els.valP95) els.valP95.textContent = snap.p95.toFixed(2);
    if (els.valCache) els.valCache.textContent = `${(snap.cacheHitRate * 100).toFixed(1)}%`;
    if (els.subCache) els.subCache.textContent = `${snap.cacheHits.toLocaleString()} hits / ${snap.cacheMisses.toLocaleString()} misses`;
    if (els.valQueue) els.valQueue.textContent = snap.queueDepth;
    if (els.subQueue) els.subQueue.textContent = `peak: ${snap.peakQueueDepth}`;

    // Engine status
    updateEngineIndicator(els.prefillDot, els.prefillState, snap.prefillBusy);
    updateEngineIndicator(els.decodeDot, els.decodeState, snap.decodeBusy);
    els.decodeBatchSize.textContent = snap.decodeBatchSize || 0;
    els.eventsRemaining.textContent = snap.eventsRemaining;
    els.decodeWaiting.textContent = snap.decodeWaiting;
    els.activeScheduler.textContent = snap.scheduler || 'FCFS';
    els.activeEviction.textContent = snap.evictionPolicy || 'None';

    // Badges
    els.latencyCount.textContent = `${snap.completed} requests`;
    els.throughputTotal.textContent = `${snap.totalTokens.toLocaleString()} tokens`;

    updateCharts(snap);
}

function updateEngineIndicator(dot, text, isBusy) {
    if (isBusy) { dot.classList.add('active'); text.textContent = 'Active'; }
    else { dot.classList.remove('active'); text.textContent = 'Idle'; }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Chart Updates
// ═══════════════════════════════════════════════════════════════════════════════

let lastChartUpdate = 0;

function updateCharts(snap, isFinal = false) {
    const now = performance.now();
    if (now - lastChartUpdate < 300 && snap.completed < snap.total) return;
    lastChartUpdate = now;

    if (!snap) return;
    const lats = snap.latencies || [];
    const keys = Object.keys(TRAFFIC_LABELS);

    // ── 0. Tradeoff Chart ──
    const activeSched = snap.scheduler || 'FCFS';
    const tempHistory = { ...runHistory };
    tempHistory[activeSched] = snap;

    const schedulers = Object.keys(tempHistory);
    charts.tradeoff.data.labels = schedulers;
    charts.tradeoff.data.datasets[0].data = schedulers.map(s => tempHistory[s].throughput > 0 ? tempHistory[s].throughput : null);
    charts.tradeoff.data.datasets[1].data = schedulers.map(s => tempHistory[s].p95 > 0 ? tempHistory[s].p95 : null);
    charts.tradeoff.update('none');

    if (lats.length > 0) {
        // ── 1. Queue Evacuation Curve ──
        const evacDatasets = [];
        const runs = isFinal ? Object.keys(runHistory) : [snap.scheduler || 'FCFS'];
        const colors = ['#4f46e5', '#10b981', '#f59e0b', '#ec4899'];
        
        runs.forEach((sName, i) => {
            const sSnap = isFinal ? runHistory[sName] : snap;
            if (!sSnap.latencies) return;
            const pts = [];
            const sortedLats = [...sSnap.latencies].sort((a,b) => a.completion - b.completion);
            const step = Math.max(1, Math.floor(sortedLats.length / 300));
            sortedLats.forEach((r, idx) => {
                if (idx % step === 0 || idx === sortedLats.length - 1) {
                    pts.push({ x: r.completion, y: idx + 1 });
                }
            });
            evacDatasets.push({
                label: sName,
                data: pts,
                borderColor: colors[i % colors.length],
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.1
            });
        });
        charts.latency.data.datasets = evacDatasets;
        charts.latency.update('none');

        // ── 2. Throughput ──
        const cumTokens = [];
        let totalTok = 0;
        const sortedByCompletion = [...lats].sort((a, b) => a.completion - b.completion);
        sortedByCompletion.forEach((r, idx) => {
            totalTok += r.gen;
            cumTokens.push({ x: idx + 1, y: totalTok });
        });
        charts.throughput.data.datasets[0].data = cumTokens;
        charts.throughput.update('none');

        // ── 3. Phase Breakdown (Saliency) ──
        let totalPrefill = 0, totalDecode = 0, totalE2e = 0;
        for (const r of lats) {
            totalPrefill += r.prefill;
            totalDecode += r.decode;
            totalE2e += r.e2e;
        }
        if (totalE2e > 0) {
            charts.phase.data.labels = ['% Total Time'];
            const pPct = (totalPrefill / totalE2e) * 100;
            const dPct = (totalDecode / totalE2e) * 100;
            const oPct = Math.max(0, 100 - pPct - dPct);
            charts.phase.data.datasets[0].data = [pPct];
            charts.phase.data.datasets[1].data = [dPct];
            charts.phase.data.datasets[2].data = [oPct];
        }
        charts.phase.update('none');

        // ── 4. Overlaid CDF ──
        const cdfDatasets = [];
        const runKeys = isFinal ? Object.keys(runHistory) : [snap.scheduler || 'FCFS'];
        const cdfColors = ['#10b981', '#3b82f6', '#8b5cf6', '#ec4899'];
        
        runKeys.forEach((sName, i) => {
            const sSnap = isFinal ? runHistory[sName] : snap;
            if (!sSnap.latencies) return;
            const sorted = [...sSnap.latencies].map(l => l.e2e).sort((a, b) => a - b);
            const step = Math.max(1, Math.floor(sorted.length / 200));
            const pts = [];
            sorted.forEach((v, idx) => {
                if (idx % step === 0 || idx === sorted.length - 1) {
                    pts.push({ x: v, y: (idx + 1) / sorted.length });
                }
            });
            const p50 = sorted[Math.floor(sorted.length * 0.50)] || 0;
            const p95 = sorted[Math.floor(sorted.length * 0.95)] || 0;
            
            cdfDatasets.push({
                label: `${sName} CDF`,
                data: pts,
                borderColor: cdfColors[i % cdfColors.length],
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                borderDash: i === 0 ? [] : [5, 5],
                tension: 0.1
            });
            
            cdfDatasets.push({
                label: `${sName} P50`,
                data: [{x: p50, y: 0}, {x: p50, y: 1}],
                borderColor: cdfColors[i % cdfColors.length],
                borderWidth: 1,
                pointRadius: 0,
                fill: false,
                borderDash: [2, 2]
            });
            cdfDatasets.push({
                label: `${sName} P95`,
                data: [{x: p95, y: 0}, {x: p95, y: 1}],
                borderColor: cdfColors[i % cdfColors.length],
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                borderDash: [4, 4]
            });
        });
        charts.cdf.data.datasets = cdfDatasets;
        charts.cdf.update('none');

        // ── 5. Roofline (colored by traffic type) ──
        const computeTput = snap.peakComputeTput || 100;
        const memTput = snap.peakMemBwTput || 10;
        const ridgeX = computeTput / memTput;

        charts.roofline.data.datasets[0].data = [
            { x: 0.01, y: 0.01 * memTput },
            { x: ridgeX, y: computeTput },
            { x: 10000, y: computeTput }
        ];

        // Clear traffic datasets (index 1 onwards)
        for (let i = 1; i < charts.roofline.data.datasets.length; i++) {
            charts.roofline.data.datasets[i].data = [];
        }

        for (const r of lats) {
            const tput = r.decode > 0 ? r.gen / r.decode : 0;
            const idx = keys.indexOf(r.trafficType);
            if (idx !== -1) {
                charts.roofline.data.datasets[idx + 1].data.push({
                    x: Math.max(0.01, r.operationalIntensity),
                    y: Math.max(0.01, tput),
                    id: r.id
                });
            }
        }
        charts.roofline.update('none');

        // ── 6. Clustering ──
        charts.cluster.data.datasets.forEach(ds => ds.data = []);
        const ca = snap.clusterAssignments;
        if (ca && ca.length === lats.length) {
            for (let i = 0; i < lats.length; i++) {
                const c = ca[i];
                if (c >= 0 && c < 4) {
                    charts.cluster.data.datasets[c].data.push({
                        x: lats[i].prompt,
                        y: lats[i].gen,
                        id: lats[i].id,
                        prompt: lats[i].prompt,
                        gen: lats[i].gen,
                        e2e: lats[i].e2e
                    });
                }
            }
            const clusterInfo = document.getElementById('cluster-info');
            if (clusterInfo) {
                const counts = charts.cluster.data.datasets.map(d => d.data.length);
                clusterInfo.textContent = counts.map((c, i) => `C${i+1}:${c}`).join(' ');
            }
        }
        charts.cluster.update('none');
    }

    // ── 7. Queue Depth ──
    const qt = snap.queueTimeline;
    if (qt && qt.length > 0) {
        const step = Math.max(1, Math.floor(qt.length / 200));
        const sampled = qt.filter((_, i) => i % step === 0);
        charts.queue.data.datasets[0].data = sampled.map(q => ({ x: q.time, y: q.pending }));
        charts.queue.data.datasets[1].data = sampled.map(q => ({ x: q.time, y: q.decodeWaiting }));
        charts.queue.data.datasets[2].data = sampled.map(q => ({ x: q.time, y: q.activeDecode }));
    }
    charts.queue.update('none');

    // ── 8. Traffic Mix ──
    const mix = snap.trafficMix;
    if (mix) {
        charts.traffic.data.datasets[0].data = keys.map(k => mix[k] || 0);
    }
    charts.traffic.update('none');
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Theme Toggling and Sync
// ═══════════════════════════════════════════════════════════════════════════════

function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    const isLight = savedTheme === 'light';
    if (isLight) {
        document.body.classList.add('light-mode');
        currentTheme = 'light';
    } else {
        document.body.classList.remove('light-mode');
        currentTheme = 'dark';
    }
    updateChartTheme(isLight);
}

function toggleTheme() {
    const isLight = document.body.classList.toggle('light-mode');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
    currentTheme = isLight ? 'light' : 'dark';
    updateChartTheme(isLight);
}

function updateChartTheme(isLight) {
    const theme = isLight ? THEMES.light : THEMES.dark;
    
    // Update defaults
    Chart.defaults.color = theme.textColor;
    
    for (const key of Object.keys(charts)) {
        const chart = charts[key];
        
        // Update scale elements
        if (chart.options.scales) {
            for (const axisId of Object.keys(chart.options.scales)) {
                const scale = chart.options.scales[axisId];
                if (scale.grid) {
                    scale.grid.color = theme.gridColor;
                }
                if (scale.ticks) {
                    scale.ticks.color = theme.tickColor;
                }
                if (scale.title) {
                    scale.title.color = theme.axisTitleColor;
                }
            }
        }
        
        // Update tooltips
        if (chart.options.plugins && chart.options.plugins.tooltip) {
            const tooltip = chart.options.plugins.tooltip;
            tooltip.backgroundColor = theme.tooltipBg;
            tooltip.borderColor = theme.tooltipBorder;
            tooltip.titleColor = theme.tooltipTitle;
            tooltip.bodyColor = theme.tooltipBody;
        }
        
        chart.update('none');
    }
}

document.addEventListener('DOMContentLoaded', init);
