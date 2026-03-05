const fmt = (x) => x === null || x === undefined || x === '' ? '-' : x;
const esc = (x) => String(x)
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;');
const oneLine = (x) => String(fmt(x)).replace(/\s+/g, ' ').trim() || '-';
function truncCell(value, maxLen = 60, mono = false) {
  const raw = oneLine(value);
  const shown = raw.length > maxLen ? `${raw.slice(0, Math.max(1, maxLen - 1))}…` : raw;
  const cls = mono ? 'mono clip' : 'clip';
  return `<span class="${cls} copyable" title="${esc(raw)}" data-copy="${esc(raw)}">${esc(shown)}</span>`;
}
let copyToastTimer = null;
function showCopyToast(msg) {
  let el = document.getElementById('copyToast');
  if (!el) {
    el = document.createElement('div');
    el.id = 'copyToast';
    el.className = 'copy-toast';
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.add('show');
  if (copyToastTimer) clearTimeout(copyToastTimer);
  copyToastTimer = setTimeout(() => el.classList.remove('show'), 900);
}
async function copyText(text) {
  const val = String(text || '');
  if (!val) return;
  try {
    await navigator.clipboard.writeText(val);
  } catch (_) {
    const ta = document.createElement('textarea');
    ta.value = val;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  }
  showCopyToast('Copied full text');
}
function bindCopyables(containerSelector) {
  document.querySelectorAll(`${containerSelector} .copyable`).forEach(el => {
    el.addEventListener('click', () => copyText(el.getAttribute('data-copy') || ''));
  });
}
const laneOrder = ['queue', 'processing', 'private', 'complete', 'failed'];
let activeLane = 'processing';
let activeWorkerTab = 'gpu';
const chainState = {};
const batchState = { batch: '', plan: '', sort: 'started_desc' };
const laneState = { taskClass: '', task: '', worker: '', executor: '', batch: '', error: '', sort: 'task_asc' };
let latestStatus = null;
let visibleBatchIds = null;
let laneVisibleBatchIds = null;
let refreshInFlight = false;
let lastRefreshOkAt = null;
const stickyAlertStoreKey = 'orchStickyAlertsV1';
const alertSeenStoreKey = 'orchAlertSeenV1';
const dismissedStickyAlertStoreKey = 'orchDismissedStickyAlertsV1';
const dismissedAlertStoreKey = 'orchDismissedAlertsV1';
const trackedBatchesStoreKey = 'orchTrackedBatchesV1';

function isTrackableBatchId(batchId) {
  const bid = String(batchId || '').trim();
  if (!bid) return false;
  return bid.toLowerCase() !== 'system';
}

function fmtTs(ts) {
  if (!ts) return '-';
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return String(ts);
  return d.toLocaleString();
}

function loadTrackedBatches() {
  try {
    const raw = localStorage.getItem(trackedBatchesStoreKey);
    const parsed = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.map(x => String(x || '').trim()).filter(isTrackableBatchId));
  } catch (_) {
    return new Set();
  }
}

function saveTrackedBatches(set) {
  try {
    const cleaned = [...(set || new Set())]
      .map(x => String(x || '').trim())
      .filter(isTrackableBatchId);
    localStorage.setItem(trackedBatchesStoreKey, JSON.stringify(cleaned));
  } catch (_) {}
}

function addTrackedBatch(batchId) {
  if (!isTrackableBatchId(batchId)) return;
  const tracked = loadTrackedBatches();
  tracked.add(String(batchId).trim());
  saveTrackedBatches(tracked);
}

function removeTrackedBatch(batchId) {
  if (!batchId) return;
  const tracked = loadTrackedBatches();
  tracked.delete(batchId);
  saveTrackedBatches(tracked);
}

function setTrackedBatchChecked(batchId, checked) {
  if (!isTrackableBatchId(batchId)) return;
  const tracked = loadTrackedBatches();
  const id = String(batchId).trim();
  if (checked) tracked.add(id);
  else tracked.delete(id);
  saveTrackedBatches(tracked);
}

function loadStickyAlerts() {
  try {
    const raw = localStorage.getItem(stickyAlertStoreKey);
    const parsed = raw ? JSON.parse(raw) : {};
    return (parsed && typeof parsed === 'object') ? parsed : {};
  } catch (_) {
    return {};
  }
}

function saveStickyAlerts(map) {
  try {
    localStorage.setItem(stickyAlertStoreKey, JSON.stringify(map || {}));
  } catch (_) {}
}

function loadAlertSeenMap() {
  try {
    const raw = localStorage.getItem(alertSeenStoreKey);
    const parsed = raw ? JSON.parse(raw) : {};
    return (parsed && typeof parsed === 'object') ? parsed : {};
  } catch (_) {
    return {};
  }
}

function saveAlertSeenMap(map) {
  try {
    localStorage.setItem(alertSeenStoreKey, JSON.stringify(map || {}));
  } catch (_) {}
}

function loadDismissedStickyAlerts() {
  try {
    const raw = localStorage.getItem(dismissedStickyAlertStoreKey);
    const parsed = raw ? JSON.parse(raw) : {};
    return (parsed && typeof parsed === 'object') ? parsed : {};
  } catch (_) {
    return {};
  }
}

function saveDismissedStickyAlerts(map) {
  try {
    localStorage.setItem(dismissedStickyAlertStoreKey, JSON.stringify(map || {}));
  } catch (_) {}
}

function loadDismissedAlerts() {
  try {
    const raw = localStorage.getItem(dismissedAlertStoreKey);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (_) {
    return {};
  }
}

function saveDismissedAlerts(map) {
  try {
    localStorage.setItem(dismissedAlertStoreKey, JSON.stringify(map || {}));
  } catch (_) {}
}

function typeBadge(type) {
  return `<span class="pill ${type}">${type}</span>`;
}

function inferLlmTier(task) {
  const t = task || {};
  const modelText = [
    t.llm_model,
    t.model,
    t.command,
    t.name
  ].map(v => String(v || '').toLowerCase()).join(' ');
  if (/\b14b\b/.test(modelText)) return 'llm14b';
  if (/\b7b\b/.test(modelText)) return 'llm7b';
  return 'llm';
}

function taskType(clsOrTask, taskObj = null) {
  const task = (clsOrTask && typeof clsOrTask === 'object') ? clsOrTask : (taskObj || null);
  const norm = task ? String(task.task_class || '').toLowerCase() : String(clsOrTask || '').toLowerCase();
  if (norm === 'cpu') return 'cpu';
  if (norm === 'llm') return inferLlmTier(task);
  if (norm === 'script') return 'gpu';
  if (norm === 'meta') return 'meta';
  return norm || '-';
}
function classBadge(cls, executor = '', task = null) {
  if (String(executor || '').toLowerCase() === 'brain') {
    return `<span class="pill brain">BRAIN</span>`;
  }
  const type = task ? taskType(task) : taskType(cls);
  const map = {
    cpu: 'taskcpu',
    llm: 'llm',
    llm7b: 'llm7b',
    llm14b: 'llm14b',
    gpu: 'script',
    meta: 'meta',
    brain: 'brain'
  };
  const key = map[type] || 'script';
  const labelMap = { llm7b: 'LLM 7B', llm14b: 'LLM 14B' };
  const label = labelMap[type] || String(type).toUpperCase();
  return `<span class="pill ${key}">${label}</span>`;
}
function executorBadge(executor) {
  const e = (executor || 'worker').toLowerCase() === 'brain' ? 'brain' : 'worker';
  return `<span class="pill ${e}">${e}</span>`;
}

function tempClass(cpu) {
  if (cpu === null || cpu === undefined) return '';
  if (cpu >= 85) return 'bad';
  if (cpu >= 75) return 'warn';
  return 'ok';
}

function hbClass(age) {
  if (age === null || age === undefined) return 'bad';
  if (age >= 120) return 'bad';
  if (age >= 60) return 'warn';
  return 'ok';
}

function table(headers, rows) {
  if (!rows.length) return '<div class="k">(none)</div>';
  return `
    <table>
      <thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
      <tbody>
        ${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}
      </tbody>
    </table>
  `;
}

function renderCountCards(counts) {
  const items = [
    ['Queue', counts.queue, ''],
    ['Processing', counts.processing, 'ok'],
    ['Private', counts.private, 'warn'],
    ['Complete', counts.complete, 'ok'],
    ['Failed', counts.failed, counts.failed > 0 ? 'bad' : ''],
  ];
  return items.map(([k, v, cls]) => `
    <div class="card"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>
  `).join('');
}

function renderAlerts(alerts) {
  const stickyMap = loadStickyAlerts();
  const seenMap = loadAlertSeenMap();
  const dismissedStickyMap = loadDismissedStickyAlerts();
  const dismissedAlertMap = loadDismissedAlerts();
  const liveAlerts = Array.isArray(alerts) ? alerts : [];
  const nowIso = new Date().toISOString();
  const nowMs = Date.now();
  const dismissedTtlMs = 7 * 24 * 3600 * 1000;

  Object.keys(dismissedStickyMap).forEach((k) => {
    const ts = new Date(String((dismissedStickyMap[k] || {}).dismissed_at || '')).getTime();
    if (!Number.isFinite(ts) || (nowMs - ts) > dismissedTtlMs) delete dismissedStickyMap[k];
  });
  Object.keys(dismissedAlertMap).forEach((k) => {
    const ts = new Date(String((dismissedAlertMap[k] || {}).dismissed_at || '')).getTime();
    if (!Number.isFinite(ts) || (nowMs - ts) > dismissedTtlMs) delete dismissedAlertMap[k];
  });

  // Promote server-declared sticky alerts into local persistent storage.
  liveAlerts.forEach(a => {
    const stickyId = a && a.sticky ? String(a.sticky_id || '') : '';
    if (!stickyId) return;
    if (dismissedStickyMap[stickyId]) return;
    const prev = stickyMap[stickyId] || {};
    stickyMap[stickyId] = {
      ...prev,
      ...a,
      sticky: true,
      sticky_id: stickyId,
      first_seen_at: prev.first_seen_at || nowIso,
      last_seen_at: nowIso,
    };
  });
  saveDismissedStickyAlerts(dismissedStickyMap);
  saveDismissedAlerts(dismissedAlertMap);
  saveStickyAlerts(stickyMap);

  const nonStickyLive = liveAlerts.filter(a => !(a && a.sticky && a.sticky_id));
  const stickyRows = Object.values(stickyMap);
  const merged = [...stickyRows, ...nonStickyLive];
  const activeKeys = new Set();

  const alertKey = (a) => {
    const stickyId = a && a.sticky ? String(a.sticky_id || '') : '';
    if (stickyId) return `sticky:${stickyId}`;
    const sev = String((a && a.severity) || 'warn').toLowerCase();
    const worker = String((a && a.worker) || '');
    const message = String((a && a.message) || '');
    return `live:${sev}|${worker}|${message}`;
  };

  const ageText = (ts) => {
    const d = new Date(ts || '');
    if (Number.isNaN(d.getTime())) return '-';
    const sec = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  };

  const normalized = merged.map((a) => {
    const key = alertKey(a);
    if (dismissedAlertMap[key]) return null;
    activeKeys.add(key);
    const prev = seenMap[key] || {};
    const appearedAt = a.appeared_at || a.first_seen_at || prev.first_seen_at || new Date().toISOString();
    seenMap[key] = {
      ...prev,
      first_seen_at: appearedAt,
      last_seen_at: new Date().toISOString(),
    };
    return { ...a, appeared_at: appearedAt, _alert_key: key };
  }).filter(Boolean);

  const maxSeenAgeMs = 7 * 24 * 3600 * 1000;
  Object.keys(seenMap).forEach((k) => {
    if (activeKeys.has(k)) return;
    const lastSeen = new Date(String((seenMap[k] || {}).last_seen_at || '')).getTime();
    if (!Number.isFinite(lastSeen) || (nowMs - lastSeen) > maxSeenAgeMs) delete seenMap[k];
  });
  saveAlertSeenMap(seenMap);

  if (!normalized.length) {
    document.getElementById('alerts').innerHTML = '<div class="k">(none)</div>';
    return;
  }

  const rows = normalized.slice(0, 60).map(a => {
    const stickyId = a && a.sticky ? String(a.sticky_id || '') : '';
    const action = stickyId
      ? `<button class="small-btn" data-clear-sticky="${esc(stickyId)}" data-dismiss-alert="${esc(a._alert_key || '')}">Clear</button>`
      : `<button class="small-btn" data-dismiss-alert="${esc(a._alert_key || '')}">Hide</button>`;
    const severity = String((a && a.severity) || 'warn').toLowerCase();
    const sevClass = severity === 'bad' ? 'bad' : (severity === 'ok' ? 'ok' : 'warn');
    return [
    `<span class="${sevClass}">${esc(severity)}</span>`,
    fmt(a.worker),
    truncCell(a.message, 100, false),
    fmtTs(a.appeared_at),
    ageText(a.appeared_at),
    action,
  ];
  });

  document.getElementById('alerts').innerHTML = table(['Severity', 'Worker', 'Message', 'Appeared', 'Age', 'Action'], rows);
  document.querySelectorAll('#alerts button[data-clear-sticky]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-clear-sticky');
      const cur = loadStickyAlerts();
      const dismissed = loadDismissedStickyAlerts();
      if (id && cur[id]) {
        delete cur[id];
        saveStickyAlerts(cur);
        dismissed[id] = { dismissed_at: new Date().toISOString() };
        saveDismissedStickyAlerts(dismissed);
        renderAlerts((latestStatus && latestStatus.alerts) || []);
      }
    });
  });
  document.querySelectorAll('#alerts button[data-dismiss-alert]').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = String(btn.getAttribute('data-dismiss-alert') || '').trim();
      if (!key) return;
      const dismissed = loadDismissedAlerts();
      dismissed[key] = { dismissed_at: new Date().toISOString() };
      saveDismissedAlerts(dismissed);
      renderAlerts((latestStatus && latestStatus.alerts) || []);
    });
  });
  bindCopyables('#alerts');
}

function isSystemMetaTaskName(taskName) {
  const n = String(taskName || '').toLowerCase();
  return n.startsWith('load_llm') || n.startsWith('load_worker_model');
}

function renderTaskLane(targetId, items) {
  const classes = [...new Set(items.map(t => taskType(t)).filter(Boolean))].sort();
  const executors = [...new Set(items.map(t => (t.executor || 'worker').toLowerCase()).filter(Boolean))].sort();
  const sort = laneState.sort || 'task_asc';
  const showRuntime = activeLane === 'processing';
  const showCompletedAt = activeLane === 'complete';
  const showTimeTaken = activeLane === 'complete';
  const showQueuedAt = activeLane === 'queue';
  const showFailedAt = activeLane === 'failed';

  let filtered = items.filter(t => {
    const cls = taskType(t);
    const task = (t.name || '').toLowerCase();
    const worker = (t.assigned_to || '').toLowerCase();
    const executor = (t.executor || 'worker').toLowerCase();
    const batch = (t.batch_id || '').toLowerCase();
    const err = (t.error || '').toLowerCase();
    if (laneVisibleBatchIds && !laneVisibleBatchIds.has(t.batch_id || '') && !isSystemMetaTaskName(t.name)) return false;
    if (laneState.taskClass && cls !== laneState.taskClass) return false;
    if (laneState.task && !task.includes(laneState.task.toLowerCase())) return false;
    if (laneState.worker && !worker.includes(laneState.worker.toLowerCase())) return false;
    if (laneState.executor && executor !== laneState.executor.toLowerCase()) return false;
    if (laneState.batch && !batch.includes(laneState.batch.toLowerCase())) return false;
    if (laneState.error && !err.includes(laneState.error.toLowerCase())) return false;
    return true;
  });

  const getTimeTakenMs = (t) => {
    if (!t.started_at || !t.completed_at) return -1;
    const start = new Date(String(t.started_at).includes('T') ? t.started_at : String(t.started_at).replace(' ', 'T'));
    const end = new Date(String(t.completed_at).includes('T') ? t.completed_at : String(t.completed_at).replace(' ', 'T'));
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return -1;
    return end.getTime() - start.getTime();
  };
  const getQueuedAt = (t) => t.stale_requeued_at || t.requeued_at || t.last_attempt_at || t.created_at || '';
  const getFailedAt = (t) => t.completed_at || t.last_attempt_at || t.started_at || t.created_at || '';
  filtered.sort((a, b) => {
    const aval = (key) => String(a[key] || '').toLowerCase();
    const bval = (key) => String(b[key] || '').toLowerCase();
    const atype = taskType(a);
    const btype = taskType(b);
    if (sort === 'class_asc') return atype.localeCompare(btype);
    if (sort === 'class_desc') return btype.localeCompare(atype);
    if (sort === 'task_asc') return aval('name').localeCompare(bval('name'));
    if (sort === 'task_desc') return bval('name').localeCompare(aval('name'));
    if (sort === 'worker_asc') return aval('assigned_to').localeCompare(bval('assigned_to'));
    if (sort === 'worker_desc') return bval('assigned_to').localeCompare(aval('assigned_to'));
    if (sort === 'executor_asc') return aval('executor').localeCompare(bval('executor'));
    if (sort === 'executor_desc') return bval('executor').localeCompare(aval('executor'));
    if (sort === 'batch_asc') return aval('batch_id').localeCompare(bval('batch_id'));
    if (sort === 'batch_desc') return bval('batch_id').localeCompare(aval('batch_id'));
    if (sort === 'queued_asc') return String(getQueuedAt(a)).localeCompare(String(getQueuedAt(b)));
    if (sort === 'queued_desc') return String(getQueuedAt(b)).localeCompare(String(getQueuedAt(a)));
    if (sort === 'started_asc') return aval('started_at').localeCompare(bval('started_at'));
    if (sort === 'started_desc') return bval('started_at').localeCompare(aval('started_at'));
    if (sort === 'timetaken_asc') return getTimeTakenMs(a) - getTimeTakenMs(b);
    if (sort === 'timetaken_desc') return getTimeTakenMs(b) - getTimeTakenMs(a);
    if (sort === 'completed_asc') return aval('completed_at').localeCompare(bval('completed_at'));
    if (sort === 'completed_desc') return bval('completed_at').localeCompare(aval('completed_at'));
    if (sort === 'failedtime_asc') return String(getFailedAt(a)).localeCompare(String(getFailedAt(b)));
    if (sort === 'failedtime_desc') return String(getFailedAt(b)).localeCompare(String(getFailedAt(a)));
    if (sort === 'try_desc') return (Number(b.attempts || 0) - Number(a.attempts || 0));
    if (sort === 'try_asc') return (Number(a.attempts || 0) - Number(b.attempts || 0));
    if (sort === 'error_asc') return aval('error').localeCompare(bval('error'));
    if (sort === 'error_desc') return bval('error').localeCompare(aval('error'));
    return 0;
  });

  const runtimeText = (startedAt) => {
    if (!startedAt) return '-';
    const dt = new Date(startedAt);
    if (Number.isNaN(dt.getTime())) return '-';
    const sec = Math.max(0, Math.floor((Date.now() - dt.getTime()) / 1000));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  };
  const completedAtText = (completedAt) => {
    if (!completedAt) return '-';
    const raw = String(completedAt).trim();
    // Prefer local display always (same basis as the top clock).
    // Normalize common ISO-ish variants before parsing.
    const normalized = raw.includes('T') ? raw : raw.replace(' ', 'T');
    const dt = new Date(normalized);
    if (Number.isNaN(dt.getTime())) return '-';
    return dt.toLocaleString(undefined, { hour12: false });
  };
  const timeTakenText = (startedAt, completedAt) => {
    if (!startedAt || !completedAt) return '-';
    const start = new Date(String(startedAt).includes('T') ? startedAt : String(startedAt).replace(' ', 'T'));
    const end = new Date(String(completedAt).includes('T') ? completedAt : String(completedAt).replace(' ', 'T'));
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return '-';
    const sec = Math.max(0, Math.floor((end.getTime() - start.getTime()) / 1000));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  };
  const queuedAtValue = (t) =>
    t.stale_requeued_at ||
    t.requeued_at ||
    t.last_attempt_at ||
    t.created_at ||
    '';
  const failedAtValue = (t) =>
    t.completed_at ||
    t.last_attempt_at ||
    t.started_at ||
    t.created_at ||
    '';

  const rows = filtered.map(t => [
    classBadge(t.task_class, t.executor, t),
    truncCell(t.name, 42, false),
    truncCell(t.assigned_to, 24, false),
    executorBadge(t.executor),
    `<span class="mono">${fmt(t.batch_id)}</span>`,
    ...(showQueuedAt ? [ `<span class="mono">${completedAtText(queuedAtValue(t))}</span>` ] : []),
    ...(showRuntime ? [ `<span class="mono">${runtimeText(t.started_at)}</span>` ] : []),
    ...(showTimeTaken ? [ `<span class="mono">${timeTakenText(t.started_at, t.completed_at)}</span>` ] : []),
    ...(showCompletedAt ? [ `<span class="mono">${completedAtText(t.completed_at || t.started_at)}</span>` ] : []),
    ...(showFailedAt ? [ `<span class="mono">${completedAtText(failedAtValue(t))}</span>` ] : []),
    fmt(t.attempts),
    truncCell(t.error, 220, true)
  ]);
  const sortArrow = (ascKey, descKey) => sort === ascKey ? ' ↑' : (sort === descKey ? ' ↓' : '');
  const controls = `
    <div class="filter-bar">
      <span class="k">showing ${filtered.length} of ${items.length}</span>
      <label class="filter-field k">Type
        <select id="laneFilterClass">
          <option value="">all</option>
          ${classes.map(c => `<option value="${c}" ${laneState.taskClass === c ? 'selected' : ''}>${c}</option>`).join('')}
        </select>
      </label>
      <label class="filter-field k">Task <input id="laneFilterTask" value="${laneState.task}" placeholder="contains" /></label>
      <label class="filter-field k">Worker <input id="laneFilterWorker" value="${laneState.worker}" placeholder="contains" /></label>
      <label class="filter-field k">Executor
        <select id="laneFilterExecutor">
          <option value="">all</option>
          ${executors.map(e => `<option value="${e}" ${laneState.executor === e ? 'selected' : ''}>${e}</option>`).join('')}
        </select>
      </label>
      <label class="filter-field k">Batch <input id="laneFilterBatch" value="${laneState.batch}" placeholder="contains" /></label>
      <label class="filter-field k">Error <input id="laneFilterError" value="${laneState.error}" placeholder="contains" /></label>
      <span class="k">Click headers to sort</span>
    </div>
  `;
  const headers = [
    `<th data-sort="class">Type${sortArrow('class_asc', 'class_desc')}</th>`,
    `<th data-sort="task">Task${sortArrow('task_asc', 'task_desc')}</th>`,
    `<th data-sort="worker">Worker${sortArrow('worker_asc', 'worker_desc')}</th>`,
    `<th data-sort="executor">Executor${sortArrow('executor_asc', 'executor_desc')}</th>`,
    `<th data-sort="batch">Batch${sortArrow('batch_asc', 'batch_desc')}</th>`,
    ...(showQueuedAt ? [`<th data-sort="queued">Queued At${sortArrow('queued_asc', 'queued_desc')}</th>`] : []),
    ...(showRuntime ? [`<th data-sort="started">Runtime${sortArrow('started_asc', 'started_desc')}</th>`] : []),
    ...(showTimeTaken ? [`<th data-sort="timetaken">Time Taken${sortArrow('timetaken_asc', 'timetaken_desc')}</th>`] : []),
    ...(showCompletedAt ? [`<th data-sort="completed">Completed At${sortArrow('completed_asc', 'completed_desc')}</th>`] : []),
    ...(showFailedAt ? [`<th data-sort="failedtime">Failed At${sortArrow('failedtime_asc', 'failedtime_desc')}</th>`] : []),
    `<th data-sort="try">Try${sortArrow('try_asc', 'try_desc')}</th>`,
    `<th data-sort="error">Error${sortArrow('error_asc', 'error_desc')}</th>`
  ].join('');
  const laneTable = rows.length
    ? `<table><thead><tr>${headers}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`
    : '<div class="k">(none)</div>';
  document.getElementById(targetId).innerHTML = controls + laneTable;
  const bind = (id, key) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => { laneState[key] = el.value; renderTaskLane(targetId, items); });
    el.addEventListener('change', () => { laneState[key] = el.value; renderTaskLane(targetId, items); });
  };
  document.querySelectorAll('#laneTable th[data-sort]').forEach(th => {
    th.style.cursor = 'pointer';
    th.addEventListener('click', () => {
      const key = th.getAttribute('data-sort');
      const current = laneState.sort || 'task_asc';
      const nextMap = {
        class: ['class_asc', 'class_desc'],
        task: ['task_asc', 'task_desc'],
        worker: ['worker_asc', 'worker_desc'],
        executor: ['executor_asc', 'executor_desc'],
        batch: ['batch_asc', 'batch_desc'],
        queued: ['queued_asc', 'queued_desc'],
        started: ['started_asc', 'started_desc'],
        timetaken: ['timetaken_asc', 'timetaken_desc'],
        completed: ['completed_asc', 'completed_desc'],
        failedtime: ['failedtime_asc', 'failedtime_desc'],
        try: ['try_asc', 'try_desc'],
        error: ['error_asc', 'error_desc'],
      };
      const pair = nextMap[key] || ['task_asc', 'task_desc'];
      laneState.sort = current === pair[0] ? pair[1] : pair[0];
      renderTaskLane(targetId, items);
    });
  });
  bind('laneFilterClass', 'taskClass');
  bind('laneFilterTask', 'task');
  bind('laneFilterWorker', 'worker');
  bind('laneFilterExecutor', 'executor');
  bind('laneFilterBatch', 'batch');
  bind('laneFilterError', 'error');
  bindCopyables(`#${targetId}`);
}

function renderBatches(activeBatches) {
  const plans = [...new Set(activeBatches.map(b => b.plan).filter(Boolean))].sort();
  const sort = batchState.sort || 'started_desc';
  const tracked = loadTrackedBatches();
  let filtered = activeBatches.filter(b => {
    const bid = (b.id || '').toLowerCase();
    const plan = (b.plan || '').toLowerCase();
    if (batchState.batch && !bid.includes(batchState.batch.toLowerCase())) return false;
    if (batchState.plan && plan !== batchState.plan.toLowerCase()) return false;
    return true;
  });
  filtered.sort((a, b) => {
    if (sort === 'batch_asc') return String(a.id || '').localeCompare(String(b.id || ''));
    if (sort === 'batch_desc') return String(b.id || '').localeCompare(String(a.id || ''));
    if (sort === 'plan_asc') return String(a.plan || '').localeCompare(String(b.plan || ''));
    if (sort === 'plan_desc') return String(b.plan || '').localeCompare(String(a.plan || ''));
    if (sort === 'stage_asc') return Number(a.stage_rank || 0) - Number(b.stage_rank || 0);
    if (sort === 'stage_desc') return Number(b.stage_rank || 0) - Number(a.stage_rank || 0);
    if (sort === 'done_desc') return Number(b.done_pct || 0) - Number(a.done_pct || 0);
    if (sort === 'done_asc') return Number(a.done_pct || 0) - Number(b.done_pct || 0);
    if (sort === 'complete_desc') return Number(b.complete || 0) - Number(a.complete || 0);
    if (sort === 'complete_asc') return Number(a.complete || 0) - Number(b.complete || 0);
    if (sort === 'queue_desc') return Number(b.queue || 0) - Number(a.queue || 0);
    if (sort === 'queue_asc') return Number(a.queue || 0) - Number(b.queue || 0);
    if (sort === 'processing_desc') return Number(b.processing || 0) - Number(a.processing || 0);
    if (sort === 'processing_asc') return Number(a.processing || 0) - Number(b.processing || 0);
    if (sort === 'private_desc') return Number(b.private || 0) - Number(a.private || 0);
    if (sort === 'private_asc') return Number(a.private || 0) - Number(b.private || 0);
    if (sort === 'failed_desc') return Number(b.failed || 0) - Number(a.failed || 0);
    if (sort === 'failed_asc') return Number(a.failed || 0) - Number(b.failed || 0);
    if (sort === 'started_asc') return String(a.started_raw || '').localeCompare(String(b.started_raw || ''));
    return String(b.started_raw || '').localeCompare(String(a.started_raw || ''));
  });
  const stageClass = (stage) => {
    if (stage === 'complete') return 'ok';
    if (stage === 'failed') return 'bad';
    if (stage === 'processing') return 'warn';
    return '';
  };
  const rows = filtered.map(b => ({
    id: b.id,
    stage: b.stage,
    checked: tracked.has(String(b.id || '').trim()),
    cells: [
      `<input type="checkbox" data-batch-check="${esc(b.id)}" ${tracked.has(String(b.id || '').trim()) ? 'checked' : ''} />`,
      `<span class="mono">${b.id}</span>`,
      fmt(b.plan),
      fmt(b.priority),
      `<span class="${stageClass(b.stage)}">${fmt(b.stage)}</span>`,
      `<span class="mono">${fmt(b.complete)}/${fmt(b.total)}</span>`,
      fmt(b.goal),
      fmt(b.queue),
      fmt(b.processing),
      fmt(b.failed),
      fmt(b.private),
      fmt(b.started),
      `<button class="small-btn" data-dismiss-batch="${esc(b.id)}" title="Dismiss batch">×</button>`
    ]
  }));
  const checkedVisible = filtered
    .map(b => String(b.id || '').trim())
    .filter(id => tracked.has(id));
  visibleBatchIds = checkedVisible.length ? new Set(checkedVisible) : new Set();
  const sortArrow = (ascKey, descKey) => sort === ascKey ? ' ↑' : (sort === descKey ? ' ↓' : '');
  const trackedCount = tracked.size;
  const controls = `
    <div class="filter-bar">
      <span class="k">showing ${filtered.length} of ${activeBatches.length} active batches</span>
      <label class="filter-field k">Batch <input id="batchFilterBatch" value="${batchState.batch}" placeholder="contains" /></label>
      <button class="small-btn" id="selectVisibleBatches">Select visible</button>
      <button class="small-btn" id="unselectVisibleBatches">Unselect visible</button>
      ${trackedCount > 0 ? '<button class="small-btn" id="clearTrackedBatches">Clear selection</button>' : ''}
      <span class="k">Check batches to show their chains and task lanes. Expand chain cards to narrow further.</span>
      <span class="k">Click headers to sort</span>
    </div>
  `;
  const headers = [
    `<th>Show</th>`,
    `<th data-sort="batch">Batch${sortArrow('batch_asc', 'batch_desc')}</th>`,
    `<th data-sort="plan">Plan${sortArrow('plan_asc', 'plan_desc')}</th>`,
    `<th>Priority</th>`,
    `<th data-sort="stage">Stage${sortArrow('stage_asc', 'stage_desc')}</th>`,
    `<th data-sort="complete">Complete${sortArrow('complete_asc', 'complete_desc')}</th>`,
    `<th>Goal</th>`,
    `<th data-sort="queue">Queue${sortArrow('queue_asc', 'queue_desc')}</th>`,
    `<th data-sort="processing">Processing${sortArrow('processing_asc', 'processing_desc')}</th>`,
    `<th data-sort="failed">Failed${sortArrow('failed_asc', 'failed_desc')}</th>`,
    `<th data-sort="private">Private${sortArrow('private_asc', 'private_desc')}</th>`,
    `<th data-sort="started">Started${sortArrow('started_asc', 'started_desc')}</th>`,
    `<th></th>`,
  ].join('');
  const batchTable = rows.length
    ? `<table><thead><tr>${headers}</tr></thead><tbody>${rows.map(r => `<tr data-batch-row="${esc(r.id)}">${r.cells.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`
    : '<div class="k">(none)</div>';
  document.getElementById('batches').innerHTML = controls + batchTable;
  const bind = (id, key) => {
    const el = document.getElementById(id);
    if (!el) return;
    const rerender = () => {
      batchState[key] = el.value;
      renderBatches(activeBatches);
      if (latestStatus) {
        renderBatchChains(latestStatus, visibleBatchIds);
        syncLaneVisibleBatchIds(latestStatus);
        renderLaneTabs(latestStatus.counts, latestStatus.lanes);
      }
    };
    el.addEventListener('input', rerender);
    el.addEventListener('change', rerender);
  };
  document.querySelectorAll('#batches th[data-sort]').forEach(th => {
    th.style.cursor = 'pointer';
    th.addEventListener('click', () => {
      const key = th.getAttribute('data-sort');
      const current = batchState.sort || 'started_desc';
      const nextMap = {
        batch: ['batch_asc', 'batch_desc'],
        plan: ['plan_asc', 'plan_desc'],
        stage: ['stage_asc', 'stage_desc'],
        complete: ['complete_asc', 'complete_desc'],
        queue: ['queue_asc', 'queue_desc'],
        processing: ['processing_asc', 'processing_desc'],
        failed: ['failed_asc', 'failed_desc'],
        private: ['private_asc', 'private_desc'],
        started: ['started_asc', 'started_desc'],
      };
      const pair = nextMap[key] || ['started_asc', 'started_desc'];
      batchState.sort = current === pair[0] ? pair[1] : pair[0];
      renderBatches(activeBatches);
      if (latestStatus) {
        renderBatchChains(latestStatus, visibleBatchIds);
        syncLaneVisibleBatchIds(latestStatus);
        renderLaneTabs(latestStatus.counts, latestStatus.lanes);
      }
    });
  });
  bind('batchFilterBatch', 'batch');
  // Remove-tracked batch buttons
  document.querySelectorAll('#batches input[data-batch-check]').forEach(inp => {
    inp.addEventListener('change', () => {
      const batchId = inp.getAttribute('data-batch-check');
      setTrackedBatchChecked(batchId, !!inp.checked);
      renderBatches(activeBatches);
      if (latestStatus) {
        renderBatchChains(latestStatus, visibleBatchIds);
        syncLaneVisibleBatchIds(latestStatus);
        renderLaneTabs(latestStatus.counts, latestStatus.lanes);
      }
    });
  });
  document.querySelectorAll('#batches button[data-dismiss-batch]').forEach(btn => {
    btn.addEventListener('click', () => {
      const batchId = btn.getAttribute('data-dismiss-batch');
      removeTrackedBatch(batchId);
      renderBatches(activeBatches);
      if (latestStatus) {
        renderBatchChains(latestStatus, visibleBatchIds);
        syncLaneVisibleBatchIds(latestStatus);
        renderLaneTabs(latestStatus.counts, latestStatus.lanes);
      }
    });
  });
  // Clear tracked batches button
  const clearBtn = document.getElementById('clearTrackedBatches');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      saveTrackedBatches(new Set());
      renderBatches(activeBatches);
      if (latestStatus) {
        renderBatchChains(latestStatus, visibleBatchIds);
        syncLaneVisibleBatchIds(latestStatus);
        renderLaneTabs(latestStatus.counts, latestStatus.lanes);
      }
    });
  }
  const selectVisibleBtn = document.getElementById('selectVisibleBatches');
  if (selectVisibleBtn) {
    selectVisibleBtn.addEventListener('click', () => {
      const tr = loadTrackedBatches();
      filtered.forEach((b) => { if (isTrackableBatchId(b.id)) tr.add(String(b.id).trim()); });
      saveTrackedBatches(tr);
      renderBatches(activeBatches);
      if (latestStatus) {
        renderBatchChains(latestStatus, visibleBatchIds);
        syncLaneVisibleBatchIds(latestStatus);
        renderLaneTabs(latestStatus.counts, latestStatus.lanes);
      }
    });
  }
  const unselectVisibleBtn = document.getElementById('unselectVisibleBatches');
  if (unselectVisibleBtn) {
    unselectVisibleBtn.addEventListener('click', () => {
      const tr = loadTrackedBatches();
      filtered.forEach((b) => tr.delete(String(b.id || '').trim()));
      saveTrackedBatches(tr);
      renderBatches(activeBatches);
      if (latestStatus) {
        renderBatchChains(latestStatus, visibleBatchIds);
        syncLaneVisibleBatchIds(latestStatus);
        renderLaneTabs(latestStatus.counts, latestStatus.lanes);
      }
    });
  }
  // Lane scope follows expanded chain cards for focused analysis.
}

function syncLaneVisibleBatchIds(data) {
  if (!visibleBatchIds) {
    laneVisibleBatchIds = null;
    return;
  }
  // If no batches are checked, hide lane rows/chains (explicit user selection mode).
  if (visibleBatchIds.size === 0) {
    laneVisibleBatchIds = new Set();
    return;
  }
  const chains = data.batch_chains || {};
  const expanded = Object.keys(chains).filter(batchId => {
    if (!visibleBatchIds.has(batchId)) return false;
    const st = chainState[batchId];
    return !!st && st.collapsed === false;
  });
  laneVisibleBatchIds = expanded.length ? new Set(expanded) : new Set(visibleBatchIds);
}

function renderLaneTabs(counts, lanes) {
  counts = counts || {};
  lanes = lanes || {};
  const labels = {
    queue: 'Queue',
    processing: 'Processing',
    private: 'Private',
    complete: 'Complete',
    failed: 'Failed'
  };
  const html = laneOrder.map(l => {
    const cls = l === activeLane ? 'tab-btn active' : 'tab-btn';
    const count = counts[l] ?? 0;
    return `<button class="${cls}" data-lane="${l}">${labels[l]} (${count})</button>`;
  }).join('');
  const tabs = document.getElementById('laneTabs');
  if (!tabs) {
    console.error('laneTabs element not found');
    return;
  }
  tabs.innerHTML = html;
  tabs.querySelectorAll('button[data-lane]').forEach(btn => {
    btn.addEventListener('click', () => {
      activeLane = btn.getAttribute('data-lane');
      renderLaneTabs(counts, lanes);
      renderTaskLane('laneTable', lanes[activeLane] || []);
    });
  });
  renderTaskLane('laneTable', lanes[activeLane] || []);
}

function renderWorkerTabs(workers) {
  const gpuWorkers = workers.filter(w => w.type === 'gpu');
  const cpuWorkers = workers.filter(w => w.type === 'cpu');
  const tabs = [
    { key: 'gpu', label: 'GPU', count: gpuWorkers.length },
    { key: 'cpu', label: 'CPU', count: cpuWorkers.length },
  ];
  const html = tabs.map(t => {
    const cls = t.key === activeWorkerTab ? 'tab-btn active' : 'tab-btn';
    return `<button class="${cls}" data-wtab="${t.key}">${t.label} (${t.count})</button>`;
  }).join('');
  const container = document.getElementById('workerTabs');
  container.innerHTML = html;
  container.querySelectorAll('button[data-wtab]').forEach(btn => {
    btn.addEventListener('click', () => {
      activeWorkerTab = btn.getAttribute('data-wtab');
      renderWorkerTabs(workers);
    });
  });
  const active = activeWorkerTab === 'gpu' ? gpuWorkers : cpuWorkers;
  renderWorkerTable(active, activeWorkerTab);
}

function renderWorkerTable(workers, type) {
  if (type === 'gpu') {
    const splitGroupTotals = {};
    const splitGroupHoldings = {};
    workers.forEach((w) => {
      const gid = String(w.runtime_group_id || '').trim();
      const placement = String(w.runtime_placement || '').trim();
      if (!gid || placement !== 'split_gpu') return;
      if (!splitGroupTotals[gid]) {
        splitGroupTotals[gid] = { used: 0, total: 0, members: 0 };
      }
      if (!splitGroupHoldings[gid]) {
        splitGroupHoldings[gid] = new Set();
      }
      if (w.vram_used_mb !== null && w.vram_used_mb !== undefined) {
        splitGroupTotals[gid].used += Number(w.vram_used_mb) || 0;
      }
      if (w.vram_total_mb !== null && w.vram_total_mb !== undefined) {
        splitGroupTotals[gid].total += Number(w.vram_total_mb) || 0;
      }
      (w.holding || []).forEach((h) => {
        const text = String(h || '').trim();
        if (text) splitGroupHoldings[gid].add(text);
      });
      splitGroupTotals[gid].members += 1;
    });
    const stateLabel = (w) => {
      const placement = String(w.runtime_placement || '').trim();
      const gid = String(w.runtime_group_id || '').trim();
      if (placement === 'split_gpu' && gid) {
        return `LINKED (${gid})`;
      }
      return fmt(w.state);
    };
    const vram = (w) => {
      const self = (w.vram_used_mb !== null && w.vram_total_mb !== null)
        ? `${w.vram_used_mb}/${w.vram_total_mb}` : '-';
      const gid = String(w.runtime_group_id || '').trim();
      const placement = String(w.runtime_placement || '').trim();
      if (!gid || placement !== 'split_gpu') return self;
      const grp = splitGroupTotals[gid];
      if (!grp || grp.members < 2 || grp.total <= 0) return self;
      return `${self} (grp ${grp.used}/${grp.total})`;
    };
    const holdingLabel = (w) => {
      const gid = String(w.runtime_group_id || '').trim();
      const placement = String(w.runtime_placement || '').trim();
      if (!gid || placement !== 'split_gpu') {
        return truncCell((w.holding || []).slice(0,2).join(' | ') || '-', 64, true);
      }
      const groupItems = [...(splitGroupHoldings[gid] || new Set())].slice(0, 3);
      const groupText = groupItems.length ? `GROUP ${groupItems.join(' | ')}` : 'GROUP -';
      return truncCell(groupText, 80, true);
    };
    const rows = workers.map(w => [
      fmt(w.name),
      stateLabel(w),
      fmt(w.host),
      `<span class="${tempClass(w.cpu_temp_c)}">${fmt(w.cpu_temp_c)}</span>`,
      fmt(w.gpu_temp_c),
      fmt(w.gpu_util),
      fmt(w.power_w),
      vram(w),
      fmt(w.thermal_cause && w.thermal_cause !== 'none' ? w.thermal_cause : '-'),
      holdingLabel(w),
      `<span class="${hbClass(w.age_s)}">${fmt(w.age_s)}</span>`
    ]);
    document.getElementById('workerTable').innerHTML = table(
      ['Name', 'State', 'Host', 'CPU C', 'GPU C', 'GPU %', 'W', 'VRAM', 'Thermal', 'Holding', 'HB s'],
      rows
    );
  } else {
    const rows = workers.map(w => [
      fmt(w.name),
      fmt(w.state),
      fmt(w.host),
      `<span class="${tempClass(w.cpu_temp_c)}">${fmt(w.cpu_temp_c)}</span>`,
      fmt(w.thermal_cause && w.thermal_cause !== 'none' ? w.thermal_cause : '-'),
      truncCell((w.holding || []).slice(0,2).join(' | ') || '-', 64, true),
      `<span class="${hbClass(w.age_s)}">${fmt(w.age_s)}</span>`
    ]);
    document.getElementById('workerTable').innerHTML = table(
      ['Name', 'State', 'Host', 'CPU C', 'Thermal', 'Holding', 'HB s'],
      rows
    );
  }
  bindCopyables('#workerTable');
}

function laneChip(lane) {
  if (!lane || lane === '-') return '<span class="chip missing">-</span>';
  return `<span class="chip ${lane}">${lane}</span>`;
}

function renderBatchChains(data, allowedBatchIds = null) {
  const perPage = 15;
  const out = [];
  const chains = data.batch_chains || {};
  const laneRank = { '-': 0, queue: 1, private: 2, processing: 3, complete: 4, failed: 5 };
  Object.entries(chains).forEach(([batchId, chain]) => {
    if (allowedBatchIds && !allowedBatchIds.has(batchId)) return;
    const stages = chain.stage_order || [];
    const stageTypes = chain.stage_types || {};
    const collapsedStages = chain.collapsed_order || stages;
    const collapsedTypes = chain.collapsed_types || stageTypes;
    const collapsedCounts = chain.collapsed_counts || {};
    const collapsedMap = chain.collapsed_map || {};
    if (!stages.length) return;
    const totalRows = chain.row_count || (chain.rows || []).length;

    // Compute stage status counts for highlighting active stages
    const stageLaneCounts = {};
    stages.forEach(s => { stageLaneCounts[s] = { queue: 0, processing: 0, private: 0, complete: 0, failed: 0 }; });
    (chain.rows || []).forEach(row => {
      stages.forEach(s => {
        const lane = (row.stages || {})[s];
        if (lane && stageLaneCounts[s] && stageLaneCounts[s][lane] !== undefined) {
          stageLaneCounts[s][lane]++;
        }
      });
    });

    // Aggregate counts for collapsed stages using server-provided normalization map.
    const collapsedLaneCounts = {};
    collapsedStages.forEach(cs => {
      collapsedLaneCounts[cs] = { queue: 0, processing: 0, private: 0, complete: 0, failed: 0 };
    });
    stages.forEach(s => {
      const base = collapsedMap[s] || s;
      if (collapsedLaneCounts[base]) {
        const counts = stageLaneCounts[s] || {};
        collapsedLaneCounts[base].queue += counts.queue || 0;
        collapsedLaneCounts[base].processing += counts.processing || 0;
        collapsedLaneCounts[base].private += counts.private || 0;
        collapsedLaneCounts[base].complete += counts.complete || 0;
        collapsedLaneCounts[base].failed += counts.failed || 0;
      }
    });

    const stageStatus = (s) => {
      const c = stageLaneCounts[s] || {};
      if (c.processing > 0) return 'processing';
      if (c.queue > 0 || c.private > 0) return 'queued';
      if (c.failed > 0 && c.complete === 0) return 'failed';
      if (c.complete > 0 && c.queue === 0 && c.processing === 0 && c.private === 0) return 'complete';
      return 'waiting';
    };
    const collapsedStageStatus = (s) => {
      const c = collapsedLaneCounts[s] || {};
      if (c.processing > 0) return 'processing';
      if (c.queue > 0 || c.private > 0) return 'queued';
      if (c.failed > 0 && c.complete === 0) return 'failed';
      if (c.complete > 0 && c.queue === 0 && c.processing === 0 && c.private === 0) return 'complete';
      return 'waiting';
    };
    const formatStage = (s) => {
      const status = stageStatus(s);
      if (status === 'processing') return `<strong style="color:#4cc9f0">${s}</strong>`;
      if (status === 'queued') return `<span style="color:#f9c74f">${s}</span>`;
      if (status === 'complete') return `<span style="color:#4ad66d">${s}</span>`;
      if (status === 'failed') return `<span style="color:#ff6b6b">${s}</span>`;
      return s;
    };
    const formatCollapsedStage = (s) => {
      const count = Number(collapsedCounts[s] || 0);
      const label = count > 1 ? `${s}×${count}` : s;
      const status = collapsedStageStatus(s);
      if (status === 'processing') return `<strong style="color:#4cc9f0">${label}</strong>`;
      if (status === 'queued') return `<span style="color:#f9c74f">${label}</span>`;
      if (status === 'complete') return `<span style="color:#4ad66d">${label}</span>`;
      if (status === 'failed') return `<span style="color:#ff6b6b">${label}</span>`;
      return label;
    };
    const chainHeader = collapsedStages.map(formatCollapsedStage).join(' <span style="color:#9db4c8">-></span> ');
    if (!chainState[batchId]) {
      chainState[batchId] = { collapsed: true, page: 1, sortKey: 'item', sortDir: 'asc' };
    }
    const st = chainState[batchId];
    const totalPages = Math.max(1, Math.ceil(totalRows / perPage));
    if (st.page > totalPages) st.page = totalPages;
    const sortKey = st.sortKey || 'item';
    const sortDir = st.sortDir || 'asc';
    const direction = sortDir === 'desc' ? -1 : 1;
    const sortedRows = [...(chain.rows || [])].sort((a, b) => {
      if (sortKey === 'item') {
        return direction * String(a.item || '').localeCompare(String(b.item || ''));
      }
      const ar = laneRank[(a.stages || {})[sortKey] || '-'] || 0;
      const br = laneRank[(b.stages || {})[sortKey] || '-'] || 0;
      if (ar !== br) return direction * (ar - br);
      return String(a.item || '').localeCompare(String(b.item || ''));
    });
    const start = (st.page - 1) * perPage;
    const end = start + perPage;

    const visibleRows = sortedRows.slice(start, end).map(r => {
      const cols = [ `<span class="mono">${r.item}</span>` ];
      stages.forEach(s => cols.push(laneChip((r.stages || {})[s])));
      return cols;
    });
    const arrow = (key) => (sortKey === key ? (sortDir === 'asc' ? ' ↑' : ' ↓') : '');
    const headers = [
      `<th data-batch="${batchId}" data-sort="item">Item${arrow('item')}</th>`,
      ...stages.map(s => {
        const stype = stageTypes[s] || '-';
        const badge = stype && stype !== '-' ? classBadge(stype) : '<span class="k">-</span>';
        return `<th data-batch="${batchId}" data-sort="${s}">${s}${arrow(s)}<div style="margin-top:4px">${badge}</div></th>`;
      })
    ];
    const chainTable = visibleRows.length
      ? `
        <table>
          <thead><tr>${headers.join('')}</tr></thead>
          <tbody>
            ${visibleRows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      `
      : '<div class="k">(none)</div>';
    const collapsed = st.collapsed;
    const body = collapsed
      ? ''
      : `${chainTable}
         <div class="k">showing ${visibleRows.length} of ${totalRows}</div>`;

    out.push(`
      <div class="chain-box" data-batch="${batchId}">
        <div class="chain-head">
          <div class="k">${batchId} chain: ${chainHeader}</div>
          <div class="chain-controls">
            <button class="small-btn" data-action="toggle" data-batch="${batchId}">${collapsed ? 'Expand' : 'Collapse'}</button>
            <button class="small-btn" data-action="prev" data-batch="${batchId}" ${collapsed || st.page <= 1 ? 'disabled' : ''}>Prev</button>
            <span class="k">Page ${st.page}/${totalPages}</span>
            <button class="small-btn" data-action="next" data-batch="${batchId}" ${collapsed || st.page >= totalPages ? 'disabled' : ''}>Next</button>
          </div>
        </div>
        <div>${body}</div>
      </div>
    `);
  });
  const container = document.getElementById('batchChains');
  container.innerHTML = out.join('') || '<div class="k">(no itemized dependency chains detected for current batch/plan filter)</div>';
  container.querySelectorAll('button[data-action]').forEach(btn => {
    btn.addEventListener('click', () => {
      const action = btn.getAttribute('data-action');
      const batchId = btn.getAttribute('data-batch');
      const st = chainState[batchId] || { collapsed: true, page: 1, sortKey: 'item', sortDir: 'asc' };
      const total = (chains[batchId]?.row_count) || 0;
      const pages = Math.max(1, Math.ceil(total / perPage));
      if (action === 'toggle') st.collapsed = !st.collapsed;
      if (action === 'prev' && st.page > 1) st.page -= 1;
      if (action === 'next' && st.page < pages) st.page += 1;
      chainState[batchId] = st;
      renderBatchChains(data, allowedBatchIds);
      syncLaneVisibleBatchIds(data);
      renderLaneTabs(data.counts, data.lanes);
    });
  });
  container.querySelectorAll('th[data-sort]').forEach(th => {
    th.style.cursor = 'pointer';
    th.addEventListener('click', () => {
      const batchId = th.getAttribute('data-batch');
      const key = th.getAttribute('data-sort');
      const st = chainState[batchId] || { collapsed: true, page: 1, sortKey: 'item', sortDir: 'asc' };
      if (st.sortKey === key) {
        st.sortDir = st.sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        st.sortKey = key;
        st.sortDir = 'asc';
      }
      st.page = 1;
      chainState[batchId] = st;
      renderBatchChains(data, allowedBatchIds);
      syncLaneVisibleBatchIds(data);
      renderLaneTabs(data.counts, data.lanes);
    });
  });
}

function refreshFromData(data) {
  latestStatus = data;
  document.getElementById('meta').textContent = `Updated ${new Date(data.generated_at).toLocaleTimeString()}`;
  document.getElementById('countCards').innerHTML = renderCountCards(data.counts);

  const batchRows = Object.entries(data.active_batches).map(([id, b]) => {
    const c = b.counts;
    const total = Math.max(b.total_hint || 0, c.queue + c.processing + c.private + c.complete + c.failed);
    let stage = 'idle';
    let stageRank = 0;
    const hasLiveWork = (c.processing > 0) || (c.queue > 0) || (c.private > 0);
    if (hasLiveWork && c.failed > 0) { stage = 'processing (warnings)'; stageRank = 3; }
    else if (c.processing > 0) { stage = 'processing'; stageRank = 3; }
    else if (c.failed > 0 && c.complete === 0) { stage = 'failed'; stageRank = 4; }
    else if (c.failed > 0 && c.complete > 0) { stage = 'complete (warnings)'; stageRank = 5; }
    else if (c.queue > 0 || c.private > 0) { stage = 'queued'; stageRank = 2; }
    else if (total > 0 && c.complete >= total) { stage = 'complete'; stageRank = 5; }
    else if (c.complete > 0) { stage = 'partial'; stageRank = 1; }
    // Goal progress string
    let goalStr = '-';
    if (b.goal) {
      const g = b.goal;
      const statusBadge = g.status === 'complete' ? '\u2705' : g.status === 'exhausted' ? '\u26a0' : g.status === 'draining' ? '\u23f3' : '';
      goalStr = `${g.accepted}/${g.target} ${statusBadge} (${g.in_flight} fly, ${g.rejected} rej, ${g.pool_remaining} pool)`;
    }
    return {
      id: id,
      plan: b.plan || '',
      priority: b.priority || 'normal',
      stage: stage,
      stage_rank: stageRank,
      done_pct: total > 0 ? (c.complete / total) : 0,
      complete: c.complete,
      total: total,
      queue: c.queue,
      processing: c.processing,
      failed: c.failed,
      private: c.private,
      goal: goalStr,
      started_raw: b.started_at || '',
      started: fmt(b.started_at ? new Date(b.started_at).toLocaleTimeString() : '-')
    };
  });
  // Auto-track only live batches so stale terminal-only rows do not
  // keep reappearing as "current" status noise.
  batchRows
    .filter((b) => Number(b.queue || 0) + Number(b.processing || 0) + Number(b.private || 0) > 0)
    .forEach((b) => addTrackedBatch(b.id));
  renderBatches(batchRows);
  renderBatchChains(data, visibleBatchIds);
  syncLaneVisibleBatchIds(data);

  const brainRows = (data.brain_gpus || []).map(w => {
    const vram = (w.vram_used_mb !== null && w.vram_total_mb !== null)
      ? `${w.vram_used_mb}/${w.vram_total_mb}`
      : '-';
    const thermal = w.thermal_cause && w.thermal_cause !== 'none' ? w.thermal_cause : '-';
    return [
      fmt(w.model),
      fmt(w.name),
      fmt(w.state),
      vram,
      `<span class="${tempClass(w.cpu_temp_c)}">${fmt(w.cpu_temp_c)}</span>`,
      fmt(w.gpu_temp_c),
      fmt(w.gpu_util),
      fmt(w.power_w),
      fmt(thermal),
      truncCell((w.holding || []).slice(0,2).join(' | ') || '-', 64, true),
      `<span class="${hbClass(w.age_s)}">${fmt(w.age_s)}</span>`
    ];
  });
  document.getElementById('brainGpus').innerHTML = table(
    ['Model', 'Name', 'State', 'VRAM', 'CPU C', 'GPU C', 'GPU %', 'W', 'Thermal', 'Holding', 'HB'],
    brainRows
  );
  bindCopyables('#brainGpus');

  renderWorkerTabs(data.workers);

  renderAlerts(data.alerts || []);
  renderLaneTabs(data.counts, data.lanes);
}

async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    const trackedIds = [...loadTrackedBatches()]
      .map(x => String(x || '').trim())
      .filter(Boolean)
      .filter(x => /^[A-Za-z0-9_.:-]{1,80}$/.test(x))
      .slice(0, 40);
    const query = trackedIds.length ? `?batch_ids=${encodeURIComponent(trackedIds.join(','))}` : '';
    let res;
    try {
      res = await fetch(`/api/status${query}`, { cache: 'no-store' });
    } catch (err) {
      // Recover from malformed/local-storage-driven query failures by retrying
      // without tracked ids and clearing local tracked state.
      if (!query) throw err;
      saveTrackedBatches(new Set());
      res = await fetch('/api/status', { cache: 'no-store' });
    }
    if (!res.ok) {
      console.error('Dashboard refresh failed:', res.status, res.statusText);
      const stamp = lastRefreshOkAt ? `last good ${new Date(lastRefreshOkAt).toLocaleTimeString()}` : 'no successful refresh yet';
      const meta = document.getElementById('meta');
      if (meta) meta.textContent = `Update failed (${res.status}) - ${stamp}`;
      return;
    }
    const data = await res.json();
    refreshFromData(data);
    lastRefreshOkAt = Date.now();
  } catch (err) {
    console.error('Dashboard refresh error:', err);
    const stamp = lastRefreshOkAt ? `last good ${new Date(lastRefreshOkAt).toLocaleTimeString()}` : 'no successful refresh yet';
    const meta = document.getElementById('meta');
    const hint = err && err.message ? ` (${String(err.message).slice(0, 120)})` : '';
    if (meta) meta.textContent = `Update error${hint} - ${stamp}`;
  } finally {
    refreshInFlight = false;
  }
}

refresh();
setInterval(refresh, 2000);
