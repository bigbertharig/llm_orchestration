let planDefaults = {};
let planStarters = {};
let planDefaultStarter = {};
let planInputs = {};
let planInputFiles = {};
let planScopes = {};
let shoulderPlans = [];
let armPlans = [];
let shoulderArmBindings = {};
let outputFiles = [];
let batchLabels = {};
const GLOBAL_HIDDEN_KEYS = new Set(['PRIORITY', 'PREEMPTIBLE']);
const PLAN_HIDDEN_KEYS = {
  github_analyzer: [
    'REPO_PATH',
    'REPO_URL',
    'CLAIMED_BEHAVIOR',
    'ANALYSIS_DEPTH',
    'HOT_WORKERS',
    'WORKER_MODEL',
    'WORKER_SHARDS',
    'WORKER_CONTEXT_TOKENS',
    'WORKER_CONTEXT_UTILIZATION',
    'BRAIN_MODEL',
    'PRIORITY',
    'PREEMPTIBLE'
  ]
};

async function api(path, payload) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {})
  });
  return await res.json();
}

function showResult(data) {
  const ok = data.ok ? '[ok]' : '[error]';
  const text = [
    ok + ' ' + (data.message || ''),
    data.cmd ? ('cmd: ' + data.cmd) : '',
    data.stdout ? ('\nstdout:\n' + data.stdout) : '',
    data.stderr ? ('\nstderr:\n' + data.stderr) : '',
  ].filter(Boolean).join('\n');
  const el = document.getElementById('result');
  el.textContent = text || '(no output)';
  el.className = data.ok ? 'ok' : 'bad';
}

function fmtTs(ts) {
  if (!ts) return '-';
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return String(ts);
  return d.toLocaleString();
}

function parseOptions(optionsText) {
  const raw = String(optionsText || '').trim();
  if (!raw) return [];
  return raw
    .split(/[|,/]/)
    .map(x => x.trim())
    .filter(Boolean);
}

function hiddenConfigKeys(planName) {
  const list = PLAN_HIDDEN_KEYS[planName] || [];
  return new Set(list);
}

function visibleConfigKeys(planName, cfg) {
  const hidden = hiddenConfigKeys(planName);
  return Object.keys(cfg || {}).filter(k => !hidden.has(k) && !GLOBAL_HIDDEN_KEYS.has(k));
}

function setQuickInputsFromConfig(planName, cfg) {
  const wrap = document.getElementById('quickInputsWrap');
  const showQuick = planName === 'github_analyzer';
  if (wrap) wrap.style.display = showQuick ? '' : 'none';
  if (!showQuick) return;
  const obj = (cfg && typeof cfg === 'object') ? cfg : {};
  document.getElementById('quickRepoUrl').value = String(obj.REPO_URL || '');
  document.getElementById('quickClaim').value = String(obj.CLAIMED_BEHAVIOR || '');
}

function getCurrentConfig() {
  try {
    const obj = JSON.parse(document.getElementById('planConfig').value || '{}');
    return (obj && typeof obj === 'object' && !Array.isArray(obj)) ? obj : {};
  } catch (e) {
    return {};
  }
}

function readFormConfig() {
  const out = {};
  document.querySelectorAll('[data-config-key]').forEach(el => {
    const key = el.getAttribute('data-config-key');
    if (!key) return;
    out[key] = String(el.value ?? '').trim();
  });
  return out;
}

function applyFormToJson() {
  const planName = document.getElementById('planName').value;
  const current = getCurrentConfig();
  const formCfg = readFormConfig();
  const cfg = { ...current, ...formCfg };
  const hidden = hiddenConfigKeys(planName);
  hidden.forEach(k => {
    if (!(k in cfg) && (k in current)) {
      cfg[k] = current[k];
    }
  });
  document.getElementById('planConfig').value = JSON.stringify(cfg, null, 2);
}

function loadJsonToForm() {
  renderConfigForm(document.getElementById('planName').value, getCurrentConfig());
}

function renderConfigForm(planName, overrideCfg) {
  const cfg = (overrideCfg && typeof overrideCfg === 'object') ? overrideCfg : (planDefaults[planName] || getCurrentConfig());
  const starterFile = document.getElementById('starterFile').value;
  const perStarter = planInputs[planName] || {};
  const helpList = perStarter[starterFile] || [];
  const helpByKey = {};
  helpList.forEach(x => { helpByKey[x.key] = x; });
  const keys = visibleConfigKeys(planName, cfg);
  const inputFiles = planInputFiles[planName] || [];
  const rows = keys.map(key => {
    const v = String(cfg[key] ?? '');
    const help = helpByKey[key] || {};
    const opts = parseOptions(help.options || '');
    const desc = help.description ? `<div class="field-help">${help.description}</div>` : '';
    const hint = help.options ? `<div class="field-help">Options: ${help.options}</div>` : '<div class="field-help">(free text)</div>';
    if (opts.length) {
      const options = opts.map(o => `<option value="${o}" ${o === v ? 'selected' : ''}>${o}</option>`).join('');
      return `<div class="field-row"><div class="field-key">${key}</div><div>${desc}${hint}<select data-config-key="${key}">${options}</select></div></div>`;
    }
    if (key.endsWith('_FILE')) {
      const listId = `list_${key.replace(/[^a-zA-Z0-9_]/g, '_')}`;
      const dlist = inputFiles.map(p => `<option value="${p}"></option>`).join('');
      return `<div class="field-row"><div class="field-key">${key}</div><div>${desc}${hint}<input data-config-key="${key}" value="${v}" list="${listId}" /><datalist id="${listId}">${dlist}</datalist></div></div>`;
    }
    return `<div class="field-row"><div class="field-key">${key}</div><div>${desc}${hint}<input data-config-key="${key}" value="${v}" /></div></div>`;
  }).join('');
  document.getElementById('planConfigForm').innerHTML = rows || '<div class="k">(no config keys)</div>';
}

function renderActiveBatches(rows) {
  const tbody = document.getElementById('activeBatchRows');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="k">(no active batches)</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td class="mono">${r.batch_id || '-'}</td>
      <td>${r.plan || '-'}</td>
      <td>${fmtTs(r.started_at)}</td>
      <td class="row">
        <button class="danger" onclick="killBatchInline('${r.batch_id}')">Kill</button>
        <button onclick="resumeBatchInline('${r.batch_id}')">Resume</button>
        <button onclick="viewBatchInline('${r.batch_id}')">Outputs</button>
      </td>
    </tr>
  `).join('');
}

function currentEffectivePlan() {
  const el = document.getElementById('planName');
  return el ? el.value : '';
}

function setEffectivePlan(planName) {
  const el = document.getElementById('planName');
  if (!el) return;
  el.value = planName;
}

function currentEffectiveScope() {
  const planName = currentEffectivePlan();
  return planScopes[planName] || 'shoulders';
}

function renderPlanSelectors() {
  const shoulderEl = document.getElementById('shoulderPlan');
  const armEl = document.getElementById('armPlan');
  const planEl = document.getElementById('planName');
  if (!shoulderEl || !armEl || !planEl) return;

  const currentShoulder = shoulderEl.value;
  const currentArm = armEl.value;
  shoulderEl.innerHTML = '';
  shoulderPlans.forEach(p => {
    const o = document.createElement('option');
    o.value = p;
    o.textContent = p;
    shoulderEl.appendChild(o);
  });
  if (!shoulderEl.options.length) {
    const o = document.createElement('option');
    o.value = '';
    o.textContent = '(no shoulder plans)';
    shoulderEl.appendChild(o);
  }
  if (currentShoulder && shoulderPlans.includes(currentShoulder)) {
    shoulderEl.value = currentShoulder;
  }

  const shoulder = shoulderEl.value;
  const binding = shoulderArmBindings[shoulder] || { arms: [] };
  const armRows = Array.isArray(binding.arms) ? binding.arms : [];
  armEl.innerHTML = '';
  if (armRows.length) {
    const shoulderOpt = document.createElement('option');
    shoulderOpt.value = '__shoulder__';
    shoulderOpt.textContent = `${shoulder} (run shoulder)`;
    armEl.appendChild(shoulderOpt);
    armRows.forEach(row => {
      const name = row && row.name ? String(row.name) : '';
      if (!name) return;
      const o = document.createElement('option');
      o.value = name;
      o.textContent = name;
      armEl.appendChild(o);
    });
    const armValues = [...armEl.options].map(o => o.value);
    if (currentArm && armValues.includes(currentArm)) {
      armEl.value = currentArm;
    } else {
      armEl.value = '__shoulder__';
    }
    armEl.style.display = '';
  } else {
    armEl.style.display = 'none';
  }

  const selected = armEl.style.display !== 'none' && armEl.value && armEl.value !== '__shoulder__'
    ? armEl.value
    : shoulder;
  setEffectivePlan(selected || shoulder);
  if (selected || shoulder) applyPlanDefault(selected || shoulder);
}

function fillBatchSelect(selectId, ids, labels) {
  const el = document.getElementById(selectId);
  el.innerHTML = '';
  ids.forEach(b => {
    const o = document.createElement('option');
    o.value = b;
    o.textContent = (labels && labels[b]) ? labels[b] : b;
    el.appendChild(o);
  });
  if (!el.options.length) {
    const o = document.createElement('option');
    o.value = '';
    o.textContent = '(none)';
    el.appendChild(o);
  }
}

async function refreshOptions() {
  const res = await fetch('/api/control/options');
  const data = await res.json();
  planDefaults = data.plan_defaults || {};
  planStarters = data.plan_starters || {};
  planDefaultStarter = data.plan_default_starter || {};
  planInputs = data.plan_inputs || {};
  planInputFiles = data.plan_input_files || {};
  planScopes = data.plan_scopes || {};
  shoulderPlans = data.shoulder_plans || [];
  armPlans = data.arm_plans || [];
  shoulderArmBindings = data.shoulder_arm_bindings || {};
  batchLabels = data.batch_labels || {};
  renderActiveBatches(data.active_batches_meta || []);

  const ids = [];
  const seen = new Set();
  (data.active_batches || []).forEach(b => {
    if (!seen.has(b)) { ids.push(b); seen.add(b); }
  });
  (data.recent_batches || []).forEach(r => {
    const b = r.batch_id;
    if (b && !seen.has(b)) { ids.push(b); seen.add(b); }
  });
  fillBatchSelect('killBatch', ids, batchLabels);
  fillBatchSelect('resumeBatch', ids, batchLabels);
  fillBatchSelect('outputBatch', ids, batchLabels);

  const plan = document.getElementById('planName');
  const shoulder = document.getElementById('shoulderPlan');
  const arm = document.getElementById('armPlan');
  const starter = document.getElementById('starterFile');
  plan.innerHTML = '';
  const allPlans = data.plans || [];
  allPlans.forEach(p => {
    const o = document.createElement('option');
    o.value = p;
    o.textContent = p;
    plan.appendChild(o);
  });
  if (!shoulderPlans.length && armPlans.length) {
    shoulderPlans = [...armPlans];
  }
  shoulder.onchange = () => renderPlanSelectors();
  arm.onchange = () => renderPlanSelectors();
  plan.onchange = () => applyPlanDefault(plan.value);
  starter.onchange = () => {
    renderPlanInputHelp(currentEffectivePlan(), starter.value);
    renderConfigForm(currentEffectivePlan());
  };
  renderPlanSelectors();
}

function applyPlanDefault(planName) {
  const defaults = planDefaults[planName];
  if (defaults) {
    document.getElementById('planConfig').value = JSON.stringify(defaults, null, 2);
  }
  setQuickInputsFromConfig(planName, defaults || {});
  const starterSelect = document.getElementById('starterFile');
  starterSelect.innerHTML = '';
  const starters = planStarters[planName] || [];
  starters.forEach(s => {
    const o = document.createElement('option');
    o.value = s;
    o.textContent = s;
    starterSelect.appendChild(o);
  });
  if (starters.length) {
    starterSelect.value = planDefaultStarter[planName] || starters[0];
  }
  renderPlanInputHelp(planName, starterSelect.value);
  renderConfigForm(planName, defaults || {});
}

function renderPlanInputHelp(planName, starterFile) {
  const perStarter = planInputs[planName] || {};
  const inputList = perStarter[starterFile] || [];
  const help = {};
  inputList.forEach(x => { help[x.key] = x; });
  const defaults = planDefaults[planName] || {};
  const src = Object.keys(defaults).length ? defaults : help;
  const keys = visibleConfigKeys(planName, src);
  const rows = keys.map(key => {
    const h = help[key] || {};
    const opts = h.options ? `<div class="field-help">Options: ${h.options}</div>` : '';
    const desc = h.description ? `<div class="field-help">${h.description}</div>` : '';
    return `
      <div class="field-row">
        <div class="field-key">${key}</div>
        <div>${desc}${opts || '<div class="field-help">(no explicit options listed)</div>'}</div>
      </div>
    `;
  }).join('');
  document.getElementById('planInputs').innerHTML = rows
    ? `<div class="field-grid">${rows}</div>`
    : '<div class="k">(no input metadata found)</div>';
}

async function killBatchInline(batchId) {
  if (!batchId) return;
  showResult(await api('/api/control/kill_plan', { batch_id: batchId }));
  await refreshOptions();
}

async function resumeBatchInline(batchId) {
  if (!batchId) return;
  showResult(await api('/api/control/resume_plan', { batch_id: batchId }));
  await refreshOptions();
}

async function viewBatchInline(batchId) {
  document.getElementById('outputBatch').value = batchId;
  const outputMeta = document.getElementById('outputMeta');
  if (outputMeta) outputMeta.scrollIntoView({ behavior: 'smooth', block: 'start' });
  await loadBatchOutputs();
}

async function killPlan() {
  const batchId = document.getElementById('killBatch').value;
  if (!batchId) return;
  showResult(await api('/api/control/kill_plan', { batch_id: batchId }));
  await refreshOptions();
}

async function killAllActive() {
  showResult(await api('/api/control/kill_all_active', {}));
  await refreshOptions();
}

async function returnDefault() {
  showResult(await api('/api/control/return_default', {}));
  await refreshOptions();
}

async function resumePlan() {
  const batchId = document.getElementById('resumeBatch').value;
  if (!batchId) return;
  showResult(await api('/api/control/resume_plan', { batch_id: batchId }));
  await refreshOptions();
}

async function startPlan() {
  const planName = currentEffectivePlan();
  const planScope = currentEffectiveScope();
  const starterFile = document.getElementById('starterFile').value;
  const highPriority = document.getElementById('highPriority').checked;
  const quickRepoUrl = document.getElementById('quickRepoUrl').value;
  const quickClaim = document.getElementById('quickClaim').value;
  const cfg = { ...getCurrentConfig(), ...readFormConfig() };
  if (quickRepoUrl && quickRepoUrl.trim()) cfg.REPO_URL = quickRepoUrl.trim();
  if (quickClaim && quickClaim.trim()) cfg.CLAIMED_BEHAVIOR = quickClaim.trim();
  cfg.PRIORITY = highPriority ? 'high' : 'normal';
  cfg.PREEMPTIBLE = true;
  const configText = JSON.stringify(cfg);
  showResult(await api('/api/control/start_plan', {
    plan_name: planName,
    plan_scope: planScope,
    starter_file: starterFile,
    config_json: configText,
    repo_url: quickRepoUrl,
    claimed_behavior: quickClaim
  }));
  await refreshOptions();
}

function showOutputPreview(idx) {
  const row = outputFiles[idx];
  const box = document.getElementById('outputPreview');
  if (!row) {
    box.textContent = '(no preview)';
    return;
  }
  const header = [
    `file: ${row.relative_path}`,
    `mnt: ${row.mnt_path}`,
    `updated: ${fmtTs(row.updated_at)}`,
    `size: ${row.size_bytes} bytes`,
    '',
  ].join('\n');
  box.textContent = header + (row.preview || '(preview unavailable for this file type)');
}

async function loadBatchOutputs() {
  const batchId = document.getElementById('outputBatch').value;
  if (!batchId) return;
  const data = await api('/api/control/batch_outputs', { batch_id: batchId });
  showResult(data);
  if (!data.ok) {
    document.getElementById('outputMeta').textContent = '';
    document.getElementById('outputFiles').innerHTML = '';
    document.getElementById('outputPreview').textContent = '(no output)';
    return;
  }
  outputFiles = data.files || [];
  document.getElementById('outputMeta').textContent =
    `Batch ${data.batch_id} | Plan ${data.plan} | ${outputFiles.length} files | ${data.batch_mnt_path}`;
  const rows = outputFiles.map((f, i) => `
    <tr>
      <td class="mono">${f.relative_path}</td>
      <td>${f.size_bytes}</td>
      <td>${fmtTs(f.updated_at)}</td>
      <td><button onclick="showOutputPreview(${i})">Preview</button></td>
    </tr>
  `).join('');
  document.getElementById('outputFiles').innerHTML = `
    <table>
      <thead><tr><th>File</th><th>Bytes</th><th>Updated</th><th></th></tr></thead>
      <tbody>${rows || '<tr><td colspan="4" class="k">(no files)</td></tr>'}</tbody>
    </table>
  `;
  showOutputPreview(0);
}

refreshOptions().catch(console.error);
