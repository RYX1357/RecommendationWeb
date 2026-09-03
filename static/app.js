/* Frontend for the recommendation web app. Wires the SPA to the Flask API. */

const state = {
    config: null,       // {tasks: {STSECL: {...}, POIRec: {...}}}
    task: null,         // current task key ('STSECL' | 'POIRec')
    dataset: null,      // current dataset key
    jobId: null,
    pollTimer: null,
    result: null,       // latest /api/result payload
    running: false,     // whether a training job is currently running
};

let map = null;         // Leaflet map instance (POIRec only)
let markerLayer = null; // layer group for user + POI markers

const $ = (sel) => document.querySelector(sel);

function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (children) for (const c of [].concat(children)) {
        node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    }
    return node;
}

async function api(path, opts) {
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `请求失败 (${res.status})`);
    return data;
}

function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

/* ---- 1. task selection ---- */
function renderTasks() {
    const box = $('#taskCards');
    box.innerHTML = '';
    const tasks = state.config.tasks;
    Object.entries(tasks).forEach(([key, cfg]) => {
        const card = el('div', { class: 'task-card' }, [
            el('div', { class: 't-name' }, cfg.name),
            el('div', { class: 't-desc' }, `数据集: ${cfg.datasets.join(', ')}`),
        ]);
        card.addEventListener('click', () => selectTask(key));
        if (key === state.task) card.classList.add('active');
        box.appendChild(card);
    });
}

function selectTask(key) {
    state.task = key;
    state.dataset = state.config.tasks[key].datasets[0];
    renderTasks();
    renderDatasets();
    renderParams();
}

/* ---- 2. dataset + params ---- */
function renderDatasets() {
    const sel = $('#datasetSelect');
    sel.innerHTML = '';
    state.config.tasks[state.task].datasets.forEach((d) => {
        sel.appendChild(el('option', { value: d }, d));
    });
    sel.value = state.dataset;
    sel.onchange = () => { state.dataset = sel.value; };
}

function paramControl(p) {
    const wrap = el('div', { class: 'param-field' });
    wrap.appendChild(el('label', {}, p.label));
    let ctrl;
    if (p.type === 'select') {
        ctrl = el('select', {});
        p.options.forEach((o) => {
            const opt = el('option', { value: o }, o);
            if (String(o) === String(p.default)) opt.selected = true;
            ctrl.appendChild(opt);
        });
    } else {
        ctrl = el('input', { type: p.type === 'int' || p.type === 'float' ? 'number' : 'text',
                             step: p.type === 'float' ? 'any' : '1' });
        ctrl.value = p.default;
    }
    ctrl.dataset.name = p.name;
    wrap.appendChild(ctrl);
    return wrap;
}

function renderParams() {
    const grid = $('#paramsGrid');
    grid.innerHTML = '';
    state.config.tasks[state.task].params.forEach((p) => grid.appendChild(paramControl(p)));
}

function collectParams() {
    const out = {};
    document.querySelectorAll('#paramsGrid [data-name]').forEach((ctrl) => {
        let v = ctrl.value;
        if (v === '' || v === null || v === undefined) return;
        out[ctrl.dataset.name] = v;
    });
    return out;
}

/* ---- 3. run + poll ---- */
async function run() {
    const btn = $('#runBtn');
    btn.disabled = true;
    btn.textContent = '推荐中…';
    hideResults();

    try {
        const { job_id } = await api('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task: state.task,
                dataset: state.dataset,
                params: collectParams(),
            }),
        });
        state.jobId = job_id;
        state.running = true;
        poll();
    } catch (err) {
        alert('提交失败: ' + err.message);
        btn.disabled = false;
        btn.textContent = '推荐';
    }
}

function poll() {
    clearInterval(state.pollTimer);
    state.pollTimer = setInterval(async () => {
        try {
            const st = await api(`/api/status/${state.jobId}`);
            if (st.status === 'running') {
                // still training, wait silently
            } else if (st.status === 'done') {
                clearInterval(state.pollTimer);
                await loadResult();
                finishRun();
            } else {
                clearInterval(state.pollTimer);
                alert('运行出错，请检查日志');
                finishRun();
            }
        } catch (err) {
            clearInterval(state.pollTimer);
            alert('状态查询失败: ' + err.message);
            finishRun();
        }
    }, 1500);
}

function finishRun() {
    state.running = false;
    const btn = $('#runBtn');
    btn.disabled = false;
    btn.textContent = '推荐';
}

/* ---- 4. results ---- */
async function loadResult() {
    const res = await api(`/api/result/${state.jobId}`);
    state.result = res;
    renderTable(res.header, [], 0, '请输入用户 ID，点击「查询」查看推荐结果');
    $('#resultCard').hidden = false;
}

function renderTable(header, rows, totalRows, emptyMsg) {
    const table = $('#recTable');
    const thead = table.querySelector('thead');
    const tbody = table.querySelector('tbody');
    thead.innerHTML = '';
    tbody.innerHTML = '';

    if (!header || header.length === 0) {
        thead.appendChild(el('tr', {}, el('th', {}, '推荐结果')));
        return;
    }
    const hr = el('tr', {});
    header.forEach((h) => hr.appendChild(el('th', {}, h)));
    thead.appendChild(hr);

    if (!rows || rows.length === 0) {
        tbody.appendChild(el('tr', {}, el('td', { colspan: String(header.length) },
            emptyMsg || '暂无推荐结果')));
    } else {
        rows.forEach((row) => {
            const tr = el('tr', {});
            header.forEach((h) => tr.appendChild(el('td', {}, String(row[h] ?? ''))));
            tbody.appendChild(tr);
        });
    }

    $('#recInfo').textContent = (rows && rows.length)
        ? `共 ${totalRows ?? rows.length} 条推荐`
        : '';
}

async function queryUser() {
    if (!state.jobId) return;
    const user = $('#userInput').value.trim();
    if (user === '') {
        alert('请输入用户 ID');
        return;
    }
    const top = $('#topInput').value.trim() || '10';
    try {
        const res = await api(`/api/recommendations/${state.jobId}?user=${encodeURIComponent(user)}&top=${encodeURIComponent(top)}`);
        renderTable(res.header, res.rows, res.rows.length, '该用户暂无推荐结果');
        if (state.task === 'POIRec') {
            renderPoiMap(res.rows, res.user_location);
        } else {
            hidePoiMap();
        }
    } catch (err) {
        alert(err.message);
    }
}

function hideResults() {
    $('#resultCard').hidden = true;
    $('#recTable thead').innerHTML = '';
    $('#recTable tbody').innerHTML = '';
    $('#recInfo').textContent = '';
    state.result = null;
    hidePoiMap();
}

/* ---- 5. POI map (POIRec only) ---- */
function ensureMap() {
    if (map) return;
    map = L.map('map', { zoomControl: true });
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors',
    }).addTo(map);
    markerLayer = L.layerGroup().addTo(map);
    const legend = $('#mapLegend');
    legend.hidden = false;
    legend.innerHTML =
        '<span class="lg-user"></span> 目标用户' +
        '<span class="lg-poi"></span> 推荐地点';
}

function renderPoiMap(rows, userLocation) {
    if (typeof L === 'undefined') return;
    $('#mapWrap').hidden = false;
    ensureMap();
    map.invalidateSize();
    markerLayer.clearLayers();

    const bounds = [];
    (rows || []).forEach((r) => {
        const lat = parseFloat(r.lat), lng = parseFloat(r.lng);
        if (!Number.isFinite(lat) || !Number.isFinite(lng)) return;
        bounds.push([lat, lng]);
        const m = L.circleMarker([lat, lng], {
            radius: 6, color: '#1d4ed8', weight: 2,
            fillColor: '#3b82f6', fillOpacity: 0.85,
        });
        m.bindPopup(`<b>${escapeHtml(r.name || r.poi)}</b><br>` +
            `POI #${escapeHtml(r.poi)} · 评分 ${escapeHtml(r.score)}`);
        markerLayer.addLayer(m);
    });

    if (userLocation) {
        const lat = parseFloat(userLocation.lat), lng = parseFloat(userLocation.lng);
        if (Number.isFinite(lat) && Number.isFinite(lng)) {
            bounds.push([lat, lng]);
            const u = L.circleMarker([lat, lng], {
                radius: 9, color: '#b91c1c', weight: 3,
                fillColor: '#ef4444', fillOpacity: 0.95,
            });
            u.bindPopup(`<b>目标用户</b><br>ID: ${escapeHtml(userLocation.raw_user_id ?? '')}`);
            markerLayer.addLayer(u);
        }
    }

    if (bounds.length) {
        map.fitBounds(L.latLngBounds(bounds), { padding: [40, 40], maxZoom: 16 });
    }
}

function hidePoiMap() {
    $('#mapWrap').hidden = true;
    if (markerLayer) markerLayer.clearLayers();
}

/* ---- init ---- */
async function init() {
    try {
        state.config = await api('/api/config');
        state.task = Object.keys(state.config.tasks)[0];
        renderTasks();
        renderDatasets();
        renderParams();
    } catch (err) {
        alert('无法加载配置: ' + err.message);
        return;
    }

    $('#runBtn').addEventListener('click', run);
    $('#queryBtn').addEventListener('click', queryUser);
    $('#userInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') queryUser(); });
    $('#topInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') queryUser(); });
}

document.addEventListener('DOMContentLoaded', init);

/* Interrupt any running training job when the page is closed or refreshed. */
window.addEventListener('pagehide', () => {
    if (state.running) {
        navigator.sendBeacon('/api/cancel');
    }
});
