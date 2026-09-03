/* Static demo: renders precomputed results from data.json (no backend). */

let DATA = null;
let state = { task: null, dataset: null, user: null };
let map = null;
let markerLayer = null;

const $ = (sel) => document.querySelector(sel);

// column defs per task: [key, label]
const COLUMNS = {
    POIRec: [['rank', '排名'], ['poi', 'POI'], ['name', '类别'],
             ['score', '评分'], ['visit_count', '访问数']],
    STSECL: [['rank', '排名'], ['friend', '好友'], ['score', '评分']],
};

function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

async function init() {
    DATA = await (await fetch('data.json')).json();
    const tasks = DATA.tasks;
    const tabs = $('#taskTabs');
    Object.entries(tasks).forEach(([key, cfg]) => {
        const d = document.createElement('div');
        d.className = 'task-tab';
        d.innerHTML = `<div class="t-name">${escapeHtml(cfg.name)}</div>`;
        d.addEventListener('click', () => selectTask(key));
        tabs.appendChild(d);
    });
    selectTask(Object.keys(tasks)[0]);
}

function selectTask(key) {
    state.task = key;
    document.querySelectorAll('.task-tab').forEach((el, i) => {
        el.classList.toggle('active', Object.keys(DATA.tasks)[i] === key);
    });
    const ds = Object.keys(DATA.tasks[key].datasets);
    state.dataset = ds[0];
    const sel = $('#datasetSelect');
    sel.innerHTML = '';
    ds.forEach((d) => {
        const o = document.createElement('option');
        o.value = d; o.textContent = d;
        sel.appendChild(o);
    });
    sel.value = state.dataset;
    selectDataset();
}

function selectDataset() {
    state.dataset = $('#datasetSelect').value;
    const entry = DATA.tasks[state.task].datasets[state.dataset];
    renderUsers(entry.users);
    state.user = entry.users[0]?.user;
    if (entry.users.length) {
        $('#userSelect').value = String(state.user);
        renderResult(entry.users[0]);
        $('#resultCard').hidden = false;
    } else {
        $('#resultCard').hidden = true;
    }
}

function renderUsers(users) {
    const sel = $('#userSelect');
    sel.innerHTML = '';
    users.forEach((u) => {
        const o = document.createElement('option');
        o.value = String(u.user);
        const label = u.raw_id != null ? `用户 #${u.user} (ID ${u.raw_id})` : `用户 #${u.user}`;
        o.textContent = label;
        sel.appendChild(o);
    });
    sel.value = String(state.user ?? users[0]?.user);
}

function selectUser() {
    state.user = parseInt($('#userSelect').value, 10);
    const entry = DATA.tasks[state.task].datasets[state.dataset];
    const u = entry.users.find((x) => x.user === state.user);
    if (u) renderResult(u);
}

function renderResult(u) {
    const cols = COLUMNS[state.task];
    const table = $('#recTable');
    const thead = table.querySelector('thead');
    const tbody = table.querySelector('tbody');
    thead.innerHTML = '';
    tbody.innerHTML = '';

    const hr = document.createElement('tr');
    cols.forEach(([k, label]) => {
        const th = document.createElement('th');
        th.textContent = label;
        hr.appendChild(th);
    });
    thead.appendChild(hr);

    u.recs.forEach((r) => {
        const tr = document.createElement('tr');
        cols.forEach(([k]) => {
            const td = document.createElement('td');
            td.textContent = r[k] ?? '';
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });

    if (state.task === 'POIRec') {
        renderMap(u);
    } else {
        hideMap();
    }
}

function ensureMap() {
    if (map) return;
    map = L.map('map');
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19, attribution: '&copy; OpenStreetMap contributors',
    }).addTo(map);
    markerLayer = L.layerGroup().addTo(map);
}

function renderMap(u) {
    if (typeof L === 'undefined') return;
    $('#mapWrap').hidden = false;
    ensureMap();
    markerLayer.clearLayers();
    const bounds = [];
    u.recs.forEach((r) => {
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
    if (u.center) {
        const lat = parseFloat(u.center.lat), lng = parseFloat(u.center.lng);
        if (Number.isFinite(lat) && Number.isFinite(lng)) {
            bounds.push([lat, lng]);
            const c = L.circleMarker([lat, lng], {
                radius: 9, color: '#b91c1c', weight: 3,
                fillColor: '#ef4444', fillOpacity: 0.95,
            });
            c.bindPopup(`<b>目标用户</b><br>ID: ${escapeHtml(u.raw_id ?? u.user)}`);
            markerLayer.addLayer(c);
        }
    }
    if (bounds.length) map.fitBounds(L.latLngBounds(bounds), { padding: [40, 40], maxZoom: 16 });
    setTimeout(() => map.invalidateSize(), 50);
}

function hideMap() {
    $('#mapWrap').hidden = true;
    if (markerLayer) markerLayer.clearLayers();
}

document.addEventListener('DOMContentLoaded', () => {
    $('#datasetSelect').addEventListener('change', selectDataset);
    $('#userSelect').addEventListener('change', selectUser);
    init();
});
