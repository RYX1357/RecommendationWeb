"""Recommendation web app: STSECL (friend recommendation) + POIRec (POI recommendation).

Runs each project's `main.py` as a background subprocess and exposes a small JSON
API consumed by the single-page frontend in `templates/index.html`.

Endpoints:
    GET  /api/config                       -> tasks, datasets, tunable params
    POST /api/run                          -> start a training job
    GET  /api/status/<job_id>              -> running/done/error + log tail
    GET  /api/result/<job_id>              -> metrics + recommendation sample
    GET  /api/recommendations/<job_id>     -> recommendations for one user (?user=)
"""

import json
import os
import sqlite3
import subprocess
import sys
import uuid

from flask import Flask, jsonify, request, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TASKS = {
    'STSECL': {
        'name': '朋友推荐 (STSECL)',
        'dir': os.path.join(BASE_DIR, 'STSECL'),
        'datasets': ['BER', 'CHI', 'NYC', 'JK', 'KL', 'SP'],
        'params': [
            {'name': 'delta1', 'label': 'δ1 距离阈值 (km)', 'type': 'float', 'default': 1.0},
            {'name': 'delta2', 'label': 'δ2 距离阈值 (km)', 'type': 'float', 'default': 1.0},
            {'name': 'rho', 'label': 'ρ 共访 POI 阈值', 'type': 'int', 'default': 2},
        ],
    },
    'POIRec': {
        'name': '兴趣点推荐 (POIRec)',
        'dir': os.path.join(BASE_DIR, 'POIRec'),
        'datasets': ['BER', 'CHI'],
        'params': [
            {'name': 'delta1', 'label': 'δ1 U-L 半径 (km)', 'type': 'float', 'default': 1.0},
            {'name': 'delta2', 'label': 'δ2 U-U 半径 (km)', 'type': 'float', 'default': 15.0},
            {'name': 'delta3', 'label': 'δ3 L-L 同类半径 (km)', 'type': 'float', 'default': 1.0},
            {'name': 'delta4', 'label': 'δ4 L-L 共访半径 (km)', 'type': 'float', 'default': 1.0},
        ],
    },
}

JOBS = {}

# check-in + POI CSVs used to recover a target user's activity center (lat, lng)
POIREC_CITY_FILES = {
    'BER': {'checkin': 'check_in_berlin_user_in_friend.csv',
            'poi': 'berlin_poi_incheckin_and_friend.csv'},
    'CHI': {'checkin': 'check_in_chi_user_in_friend.csv',
            'poi': 'chi_poi_incheckin_and_friend.csv'},
}

_USER_CENTERS = {}


app = Flask(__name__)


def _any_running():
    return any(j['proc'].poll() is None for j in JOBS.values())


def _user_centers(task, dataset):
    """Return {user_index: (lat, lng, raw_userid)} activity centers for POIRec.

    Matches preprocess.py: the activity center is the mean (lat, lng) over the
    user's *unique* visited POIs. ``user_index`` is the 0-based index used in
    the recommendations table (sorted unique check-in user ids).
    """
    key = (task, dataset)
    if key in _USER_CENTERS:
        return _USER_CENTERS[key]

    centers = {}
    if task == 'POIRec' and dataset in POIREC_CITY_FILES:
        import pandas as pd
        data_dir = os.path.join(TASKS[task]['dir'], 'data', dataset)
        files = POIREC_CITY_FILES[dataset]
        try:
            ci = pd.read_csv(os.path.join(data_dir, files['checkin']))
            poi = pd.read_csv(os.path.join(data_dir, files['poi']))
            lat_map = dict(zip(poi['id'].astype(int), poi['lat'].astype(float)))
            lng_map = dict(zip(poi['id'].astype(int), poi['lng'].astype(float)))
            user_ids = sorted(ci['userid'].unique())
            df = ci[['userid', 'placeid']].copy()
            df['lat'] = df['placeid'].map(lat_map)
            df['lng'] = df['placeid'].map(lng_map)
            df = df.dropna(subset=['lat', 'lng']).drop_duplicates(
                subset=['userid', 'placeid'])
            g = df.groupby('userid')[['lat', 'lng']].mean()
            for i, u in enumerate(user_ids):
                if u in g.index:
                    centers[i] = (float(g.at[u, 'lat']), float(g.at[u, 'lng']), int(u))
        except Exception:
            centers = {}

    _USER_CENTERS[key] = centers
    return centers


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/config')
def api_config():
    out = {}
    for t, cfg in TASKS.items():
        out[t] = {'name': cfg['name'], 'datasets': cfg['datasets'], 'params': cfg['params']}
    return jsonify({'tasks': out})


@app.route('/api/run', methods=['POST'])
def api_run():
    body = request.get_json(silent=True) or {}
    task = body.get('task')
    dataset = body.get('dataset')
    params = body.get('params') or {}

    if task not in TASKS:
        return jsonify({'error': '未知任务'}), 400
    if dataset not in TASKS[task]['datasets']:
        return jsonify({'error': '未知数据集'}), 400
    if _any_running():
        return jsonify({'error': '已有任务在运行，请等待完成'}), 409

    job_id = uuid.uuid4().hex[:12]
    log_dir = os.path.join(BASE_DIR, 'results', job_id)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.txt')

    cmd = [sys.executable, '-u', 'main.py', dataset]
    for p in TASKS[task]['params']:
        name = p['name']
        val = params.get(name)
        if val not in (None, ''):
            cmd += [f'--{name}', str(val)]

    logf = open(log_path, 'w', encoding='utf-8')
    proc = subprocess.Popen(
        cmd, cwd=TASKS[task]['dir'], stdout=logf, stderr=subprocess.STDOUT)
    JOBS[job_id] = {'proc': proc, 'log_path': log_path, 'logf': logf,
                    'task': task, 'dataset': dataset}
    return jsonify({'job_id': job_id})


@app.route('/api/cancel', methods=['POST'])
def api_cancel():
    for job in list(JOBS.values()):
        proc = job['proc']
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        try:
            job['logf'].close()
        except Exception:
            pass
    JOBS.clear()
    return jsonify({'ok': True})


@app.route('/api/status/<job_id>')
def api_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': '未知任务'}), 404
    proc = job['proc']
    running = proc.poll() is None
    log_tail = ''
    try:
        with open(job['log_path'], 'r', encoding='utf-8', errors='replace') as f:
            log_tail = ''.join(f.readlines()[-120:])
    except Exception:
        pass
    status = 'running' if running else ('done' if proc.returncode == 0 else 'error')
    return jsonify({'status': status, 'log': log_tail,
                    'returncode': None if running else proc.returncode})


REC_HEADERS = {
    'STSECL': ['user', 'rank', 'friend', 'score'],
    'POIRec': ['user', 'rank', 'poi', 'category', 'name', 'lat', 'lng',
              'visit_count', 'score'],
}


def _db_path(task, dataset):
    return os.path.join(TASKS[task]['dir'], 'data', dataset, 'recommendations.db')


def _tsv_path(task, dataset):
    return os.path.join(TASKS[task]['dir'], 'data', dataset, 'recommendations.tsv')


def _query_recs(task, dataset, user=None, top=None):
    """Return (header, rows) for a user; a sample when user is None.

    Rows are lists of strings. Prefers the SQLite store (which holds every
    candidate) and falls back to the legacy TSV.
    """
    header = REC_HEADERS[task]
    db = _db_path(task, dataset)
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        if user is None:
            cur.execute('SELECT * FROM recs LIMIT 100')
        elif top is None:
            cur.execute('SELECT * FROM recs WHERE user=? ORDER BY rank', (user,))
        else:
            cur.execute('SELECT * FROM recs WHERE user=? AND rank<=? ORDER BY rank',
                        (user, top))
        rows = [[str(x) for x in r] for r in cur.fetchall()]
        conn.close()
        return header, rows

    rt = _tsv_path(task, dataset)
    if not os.path.exists(rt):
        return header, []
    with open(rt, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    if not lines:
        return header, []
    tsv_header = lines[0].split('\t')
    all_rows = [l.split('\t') for l in lines[1:] if l.strip()]
    if user is None:
        rows = all_rows[:100]
    else:
        rows = [r for r in all_rows if int(r[0]) == user]
        if top is not None:
            rows = [r for r in rows if int(r[1]) <= top]
    return tsv_header, rows


def _count_recs(task, dataset):
    db = _db_path(task, dataset)
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        n = conn.execute('SELECT COUNT(*) FROM recs').fetchone()[0]
        conn.close()
        return n
    rt = _tsv_path(task, dataset)
    if not os.path.exists(rt):
        return 0
    with open(rt, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1


@app.route('/api/result/<job_id>')
def api_result(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': '未知任务'}), 404
    task, dataset = job['task'], job['dataset']

    metrics = {}
    rj = os.path.join(TASKS[task]['dir'], 'data', dataset, 'results.json')
    if os.path.exists(rj):
        with open(rj, 'r', encoding='utf-8') as f:
            metrics = json.load(f).get('metrics', {})

    header, rows = _query_recs(task, dataset)
    sample = [dict(zip(header, r)) for r in rows] if header else []
    return jsonify({
        'task': task,
        'dataset': dataset,
        'metrics': metrics,
        'header': header,
        'sample': sample,
        'total_rows': _count_recs(task, dataset),
    })


@app.route('/api/recommendations/<job_id>')
def api_recommendations(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': '未知任务'}), 404
    task, dataset = job['task'], job['dataset']

    user = request.args.get('user', type=int)
    if user is None:
        header, rows = _query_recs(task, dataset)
        return jsonify({'header': header, 'rows': [dict(zip(header, r)) for r in rows]})

    top = request.args.get('top', type=int)
    header, rows = _query_recs(task, dataset, user=user, top=top)
    payload = {'header': header, 'rows': [dict(zip(header, r)) for r in rows],
               'user': user}
    if task == 'POIRec':
        loc = _user_centers(task, dataset).get(user)
        if loc:
            payload['user_location'] = {'lat': loc[0], 'lng': loc[1],
                                        'raw_user_id': loc[2]}
    return jsonify(payload)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
