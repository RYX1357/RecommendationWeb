"""POIRec training entry point.

Usage:
    python main.py BER --gpu=0
"""

import json
import os
import random
import sqlite3

import numpy as np
import torch

from config import parse
from dataset import load_poi
from model import POIRec
from preprocess import build_dataset
from utils import total_loss, evaluate, recommend


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parse()
    setup_seed(args.seed)
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')

    # rebuild the dataset when the geographic thresholds differ from the cached
    # build (the web UI exposes exactly these knobs)
    params = {'delta1': args.delta1, 'delta2': args.delta2,
              'delta3': args.delta3, 'delta4': args.delta4}
    meta = os.path.join('data', args.city, 'build_params.json')
    if not os.path.exists(meta):
        _cached = {}
    else:
        with open(meta) as f:
            _cached = json.load(f)
    if _cached != params:
        build_dataset(args.city, save=True, delta1=args.delta1, delta2=args.delta2,
                      delta3=args.delta3, delta4=args.delta4, split=args.split)

    data, user_visited, test_checkins = load_poi(args.city, kmax=args.kmax)
    nu = data['num_user']
    nl = data['num_loc']

    # move tensors to device (test_checkins stays on CPU for evaluation)
    data['user_feat'] = data['user_feat'].to(device)
    data['loc_feat'] = data['loc_feat'].to(device)
    data['net_edge_index'] = data['net_edge_index'].to(device)
    data['mp_edge'] = {p: e.to(device) for p, e in data['mp_edge'].items()}
    data['mp_sub'] = {p: {k: e.to(device) for k, e in sub.items()}
                      for p, sub in data['mp_sub'].items()}

    model = POIRec(args, nu, nl).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    print(f'[{args.city}] users={nu} locs={nl} test_checkins={test_checkins.shape[0]}')
    print(f'[{args.city}] params={sum(p.numel() for p in model.parameters())}')

    best_ndcg = 0.0
    patience = 0
    best_path = f'data/save_user_embedding/poirec_{args.city}_best.pt'

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = total_loss(out, args.tau, args.beta, args.lam, args.rho1, args.rho2)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data)
            _, ndcg = evaluate(out['me'], user_visited, test_checkins,
                               nu, nl, device, k_list=(10,))
        ndcg10 = ndcg[10]

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f'epoch {epoch:4d} | loss {loss.item():.4f}')

        if ndcg10 > best_ndcg:
            best_ndcg = ndcg10
            patience = 0
            os.makedirs('data/save_user_embedding', exist_ok=True)
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f'early stop at epoch {epoch} '
                      f'(no improvement for {args.patience} epochs)')
                break

    # ---- final metrics + recommendations, on the best model ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    with torch.no_grad():
        out = model(data)
        recall, ndcg = evaluate(out['me'], user_visited, test_checkins,
                                nu, nl, device, k_list=(5, 10, 20))

    out_dir = os.path.join('data', args.city)
    metrics = {f'Recall@{k}': round(float(recall[k]), 4) for k in (5, 10, 20)}
    metrics.update({f'NDCG@{k}': round(float(ndcg[k]), 4) for k in (5, 10, 20)})
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'task': 'POIRec', 'city': args.city, 'metrics': metrics}, f)

    db_path = os.path.join(out_dir, 'recommendations.db')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('DROP TABLE IF EXISTS recs')
    cur.execute('CREATE TABLE recs (user INTEGER, rank INTEGER, poi INTEGER, '
                'category INTEGER, name TEXT, lat REAL, lng REAL, visits INTEGER, score REAL)')
    buf = []
    with torch.no_grad():
        for row in recommend(out['me'], user_visited, data['loc_cat'], data['loc_total'],
                             data['loc_name'], data['poi_lat'], data['poi_lng'],
                             nu, nl, device, k=args.topk):
            buf.append(row)
            if len(buf) >= 20000:
                cur.executemany('INSERT INTO recs VALUES (?,?,?,?,?,?,?,?,?)', buf)
                buf.clear()
    if buf:
        cur.executemany('INSERT INTO recs VALUES (?,?,?,?,?,?,?,?,?)', buf)
    cur.execute('CREATE INDEX idx_recs_user ON recs(user)')
    conn.commit()
    conn.close()
    print(f'[{args.city}] saved recommendations -> {db_path}')


if __name__ == '__main__':
    main()
