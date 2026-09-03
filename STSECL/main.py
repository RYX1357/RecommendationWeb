"""STSECL training entry point.

Usage:
    python main.py BER --gpu=0
    python main.py CHI --gpu=0
"""

import json
import os
import random
import sqlite3

import numpy as np
import torch

from config import parse
from dataset import load_stsecl
from model import STSECL
from preprocess import build_dataset
from utils import contrastive_loss, evaluate, evaluate_auc, recommend_friends


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parse()
    setup_seed(args.seed)
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')

    # rebuild the dataset when the spatio-temporal thresholds differ from the
    # cached build (the web UI exposes exactly these knobs)
    params = {'delta1': args.delta1, 'delta2': args.delta2, 'rho': args.rho}
    meta = os.path.join('data', args.city, 'build_params.json')
    if not os.path.exists(meta):
        _cached = {}
    else:
        with open(meta) as f:
            _cached = json.load(f)
    if _cached != params:
        build_dataset(args.city, save=True, delta1=args.delta1,
                      delta2=args.delta2, rho=args.rho)

    data, train_friends, test_friends, all_friends_set = load_stsecl(
        args.city, args.split, args.seed)

    # move tensors to device
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device)
        elif isinstance(v, dict):
            data[k] = {kk: vv.to(device) for kk, vv in v.items()}
    train_friends = train_friends.to(device)
    test_friends = test_friends.to(device)

    nu = data['num_user']
    model = STSECL(args, nu, data['num_loc'], data['num_cat'], data['num_time']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    print(f'[{args.city}] users={nu} train_friends={train_friends.shape[1]} '
          f'test_friends={test_friends.shape[1]}')
    print(f'[{args.city}] params={sum(p.numel() for p in model.parameters())}')

    best_auc = 0.0
    patience = 0
    best_path = f'data/save_user_embedding/stsecl_{args.city}_best.pt'

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        z_em, z_hg = model(data)
        loss = contrastive_loss(z_em, z_hg, train_friends, args.tau, args.lam)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z_em, _ = model(data)
            auc = evaluate_auc(z_em, test_friends, all_friends_set, nu, device)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f'epoch {epoch:4d} | loss {loss.item():.4f}')

        if auc > best_auc:
            best_auc = auc
            patience = 0
            os.makedirs('data/save_user_embedding', exist_ok=True)
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f'early stop at epoch {epoch} '
                      f'(no improvement for {args.patience} epochs)')
                break

    # ---- final metrics + friend recommendations, on the best model ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    with torch.no_grad():
        z_em, _ = model(data)
        auc, ap, top_k = evaluate(
            z_em, test_friends, all_friends_set, train_friends, nu, device)
    top10 = top_k[2]
    print(f'[{args.city}] FINAL -> AUC {auc:.4f} AP {ap:.4f} Top@10 {top10:.4f}')

    out_dir = f'data/{args.city}'
    metrics = {'AUC': round(float(auc), 4), 'AP': round(float(ap), 4),
               'Top@10': round(float(top10), 4)}
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'task': 'STSECL', 'city': args.city, 'metrics': metrics}, f)

    db_path = os.path.join(out_dir, 'recommendations.db')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('DROP TABLE IF EXISTS recs')
    cur.execute('CREATE TABLE recs (user INTEGER, rank INTEGER, friend INTEGER, score REAL)')
    buf = []
    with torch.no_grad():
        for row in recommend_friends(z_em, train_friends, nu, device, k=args.topk):
            buf.append(row)
            if len(buf) >= 20000:
                cur.executemany('INSERT INTO recs VALUES (?,?,?,?)', buf)
                buf.clear()
    if buf:
        cur.executemany('INSERT INTO recs VALUES (?,?,?,?)', buf)
    cur.execute('CREATE INDEX idx_recs_user ON recs(user)')
    conn.commit()
    conn.close()
    print(f'[{args.city}] saved recommendations -> {db_path}')


if __name__ == '__main__':
    main()
