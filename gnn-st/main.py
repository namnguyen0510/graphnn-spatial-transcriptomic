import os
import json
import random
import scanpy as sc
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import from_scipy_sparse_matrix

import optuna
import argparse
from optuna.samplers import TPESampler


from models import MODEL_CLASSES  # Includes GCN, GAT, GraphSAGE, etc.

# ========== CLI Args ==========
parser = argparse.ArgumentParser()
parser.add_argument('--adata_dir', type = str)
parser.add_argument('--model', type=str, default='appnp', choices=['gcn', 'gat', 'sage', 'gin', 'appnp'])
parser.add_argument('--n_trials', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_epochs_search', type=int, default=10)
parser.add_argument('--num_epochs_eval', type=int, default=100)
parser.add_argument('--study_name', type=str, default='gnn_study')
parser.add_argument('--db_path', type=str, default='optuna_study.db')
args = parser.parse_args()

# ========== 0. Config ==========
selected_model = args.model
num_trials = args.n_trials
num_epochs_search = args.num_epochs_search
num_epochs_eval = args.num_epochs_eval
batch_size = args.batch_size

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ========== 1. Load Dataset ==========
adata_dir = args.adata_dir
adata_paths = [os.path.join(adata_dir, x) for x in sorted(os.listdir(adata_dir))]
pnames = sorted(os.listdir(adata_dir))
for train_pid in range(len(adata_paths)):

    results_dir = f'results_pid_{pnames[train_pid]}_{selected_model}'
    os.makedirs(results_dir, exist_ok=True)

    adata = sc.read_h5ad(adata_paths[train_pid])
    X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    Y = LabelEncoder().fit_transform(adata.obs['Region'])

    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    edge_index, _ = from_scipy_sparse_matrix(adata.obsp['connectivities'])

    train_idx, eval_idx = train_test_split(np.arange(len(Y)), test_size=0.2, random_state=42, stratify=Y)

    data = Data(x=torch.tensor(X, dtype=torch.float),
                edge_index=edge_index,
                y=torch.tensor(Y, dtype=torch.long))
    data.train_mask = torch.zeros(len(Y), dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.eval_mask = torch.zeros(len(Y), dtype=torch.bool)
    data.eval_mask[eval_idx] = True

    # ========== 2. Evaluation Function ==========
    def evaluate(model, loader, loss_fn):
        model.eval()
        all_preds, all_labels, total_loss = [], [], 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to('cpu')
                out = model(batch.x, batch.edge_index)[batch.batch_size:]
                loss = loss_fn(out, batch.y[batch.batch_size:])
                total_loss += loss.item() * batch.batch_size
                all_preds.append(out.argmax(dim=1))
                all_labels.append(batch.y[batch.batch_size:])
        y_true = torch.cat(all_labels)
        y_pred = torch.cat(all_preds)
        acc = accuracy_score(y_true.numpy(), y_pred.numpy())
        return total_loss / len(loader.dataset), acc

    # ========== 3. Objective Function for Optuna ==========
    def objective(trial):
        hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256, 512, 1024])
        dropout = trial.suggest_float('dropout', 0.1, 0.6)
        lr = trial.suggest_float('lr', 1e-4, 1e-2)
        num_layers = trial.suggest_int('num_layers', 2, 5)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD', 'RMSprop'])

        model = MODEL_CLASSES[selected_model](
            in_channels=X.shape[1],
            hidden_channels=hidden_channels,
            out_channels=np.unique(Y).shape[0],
            num_layers=num_layers,
            dropout=dropout
        )

        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        loss_fn = nn.CrossEntropyLoss()

        train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                                    num_neighbors=[3], batch_size=batch_size, shuffle=True, num_workers=0)
        eval_loader = NeighborLoader(data, input_nodes=data.eval_mask,
                                    num_neighbors=[3], batch_size=batch_size, shuffle=False, num_workers=0)

        for epoch in range(num_epochs_search):  
            model.train()
            for batch in train_loader:
                batch = batch.to('cpu')
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)[batch.batch_size:]
                loss = loss_fn(out, batch.y[batch.batch_size:])
                loss.backward()
                optimizer.step()

        _, eval_acc = evaluate(model, eval_loader, loss_fn)
        return eval_acc
    # ========== 3. Objective Function for Optuna ==========
    def objective(trial):
        hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.6)
        lr = trial.suggest_float('lr', 1e-4, 1e-2)
        num_layers = trial.suggest_int('num_layers', 2, 5)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD', 'RMSprop'])

        model = MODEL_CLASSES[selected_model](
            in_channels=X.shape[1],
            hidden_channels=hidden_channels,
            out_channels=np.unique(Y).shape[0],
            num_layers=num_layers,
            dropout=dropout
        )

        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        loss_fn = nn.CrossEntropyLoss()

        train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                                    num_neighbors=[3], batch_size=batch_size, shuffle=True, num_workers=0)
        eval_loader = NeighborLoader(data, input_nodes=data.eval_mask,
                                    num_neighbors=[3], batch_size=batch_size, shuffle=False, num_workers=0)

        for epoch in range(num_epochs_search):  
            model.train()
            for batch in train_loader:
                batch = batch.to('cpu')
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)[batch.batch_size:]
                loss = loss_fn(out, batch.y[batch.batch_size:])
                loss.backward()
                optimizer.step()

        _, eval_acc = evaluate(model, eval_loader, loss_fn)
        return eval_acc


    # ========== 4. Run Optuna ==========
    storage_str = f"sqlite:///{train_pid}-{selected_model}-{args.db_path}"
    study = optuna.create_study(
        direction='maximize',
        study_name=args.study_name,
        storage=storage_str,
        load_if_exists=True,
        sampler=TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=num_trials)


    # Save study result
    with open(os.path.join(results_dir, 'optuna_study.json'), 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)

    # ========== 5. Final Training ==========
    best_params = study.best_trial.params
    config_str = f"{selected_model}_h{best_params['hidden_channels']}_l{best_params['num_layers']}_d{best_params['dropout']:.2f}_lr{best_params['lr']:.0e}"
    best_model_path = os.path.join(results_dir, f"{config_str}.pth")

    model = MODEL_CLASSES[selected_model](
        in_channels=X.shape[1],
        hidden_channels=best_params['hidden_channels'],
        out_channels=np.unique(Y).shape[0],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )
    optimizer_name = best_params.get('optimizer', 'AdamW')  # Default to AdamW if not present

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'])
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=best_params['lr'], momentum=0.9)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=best_params['lr'])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


    loss_fn = nn.CrossEntropyLoss()

    train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                                num_neighbors=[3], batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = NeighborLoader(data, input_nodes=data.eval_mask,
                                num_neighbors=[3], batch_size=batch_size, shuffle=False, num_workers=0)

    train_log = []
    best_eval_acc = 0

    for epoch in range(1, num_epochs_eval+1):
        model.train()
        for batch in train_loader:
            batch = batch.to('cpu')
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[batch.batch_size:]
            loss = loss_fn(out, batch.y[batch.batch_size:])
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, loss_fn)
        eval_loss, eval_acc = evaluate(model, eval_loader, loss_fn)

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(model.state_dict(), best_model_path)

        train_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'eval_loss': eval_loss,
            'eval_acc': eval_acc
        })

        print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.4f} | Eval Acc: {eval_acc:.4f}")

    # Save logs
    with open(os.path.join(results_dir, 'train_log.json'), 'w') as f:
        json.dump(train_log, f, indent=2)

    with open(os.path.join(results_dir, 'best_model_config.json'), 'w') as f:
        json.dump({'best_model_path': best_model_path, 'params': best_params}, f, indent=2)

    # ========== 6. Test ==========
    with open(os.path.join(results_dir, 'best_model_config.json')) as f:
        best_model_meta = json.load(f)

    model = MODEL_CLASSES[selected_model](
        in_channels=X.shape[1],
        hidden_channels=best_model_meta['params']['hidden_channels'],
        out_channels=np.unique(Y).shape[0],
        num_layers=best_model_meta['params']['num_layers'],
        dropout=best_model_meta['params']['dropout']
    )
    model.load_state_dict(torch.load(best_model_meta['best_model_path']))
    print(f"\nLoaded best model from {best_model_meta['best_model_path']}")

    test_pids = [i for i in range(len(adata_paths)) if i != train_pid]
    test_results = {}

    for pid in tqdm(test_pids, desc="Evaluating"):
        adata_test = sc.read_h5ad(adata_paths[pid])
        X_test = adata_test.X.toarray() if not isinstance(adata_test.X, np.ndarray) else adata_test.X
        Y_test = LabelEncoder().fit_transform(adata_test.obs['Region'])

        sc.pp.neighbors(adata_test, n_neighbors=3, use_rep='X')
        edge_index_test, _ = from_scipy_sparse_matrix(adata_test.obsp['connectivities'])

        data_test = Data(x=torch.tensor(X_test, dtype=torch.float),
                        edge_index=edge_index_test,
                        y=torch.tensor(Y_test, dtype=torch.long))
        data_test.test_mask = torch.ones(data_test.num_nodes, dtype=torch.bool)

        test_loader = NeighborLoader(
            data_test, input_nodes=data_test.test_mask,
            num_neighbors=[3], batch_size=16, shuffle=False, num_workers=0)

        test_loss, test_acc = evaluate(model, test_loader, loss_fn)
        test_results[pid] = {'loss': float(test_loss), 'accuracy': float(test_acc)}
        print(f"Test PID {pid} | Accuracy: {test_acc:.4f}")

    with open(os.path.join(results_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)

    print("\n=== Final Test Accuracy per PID ===")
    for pid, res in test_results.items():
        print(f"PID {pid:02d} | Accuracy: {res['accuracy']:.4f}")
