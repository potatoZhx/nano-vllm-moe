"""
Train a tiny theoretical-acceptance predictor for random_cache MoE decoding.
python -u train_acceptance_predictor.py \
  --data-file /data2/group_谈海生/mumura/dynamick/predictor/random_cache_acceptance_dataset_20260613.pt \
  --output-dir /data2/group_谈海生/mumura/dynamick/predictor/random_cache_acceptance_20260613 \
  --epochs 5 \
  --batch-size 512

"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class AcceptancePredictor(nn.Module):
    """Small branch-encoder MLP for alpha=sum(min(p,q)) prediction."""

    def __init__(
        self,
        route_raw_dim: int,
        route_summary_dim: int,
        token_feature_dim: int,
        hidden_dim: int,
        history_dim: int,
        route_raw_embed: int = 32,
        hidden_embed: int = 32,
    ) -> None:
        super().__init__()
        self.route_raw_encoder = nn.Sequential(
            nn.LayerNorm(route_raw_dim),
            nn.Linear(route_raw_dim, route_raw_embed),
            nn.SiLU(),
        )
        self.route_summary_encoder = nn.Sequential(
            nn.LayerNorm(route_summary_dim),
            nn.Linear(route_summary_dim, 32),
            nn.SiLU(),
        )
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(token_feature_dim),
            nn.Linear(token_feature_dim, 16),
            nn.SiLU(),
        )
        self.hidden_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_embed),
            nn.SiLU(),
        )
        self.history_encoder = nn.Sequential(
            nn.LayerNorm(history_dim),
            nn.Linear(history_dim, 16),
            nn.SiLU(),
        )
        fused_dim = route_raw_embed + 32 + 16 + hidden_embed + 16
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, route_raw, route_summary, token_features, hidden, history):
        z = torch.cat(
            [
                self.route_raw_encoder(route_raw),
                self.route_summary_encoder(route_summary),
                self.token_encoder(token_features),
                self.hidden_encoder(hidden),
                self.history_encoder(history),
            ],
            dim=-1,
        )
        return self.head(z)


def make_dataset(split: Dict[str, torch.Tensor]) -> TensorDataset:
    return TensorDataset(
        split["route_raw"].float(),
        split["route_summary"].float(),
        split["token_features"].float(),
        split["hidden"].float(),
        split["history"].float(),
        split["y"].float(),
    )


def acceptance_loss(pred, target, log_lambda: float = 0.25):
    pred_c = pred.clamp(1e-5, 1.0)
    target_c = target.clamp(1e-5, 1.0)
    mse = torch.mean((pred - target) ** 2)
    log_mse = torch.mean((-torch.log(pred_c) + torch.log(target_c)) ** 2)
    return mse + log_lambda * log_mse, mse.detach(), log_mse.detach()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        route_raw, route_summary, token_features, hidden, history, y = [x.to(device) for x in batch]
        p = model(route_raw, route_summary, token_features, hidden, history)
        preds.append(p.cpu())
        targets.append(y.cpu())
    pred = torch.cat(preds, dim=0).numpy().reshape(-1)
    y = torch.cat(targets, dim=0).numpy().reshape(-1)
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(mse))
    y_var = float(np.var(y) + 1e-12)
    r2 = float(1.0 - mse / y_var)
    corr = float(np.corrcoef(pred, y)[0, 1]) if len(pred) > 2 and np.std(pred) > 1e-8 and np.std(y) > 1e-8 else 0.0

    # Acceptance-chain relevant error: -log(alpha).
    pred_log = -np.log(np.clip(pred, 1e-5, 1.0))
    y_log = -np.log(np.clip(y, 1e-5, 1.0))
    log_mae = float(np.mean(np.abs(pred_log - y_log)))
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "corr": corr, "log_mae": log_mae}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="/data2/group_谈海生/lagin/data/Sd_Data/data/random_cache_acceptance_dataset.pt")
    parser.add_argument("--output-dir", default="/data2/group_谈海生/lagin/models/SRDP_Experiments/random_cache_acceptance")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--log-lambda", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(args.data_file, map_location="cpu")
    train_split = data["train"]
    test_split = data["test"]
    meta = data["meta"]

    train_ds_full = make_dataset(train_split)
    total = len(train_ds_full)
    val_size = max(1, int(total * args.val_ratio))
    train_size = total - val_size
    train_ds, val_ds = random_split(train_ds_full, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    test_ds = make_dataset(test_split)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = AcceptancePredictor(
        route_raw_dim=meta["route_raw_dim"],
        route_summary_dim=meta["route_summary_dim"],
        token_feature_dim=meta["token_feature_dim"],
        hidden_dim=meta["hidden_dim"],
        history_dim=meta["history_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    exp_dir = Path(args.output_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "meta": meta}, f, indent=2, ensure_ascii=False)

    print(f"Experiment dir: {exp_dir}")
    print(f"Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Alpha train mean/std: {train_split['y'].mean().item():.4f}/{train_split['y'].std().item():.4f}")

    best_val = float("inf")
    log_path = exp_dir / "training_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_mse,train_log_mse,val_mse,val_mae,val_rmse,val_r2,val_corr,val_log_mae\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_log = 0.0
        n_batches = 0
        for batch in train_loader:
            route_raw, route_summary, token_features, hidden, history, y = [x.to(device, non_blocking=True) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            pred = model(route_raw, route_summary, token_features, hidden, history)
            loss, mse, log_mse = acceptance_loss(pred, y, log_lambda=args.log_lambda)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_mse += float(mse.item())
            total_log += float(log_mse.item())
            n_batches += 1

        val = evaluate(model, val_loader, device)
        train_loss = total_loss / max(1, n_batches)
        train_mse = total_mse / max(1, n_batches)
        train_log = total_log / max(1, n_batches)
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.6f} mse={train_mse:.6f} "
            f"| val_mse={val['mse']:.6f} val_mae={val['mae']:.6f} "
            f"r2={val['r2']:.4f} corr={val['corr']:.4f} log_mae={val['log_mae']:.6f}"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.8f},{train_mse:.8f},{train_log:.8f},"
                f"{val['mse']:.8f},{val['mae']:.8f},{val['rmse']:.8f},{val['r2']:.8f},{val['corr']:.8f},{val['log_mae']:.8f}\n"
            )

        if val["mse"] < best_val:
            best_val = val["mse"]
            torch.save(model.state_dict(), exp_dir / "best_model.pth")

    model.load_state_dict(torch.load(exp_dir / "best_model.pth", map_location=device))
    test = evaluate(model, test_loader, device)
    report = {
        "test_mse": test["mse"],
        "test_mae": test["mae"],
        "test_rmse": test["rmse"],
        "test_r2": test["r2"],
        "test_corr": test["corr"],
        "test_log_mae": test["log_mae"],
        "best_val_mse": best_val,
    }
    with open(exp_dir / "final_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Final test:", json.dumps(report, indent=2))
    print(f"Saved best model and logs to {exp_dir}")


if __name__ == "__main__":
    main()
