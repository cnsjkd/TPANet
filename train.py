from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    module_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(module_dir.parent))

try:
    from .dataset import SEEDIVRawTrialDataset, collate_trials, list_common_subject_ids
    from .logger import CSVLogger
    from .model import EEGConformerClassifier
except ImportError:  # pragma: no cover
    from seed_iv_2026_like_de_LDS.dataset import SEEDIVRawTrialDataset, collate_trials, list_common_subject_ids  # type: ignore
    from seed_iv_2026_like_de_LDS.logger import CSVLogger  # type: ignore
    from seed_iv_2026_like_de_LDS.model import EEGConformerClassifier  # type: ignore


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, lengths, y, _ in loader:
        x = x.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x, lengths)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_count += int(y.size(0))

    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
    }


def train_one_fold(args: argparse.Namespace, test_subject: int, device: torch.device) -> Dict[str, float]:
    all_subjects = list_common_subject_ids(Path(args.root), args.sessions)
    if test_subject not in all_subjects:
        raise ValueError(f"test_subject {test_subject} not found in common subjects: {all_subjects}")

    train_subjects = [s for s in all_subjects if s != test_subject]
    if not train_subjects:
        raise ValueError("LOSO requires at least 2 subjects")

    fold_cache_dir = None
    if args.cache_dir:
        fold_cache_dir = Path(args.cache_dir) / f"fold_testsub{test_subject:02d}"
        fold_cache_dir.mkdir(parents=True, exist_ok=True)

    train_set = SEEDIVRawTrialDataset(
        root=args.root,
        sessions=args.sessions,
        subject_ids=train_subjects,
        chunk_size=args.chunk_size,
        num_channel=args.num_channel,
        cache_dir=fold_cache_dir,
        per_channel_zscore=not args.no_zscore,
        mat_cache_size=args.mat_cache_size,
    )
    test_set = SEEDIVRawTrialDataset(
        root=args.root,
        sessions=args.sessions,
        subject_ids=[test_subject],
        chunk_size=args.chunk_size,
        num_channel=args.num_channel,
        cache_dir=fold_cache_dir,
        per_channel_zscore=not args.no_zscore,
        mat_cache_size=args.mat_cache_size,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_trials,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_trials,
    )

    model = EEGConformerClassifier(
        num_classes=4,
        channels=args.num_channel,
        bands=5,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.num_layers,
        conv_kernel=args.conformer_conv_kernel,
        dropout=args.dropout,
        smoother_layers=args.smoother_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
        if args.cosine
        else None
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_test_acc = -1.0
    best_epoch = -1

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for x, lengths, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                logits = model(x, lengths)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * y.size(0)
            running_correct += int((logits.detach().argmax(dim=-1) == y).sum().item())
            running_count += int(y.size(0))

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(running_count, 1)
        train_acc = running_correct / max(running_count, 1)
        test_metrics = evaluate(model, test_loader, device)

        test_acc = float(test_metrics["acc"])
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            if args.save_dir:
                save_dir = Path(args.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"seediv_e2e_conformer_testsub{test_subject:02d}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "test_subject": test_subject,
                        "best_epoch": best_epoch,
                        "best_test_acc": best_test_acc,
                    },
                    save_path,
                )

        if epoch % args.log_every == 0 or epoch in (1, args.epochs):
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[sub{test_subject:02d}] epoch {epoch:03d}/{args.epochs} "
                f"lr={current_lr:.3e} train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
                f"test_loss={test_metrics['loss']:.4f} test_acc={test_acc*100:.2f}% "
                f"best={best_test_acc*100:.2f}%@{best_epoch}"
            )

    elapsed = time.time() - start_time
    return {
        "test_subject": float(test_subject),
        "best_acc": float(best_test_acc),
        "best_epoch": float(best_epoch),
        "seconds": float(elapsed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEED-IV raw EEG -> learnable DE-like + Conformer")

    parser.add_argument("--root", type=str, required=True, help="Path to SEED-IV eeg_raw_data")
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3], help="e.g. 1 2 3")

    parser.add_argument("--chunk_size", type=int, default=800, help="4s @ 200Hz = 800")
    parser.add_argument("--num_channel", type=int, default=62)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--conformer_conv_kernel", type=int, default=15)
    parser.add_argument("--smoother_layers", type=int, default=2)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--mat_cache_size", type=int, default=8)
    parser.add_argument("--no_zscore", action="store_true")

    parser.add_argument("--test_subject", type=int, default=1)
    parser.add_argument("--loso", action="store_true")

    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--results_csv", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Device: {device}")
    print(f"Sessions: {args.sessions}")

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    subjects = list_common_subject_ids(root, args.sessions)
    if len(subjects) < 2:
        raise ValueError(f"need at least 2 common subjects across sessions {args.sessions}, got: {subjects}")

    if args.loso:
        fold_subjects = subjects
    else:
        if args.test_subject not in subjects:
            raise ValueError(f"test_subject {args.test_subject} not found in common subjects: {subjects}")
        fold_subjects = [args.test_subject]

    results_csv = (
        Path(args.results_csv)
        if args.results_csv
        else Path(__file__).resolve().parents[1] / "results_seed_iv_2026_like_de_lds.csv"
    )
    logger = CSVLogger(
        results_csv,
        headers=[
            "timestamp",
            "fold",
            "test_subject",
            "best_epoch",
            "best_test_acc",
            "seconds",
            "sessions",
            "chunk_size",
            "batch_size",
            "epochs",
            "lr",
            "weight_decay",
            "d_model",
            "num_layers",
            "num_heads",
        ],
    )

    fold_results: List[Dict[str, float]] = []
    for fold_idx, test_subject in enumerate(fold_subjects, start=1):
        print(f"\n=== Fold {fold_idx}/{len(fold_subjects)} | test_subject={test_subject:02d} ===")
        result = train_one_fold(args, test_subject=test_subject, device=device)
        fold_results.append(result)

        logger.log(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                fold_idx,
                test_subject,
                int(result["best_epoch"]),
                f"{result['best_acc']:.4f}",
                f"{result['seconds']:.1f}",
                " ".join(str(s) for s in args.sessions),
                args.chunk_size,
                args.batch_size,
                args.epochs,
                args.lr,
                args.weight_decay,
                args.d_model,
                args.num_layers,
                args.num_heads,
            ]
        )

    if args.loso:
        accs = [r["best_acc"] for r in fold_results]
        mean_acc = float(np.mean(accs)) if accs else 0.0
        std_acc = float(np.std(accs)) if accs else 0.0
        print("\nLOSO summary:")
        print(f"mean_acc={mean_acc*100:.2f}% std={std_acc*100:.2f}%")


if __name__ == "__main__":
    main()
