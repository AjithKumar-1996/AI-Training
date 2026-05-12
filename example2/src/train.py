"""
train.py — Full training pipeline with:
  - Training config display
  - Model layer display
  - User confirmation before starting
  - Epoch-wise progress (rich live table)
  - Early stopping
  - Periodic checkpoint saving + best model saving
  - Resume from checkpoint
  - Training curve plots saved to results/
"""

import os
import time
import json
import glob
import shutil
import logging
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.progress import (Progress, SpinnerColumn, BarColumn,
                            TextColumn, TimeElapsedColumn, TimeRemainingColumn)
from rich.live    import Live
from rich.prompt  import Confirm

from .model   import build_model, print_model_layers, load_model
from .dataset import load_and_split, get_dataloaders, print_dataset_summary
from .utils   import (console, section, kv_table, confirm,
                       get_device, set_seed, save_config_snapshot)

logger = logging.getLogger("example2")


# ═══════════════════════════════════════════════════════════════════════════════
# Early Stopping
# ═══════════════════════════════════════════════════════════════════════════════
class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, monitor: str = "val_loss"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.monitor    = monitor
        self.best       = None
        self.counter    = 0
        self.triggered  = False

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        improved = (
            self.best is None or
            (self.monitor == "val_loss"     and metric < self.best - self.min_delta) or
            (self.monitor == "val_accuracy" and metric > self.best + self.min_delta)
        )
        if improved:
            self.best    = metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                return True
            return False

    def status(self) -> str:
        return (f"No improvement for {self.counter}/{self.patience} epochs "
                f"(best {self.monitor}: {self.best:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint Manager
# ═══════════════════════════════════════════════════════════════════════════════
class CheckpointManager:
    def __init__(self, save_dir: str, max_keep: int = 3):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir  = save_dir
        self.max_keep  = max_keep
        self.best_path = os.path.join(save_dir, "best_model.pth")
        self._history  = []

    def save(self, epoch: int, model, optimizer, scheduler,
             metrics: dict, is_best: bool):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.save_dir, f"epoch_{epoch:03d}_{ts}.pth")
        payload = {
            "epoch":             epoch,
            "model_state_dict":  model.state_dict(),
            "optimizer_state":   optimizer.state_dict(),
            "scheduler_state":   scheduler.state_dict() if scheduler else None,
            "metrics":           metrics,
        }
        torch.save(payload, path)
        self._history.append(path)
        console.print(f"  [dim]Checkpoint saved:[/dim] [cyan]{Path(path).name}[/cyan]")
        logger.info(f"Checkpoint saved: {path}")

        # Enforce max_keep
        while len(self._history) > self.max_keep:
            old = self._history.pop(0)
            if os.path.exists(old):
                os.remove(old)
                console.print(f"  [dim]Removed old checkpoint:[/dim] {Path(old).name}")

        if is_best:
            torch.save(payload, self.best_path)
            console.print(f"  [bold green]Best model updated[/bold green] -> best_model.pth")
            logger.info("Best model updated.")

    def latest_checkpoint(self) -> str | None:
        """Find the most recent epoch_*.pth or best_model.pth in save_dir."""
        pattern = os.path.join(self.save_dir, "epoch_*.pth")
        files   = sorted(glob.glob(pattern))
        return files[-1] if files else None

    def load(self, path: str, model, optimizer=None, scheduler=None) -> int:
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        epoch = ckpt["epoch"]
        console.print(f"  [success]Resumed from epoch {epoch}:[/success] {Path(path).name}")
        return epoch


# ═══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


# ═══════════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════════
def print_training_config(cfg: dict):
    tr  = cfg["training"]
    es  = tr["early_stopping"]
    ck  = tr["checkpoint"]
    sch = tr["lr_scheduler"]

    rows = [
        ("Device",               tr.get("device", "auto")),
        ("Batch Size",           tr["batch_size"]),
        ("Max Epochs",           tr["num_epochs"]),
        ("Learning Rate",        tr["learning_rate"]),
        ("Weight Decay",         tr["weight_decay"]),
        ("Optimizer",            tr["optimizer"].upper()),
        ("LR Scheduler",         f"{sch['type']} (T_max={sch['T_max']}, eta_min={sch['eta_min']})"),
        ("Early Stopping",       f"{'Enabled' if es['enabled'] else 'Disabled'} "
                                  f"(patience={es['patience']}, monitor={es['monitor']})"),
        ("Checkpoint Dir",       ck["save_dir"]),
        ("Save Every N Epochs",  ck["save_every_n_epochs"]),
        ("Max Checkpoints",      ck["max_checkpoints_to_keep"]),
        ("Save Best Model",      str(ck["save_best_model"])),
        ("Random Seed",          tr["seed"]),
    ]
    t = kv_table("Training Configuration", rows, accent="green")
    console.print(t)


def _make_history_table(history: list[dict]) -> Table:
    t = Table(
        title="[bold cyan]Training History[/bold cyan]",
        border_style="blue", show_lines=False,
        header_style="bold cyan", min_width=100,
    )
    t.add_column("Epoch",     style="dim",         width=7,  justify="right")
    t.add_column("Tr Loss",   style="yellow",       width=10, justify="right")
    t.add_column("Tr Acc %",  style="yellow",       width=10, justify="right")
    t.add_column("Val Loss",  style="cyan",          width=10, justify="right")
    t.add_column("Val Acc %", style="cyan",          width=10, justify="right")
    t.add_column("LR",        style="dim",           width=12, justify="right")
    t.add_column("Time(s)",   style="dim",           width=10, justify="right")
    t.add_column("Status",    style="bold white",    width=22)

    for h in history:
        status_str = ""
        if h.get("is_best"):
            status_str = "[bold green]* Best[/bold green]"
        if h.get("checkpoint_saved"):
            status_str += " [cyan][CK][/cyan]"
        if h.get("early_stop"):
            status_str += " [bold red]Early Stop[/bold red]"
        t.add_row(
            str(h["epoch"]),
            f"{h['tr_loss']:.4f}",
            f"{h['tr_acc']*100:.2f}",
            f"{h['vl_loss']:.4f}",
            f"{h['vl_acc']*100:.2f}",
            f"{h['lr']:.2e}",
            f"{h['epoch_time']:.1f}",
            status_str,
        )
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Plot training curves
# ═══════════════════════════════════════════════════════════════════════════════
def save_training_curves(history: list[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs    = [h["epoch"] for h in history]
    tr_losses = [h["tr_loss"] for h in history]
    vl_losses = [h["vl_loss"] for h in history]
    tr_accs   = [h["tr_acc"]*100 for h in history]
    vl_accs   = [h["vl_acc"]*100 for h in history]
    best_ep   = next((h["epoch"] for h in history if h.get("is_best")), None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves — SimpleCNN Apple vs Pineapple", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, tr_losses, "b-o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, vl_losses, "r-o", markersize=3, label="Val Loss")
    if best_ep:
        best_vl = vl_losses[epochs.index(best_ep)]
        axes[0].axvline(best_ep, color="green", linestyle="--", alpha=0.6, label=f"Best (ep {best_ep})")
        axes[0].scatter([best_ep], [best_vl], color="green", zorder=5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, tr_accs, "b-o", markersize=3, label="Train Acc %")
    axes[1].plot(epochs, vl_accs, "r-o", markersize=3, label="Val Acc %")
    if best_ep:
        best_va = vl_accs[epochs.index(best_ep)]
        axes[1].axvline(best_ep, color="green", linestyle="--", alpha=0.6, label=f"Best (ep {best_ep})")
        axes[1].scatter([best_ep], [best_va], color="green", zorder=5)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  [dim]Training curves saved:[/dim] [cyan]{path}[/cyan]")
    logger.info(f"Training curves saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main training entry point
# ═══════════════════════════════════════════════════════════════════════════════
def run_training(cfg: dict, resume_path: str | None = None):
    """
    Full training pipeline.
    resume_path: if provided, load this checkpoint and continue from its epoch.
    """
    section("DATASET")
    train_ds, val_ds, test_ds, class_names, class_to_idx = load_and_split(cfg)
    print_dataset_summary(cfg, train_ds, val_ds, test_ds, class_names)

    section("MODEL ARCHITECTURE")
    device = get_device(cfg["training"].get("device", "auto"))
    set_seed(cfg["training"]["seed"])
    model = build_model(cfg["model"]).to(device)
    print_model_layers(model)

    section("TRAINING CONFIGURATION")
    print_training_config(cfg)

    # ── User confirmation ────────────────────────────────────────────────────
    console.print()
    if resume_path:
        console.print(f"  [yellow]Will resume from:[/yellow] [cyan]{resume_path}[/cyan]")
    if not confirm("Proceed with training?", default=True):
        console.print("[yellow]Training cancelled.[/yellow]")
        return

    section("TRAINING")

    # ── Setup ────────────────────────────────────────────────────────────────
    tr_cfg    = cfg["training"]
    sch_cfg   = tr_cfg["lr_scheduler"]
    es_cfg    = tr_cfg["early_stopping"]
    ck_cfg    = tr_cfg["checkpoint"]
    log_cfg   = cfg["logging"]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=tr_cfg["learning_rate"],
                                  weight_decay=tr_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sch_cfg["T_max"], eta_min=sch_cfg["eta_min"])

    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, tr_cfg)

    ckpt_mgr   = CheckpointManager(ck_cfg["save_dir"],
                                    max_keep=ck_cfg["max_checkpoints_to_keep"])
    early_stop = EarlyStopping(
        patience  = es_cfg["patience"],
        min_delta = es_cfg["min_delta"],
        monitor   = es_cfg["monitor"],
    ) if es_cfg["enabled"] else None

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    if resume_path:
        start_epoch = ckpt_mgr.load(resume_path, model, optimizer, scheduler) + 1
        console.print(f"  Resuming from epoch [cyan]{start_epoch}[/cyan]")

    # Save config snapshot alongside results
    save_config_snapshot(cfg, log_cfg["results_dir"])

    best_val_loss = float("inf")
    history       = []

    console.print()
    console.print(f"  Starting training: [bold cyan]{tr_cfg['num_epochs']}[/bold cyan] epochs  |  "
                  f"[bold cyan]{len(train_loader)}[/bold cyan] batches/epoch  |  "
                  f"Batch size: [bold cyan]{tr_cfg['batch_size']}[/bold cyan]")
    console.print()

    total_start = time.time()

    for epoch in range(start_epoch, tr_cfg["num_epochs"] + 1):
        epoch_start = time.time()

        # Train
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        # Validate
        vl_loss, vl_acc = _run_epoch(model, val_loader,   criterion, None,      device, train=False)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        is_best = vl_loss < best_val_loss
        if is_best:
            best_val_loss = vl_loss

        # Early stopping check
        stop_now = early_stop.step(vl_loss) if early_stop else False

        # Checkpoint
        save_ck = (epoch % ck_cfg["save_every_n_epochs"] == 0) or stop_now
        if save_ck or is_best:
            ckpt_mgr.save(
                epoch, model, optimizer, scheduler,
                {"tr_loss": tr_loss, "tr_acc": tr_acc,
                 "vl_loss": vl_loss, "vl_acc": vl_acc},
                is_best=is_best,
            )

        # Record history
        entry = {
            "epoch":            epoch,
            "tr_loss":          tr_loss,
            "tr_acc":           tr_acc,
            "vl_loss":          vl_loss,
            "vl_acc":           vl_acc,
            "lr":               current_lr,
            "epoch_time":       epoch_time,
            "is_best":          is_best,
            "checkpoint_saved": save_ck,
            "early_stop":       stop_now,
        }
        history.append(entry)
        logger.info(json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                 for k, v in entry.items()}))

        # ── Inline epoch summary ─────────────────────────────────────────────
        elapsed = timedelta(seconds=int(time.time() - total_start))
        best_mark = " [bold green][BEST][/bold green]" if is_best else ""
        ck_mark   = " [cyan][CK][/cyan]"               if save_ck  else ""
        es_info   = (f"  [dim]{early_stop.status()}[/dim]"
                     if early_stop and not is_best else "")

        console.print(
            f"  Epoch [{epoch:>3d}/{tr_cfg['num_epochs']}]  "
            f"Train — loss: [yellow]{tr_loss:.4f}[/yellow]  acc: [yellow]{tr_acc*100:.2f}%[/yellow]  |  "
            f"Val — loss: [cyan]{vl_loss:.4f}[/cyan]  acc: [cyan]{vl_acc*100:.2f}%[/cyan]  |  "
            f"LR: [dim]{current_lr:.2e}[/dim]  "
            f"Time: {epoch_time:.1f}s  Elapsed: {elapsed}"
            + best_mark + ck_mark
        )
        if es_info:
            console.print(es_info)

        if stop_now:
            console.print()
            console.print(Panel(
                f"[bold yellow]Early stopping triggered at epoch {epoch}.[/bold yellow]\n"
                f"Best val_loss: [cyan]{best_val_loss:.4f}[/cyan]  "
                f"(patience={es_cfg['patience']})",
                border_style="yellow", expand=False,
            ))
            break

    # ── Final evaluation on test set ─────────────────────────────────────────
    section("EVALUATION — TEST SET")
    console.print("  Loading best model for final evaluation...")
    best_model = load_model(ckpt_mgr.best_path, cfg["model"], device)
    te_loss, te_acc = _run_epoch(best_model, test_loader, criterion, None, device, train=False)

    total_time = timedelta(seconds=int(time.time() - total_start))
    t = Table(title="[bold green]Final Results[/bold green]",
              border_style="green", show_header=False, min_width=55)
    t.add_column(style="key",   min_width=25)
    t.add_column(style="value", min_width=18)
    t.add_row("Total Training Time",  str(total_time))
    t.add_row("Epochs Trained",       str(len(history)))
    t.add_row("Best Val Loss",        f"{best_val_loss:.4f}")
    t.add_row("Best Val Accuracy",    f"{max(h['vl_acc'] for h in history)*100:.2f}%")
    t.add_row("Test Loss",            f"{te_loss:.4f}")
    t.add_row("Test Accuracy",        f"{te_acc*100:.2f}%")
    t.add_row("Best Model Path",      ckpt_mgr.best_path)
    console.print(t)

    # ── Save training curves ─────────────────────────────────────────────────
    if cfg["logging"].get("save_training_curves", True):
        save_training_curves(history, cfg["logging"]["results_dir"])

    # ── Print full history table ──────────────────────────────────────────────
    section("TRAINING HISTORY")
    console.print(_make_history_table(history))
    console.print()
    console.print(
        "[bold green]Training complete![/bold green]  "
        f"Best model: [cyan]{ckpt_mgr.best_path}[/cyan]"
    )
