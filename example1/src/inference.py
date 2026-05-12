"""
inference.py — Runs inference on a folder of images.

Outputs:
  - Per-image: predicted class, confidence, latency
  - If ground-truth labels available (ImageFolder layout): F1, confusion matrix
  - Annotated images saved to cfg['inference']['output_dir']
  - Confusion matrix PNG saved to results/
"""

import os
import time
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from torchvision import datasets, transforms

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)

from rich.table   import Table
from rich.console import Console
from rich.panel   import Panel
from rich.progress import (Progress, BarColumn, TextColumn,
                            TimeElapsedColumn, SpinnerColumn)

from .dataset import get_inference_transform
from .model   import load_model
from .utils   import console, section, kv_table

logger = logging.getLogger("example1")


# ═══════════════════════════════════════════════════════════════════════════════
# Single-folder inference (no ground truth)
# ═══════════════════════════════════════════════════════════════════════════════
def _infer_batch(model, device, img_tensors: torch.Tensor):
    """Returns (probs, preds) for a batch tensor."""
    model.eval()
    with torch.no_grad():
        logits = model(img_tensors.to(device))
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
    return probs.cpu(), preds.cpu()


def _annotate_and_save(image_path: str, pred_class: str, confidence: float,
                        out_dir: str, cfg_inf: dict):
    """Draw label box on image and save."""
    img     = Image.open(image_path).convert("RGB")
    draw    = ImageDraw.Draw(img)
    w, h    = img.size
    label   = f"{pred_class}  {confidence*100:.1f}%"
    colour  = (0, 200, 0) if pred_class == "dog" else (0, 120, 255)

    # Font
    try:
        font = ImageFont.truetype("arial.ttf", size=max(18, h // 20))
    except Exception:
        font = ImageFont.load_default()

    # Text background
    bbox = draw.textbbox((8, 8), label, font=font)
    draw.rectangle([bbox[0]-4, bbox[1]-4, bbox[2]+4, bbox[3]+4], fill=colour)
    draw.text((8, 8), label, fill="white", font=font)

    # Border
    thick = cfg_inf.get("box_thickness", 2)
    draw.rectangle([0, 0, w-1, h-1], outline=colour, width=thick)

    os.makedirs(out_dir, exist_ok=True)
    stem = Path(image_path).stem
    out_path = os.path.join(out_dir, f"{stem}_predicted_{pred_class}.jpg")
    img.save(out_path, quality=92)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# Confusion matrix plot
# ═══════════════════════════════════════════════════════════════════════════════
def _save_confusion_matrix(cm: np.ndarray, class_names: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"  [dim]Confusion matrix saved:[/dim] [cyan]{out_path}[/cyan]")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main inference entry point
# ═══════════════════════════════════════════════════════════════════════════════
def run_inference(cfg: dict, model_path: str, input_path: str):
    """
    input_path may be:
      (a) A single image file
      (b) A flat folder of images (no sub-folders) — no ground truth
      (c) An ImageFolder-structured folder (sub-folders = class names) — has ground truth
    """
    inf_cfg  = cfg["inference"]
    mod_cfg  = cfg["model"]
    log_cfg  = cfg["logging"]
    out_dir  = inf_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    from .utils import get_device
    device = get_device(cfg["training"].get("device", "auto"))

    # ── Load model ────────────────────────────────────────────────────────────
    section("MODEL LOADING")
    model = load_model(model_path, mod_cfg, device)
    model.eval()
    class_names = mod_cfg["class_names"]
    console.print(f"  Model loaded from [cyan]{model_path}[/cyan]")
    console.print(f"  Classes: {class_names}")

    transform  = get_inference_transform(mod_cfg["input_size"])
    conf_thresh = inf_cfg.get("confidence_threshold", 0.5)

    # ── Discover images and ground truth ─────────────────────────────────────
    input_path_obj = Path(input_path)
    has_gt         = False
    image_paths    = []
    gt_labels      = []    # int indices, if available

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if input_path_obj.is_file():
        image_paths = [str(input_path_obj)]

    elif input_path_obj.is_dir():
        # Check if it's an ImageFolder layout (sub-dirs = class names)
        subdirs = [d for d in input_path_obj.iterdir() if d.is_dir()]
        known   = {c.lower() for c in class_names}
        if subdirs and all(d.name.lower() in known for d in subdirs):
            has_gt = True
            for d in sorted(subdirs):
                label_idx = class_names.index(d.name.lower()) \
                            if d.name.lower() in class_names else \
                            [c.lower() for c in class_names].index(d.name.lower())
                for f in sorted(d.iterdir()):
                    if f.suffix.lower() in EXTS:
                        image_paths.append(str(f))
                        gt_labels.append(label_idx)
        else:
            # Flat folder
            for f in sorted(input_path_obj.iterdir()):
                if f.suffix.lower() in EXTS:
                    image_paths.append(str(f))

    if not image_paths:
        console.print("[bold red]No images found at the specified path.[/bold red]")
        return

    console.print(f"  Found [cyan]{len(image_paths)}[/cyan] images  |  "
                  f"Ground truth available: [{'green]Yes' if has_gt else 'red]No'}[/]")

    # ── Run inference ─────────────────────────────────────────────────────────
    section("RUNNING INFERENCE")

    results      = []
    batch_size   = inf_cfg.get("batch_size", 16)
    latencies_ms = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Classifying images...", total=len(image_paths))

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            tensors     = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(transform(img))
                except Exception as e:
                    console.print(f"  [warning]Skipping {p}: {e}[/warning]")
                    tensors.append(None)

            valid_idx     = [j for j, t in enumerate(tensors) if t is not None]
            valid_tensors = torch.stack([tensors[j] for j in valid_idx])

            t0         = time.perf_counter()
            probs, preds = _infer_batch(model, device, valid_tensors)
            t1         = time.perf_counter()

            batch_lat = (t1 - t0) * 1000 / len(valid_idx)   # ms per image

            result_map = {}
            for k, j in enumerate(valid_idx):
                pred_idx   = preds[k].item()
                confidence = probs[k][pred_idx].item()
                result_map[j] = {
                    "path":       batch_paths[j],
                    "pred_idx":   pred_idx,
                    "pred_class": class_names[pred_idx],
                    "confidence": confidence,
                    "latency_ms": batch_lat,
                    "all_probs":  {class_names[c]: round(probs[k][c].item(), 4)
                                   for c in range(len(class_names))},
                }
                latencies_ms.append(batch_lat)

            for j in range(len(batch_paths)):
                if j in result_map:
                    r = result_map[j]
                else:
                    r = {"path": batch_paths[j], "pred_idx": -1,
                         "pred_class": "error", "confidence": 0.0,
                         "latency_ms": 0.0, "all_probs": {}}

                if has_gt:
                    gt_idx         = gt_labels[i + j]
                    r["gt_class"]  = class_names[gt_idx]
                    r["gt_idx"]    = gt_idx
                    r["correct"]   = r["pred_idx"] == gt_idx

                results.append(r)

                # Annotate
                if inf_cfg.get("save_annotated_images", True) and r["pred_idx"] != -1:
                    _annotate_and_save(r["path"], r["pred_class"],
                                       r["confidence"], out_dir, inf_cfg)

            prog.advance(task, len(batch_paths))

    # ── Per-image results table ───────────────────────────────────────────────
    section("RESULTS")

    rt = Table(
        title=f"[bold cyan]Per-Image Inference Results[/bold cyan]",
        border_style="cyan", show_lines=False,
        header_style="bold cyan", min_width=110,
    )
    rt.add_column("Image",           style="dim",         min_width=30)
    rt.add_column("Predicted",       style="bold white",  width=10)
    rt.add_column("Confidence",      style="yellow",      width=12, justify="right")
    for cn in class_names:
        rt.add_column(f"P({cn})",    style="dim",         width=10, justify="right")
    rt.add_column("Latency (ms)",    style="dim",         width=14, justify="right")
    if has_gt:
        rt.add_column("True Label",  style="cyan",        width=10)
        rt.add_column("Correct",     style="bold",        width=9)

    for r in results:
        row = [
            Path(r["path"]).name,
            r["pred_class"],
            f"{r['confidence']*100:.1f}%",
        ]
        for cn in class_names:
            row.append(f"{r['all_probs'].get(cn, 0.0)*100:.1f}%")
        row.append(f"{r['latency_ms']:.2f}")
        if has_gt:
            row.append(r.get("gt_class", "—"))
            row.append("[green]✓[/green]" if r.get("correct") else "[red]✗[/red]")
        rt.add_row(*row)

    console.print(rt)

    # ── Summary metrics ────────────────────────────────────────────────────────
    section("INFERENCE SUMMARY")

    lat = np.array(latencies_ms)
    rows = [
        ("Total Images",            len(results)),
        ("Avg Latency per Image",   f"{lat.mean():.2f} ms"),
        ("Min Latency",             f"{lat.min():.2f} ms"),
        ("Max Latency",             f"{lat.max():.2f} ms"),
        ("P95 Latency",             f"{np.percentile(lat, 95):.2f} ms"),
        ("Throughput",              f"{1000/lat.mean():.1f} img/sec"),
        ("Output Dir",              out_dir),
    ]

    if has_gt:
        y_true = [r["gt_idx"]   for r in results if r.get("gt_idx")  is not None]
        y_pred = [r["pred_idx"] for r in results if r.get("gt_idx")  is not None]

        acc    = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average="macro")
        f1_wt  = f1_score(y_true, y_pred, average="weighted")
        cm     = confusion_matrix(y_true, y_pred)

        rows += [
            ("─── Accuracy Metrics ───",   ""),
            ("Overall Accuracy",           f"{acc*100:.2f}%"),
            ("F1 Score (Macro)",           f"{f1_mac:.4f}"),
            ("F1 Score (Weighted)",        f"{f1_wt:.4f}"),
        ]
        # Per-class F1
        f1_per = f1_score(y_true, y_pred, average=None)
        for i, cn in enumerate(class_names):
            rows.append((f"  F1 — {cn}", f"{f1_per[i]:.4f}"))

    t = kv_table("Inference Summary", rows, accent="green")
    console.print(t)

    if has_gt:
        # Confusion matrix
        console.print()
        cm_t = Table(title="[bold cyan]Confusion Matrix[/bold cyan]",
                     border_style="cyan", show_lines=True,
                     header_style="bold cyan", min_width=40)
        cm_t.add_column("True \\ Pred", style="bold white", width=14)
        for cn in class_names:
            cm_t.add_column(cn, style="yellow", width=10, justify="center")
        for i, cn in enumerate(class_names):
            row = [cn] + [str(cm[i, j]) for j in range(len(class_names))]
            cm_t.add_row(*row)
        console.print(cm_t)

        # Per-class report
        report = classification_report(y_true, y_pred,
                                        target_names=class_names, digits=4)
        console.print()
        console.print(Panel(
            f"[bold cyan]Classification Report[/bold cyan]\n[dim]{report}[/dim]",
            border_style="cyan", expand=False,
        ))

        # Save confusion matrix PNG
        _save_confusion_matrix(cm, class_names, cfg["logging"]["results_dir"])

    console.print()
    console.print(
        "[bold green]Inference complete.[/bold green]  "
        f"Annotated images saved to [cyan]{out_dir}[/cyan]"
    )
    logger.info(f"Inference done. {len(results)} images processed.")
