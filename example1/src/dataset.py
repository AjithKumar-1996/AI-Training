"""
dataset.py — Dataset loading, splitting, augmentation, and DataLoader creation.

Expected folder layout under cfg['dataset']['data_dir']:
  data/cats_and_dogs/
      cat/
          cat.001.jpg
          cat.002.jpg
          ...
      dog/
          dog.001.jpg
          ...

Uses torchvision.datasets.ImageFolder for automatic label discovery.
"""

import os
import math
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from rich.table import Table
from rich.console import Console
from rich.panel  import Panel

console = Console()


# ── Transform factories ───────────────────────────────────────────────────────

def get_train_transform(input_size: list, aug_cfg: dict) -> transforms.Compose:
    ops = [transforms.Resize((input_size[0] + 32, input_size[1] + 32))]  # slight oversize then crop
    ops.append(transforms.RandomCrop((input_size[0], input_size[1])))
    if aug_cfg.get("random_horizontal_flip", True):
        ops.append(transforms.RandomHorizontalFlip())
    if aug_cfg.get("random_vertical_flip", False):
        ops.append(transforms.RandomVerticalFlip())
    deg = aug_cfg.get("random_rotation_degrees", 0)
    if deg:
        ops.append(transforms.RandomRotation(deg))
    jitter_b = aug_cfg.get("color_jitter_brightness", 0)
    jitter_c = aug_cfg.get("color_jitter_contrast",   0)
    jitter_s = aug_cfg.get("color_jitter_saturation", 0)
    if any([jitter_b, jitter_c, jitter_s]):
        ops.append(transforms.ColorJitter(
            brightness=jitter_b, contrast=jitter_c, saturation=jitter_s))
    p_gray = aug_cfg.get("random_grayscale_prob", 0)
    if p_gray:
        ops.append(transforms.RandomGrayscale(p=p_gray))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


def get_eval_transform(input_size: list) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size[0], input_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


def get_inference_transform(input_size: list) -> transforms.Compose:
    """Same as eval but without normalisation (so we can also save annotated originals)."""
    return transforms.Compose([
        transforms.Resize((input_size[0], input_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ── Dataset loading & splitting ───────────────────────────────────────────────

def load_and_split(cfg: dict):
    """
    Loads the full ImageFolder dataset and splits it into train/val/test subsets.
    Returns (train_ds, val_ds, test_ds, class_names, class_to_idx).
    """
    ds_cfg  = cfg["dataset"]
    mod_cfg = cfg["model"]

    data_dir   = ds_cfg["data_dir"]
    input_size = mod_cfg["input_size"]
    aug_cfg    = ds_cfg["augmentation"]

    if not os.path.isdir(data_dir):
        console.print(f"  [bold red]ERROR:[/bold red] Dataset directory not found: [cyan]{data_dir}[/cyan]")
        console.print("  Please run [bold]download_dataset.py[/bold] or create the folder manually.")
        raise FileNotFoundError(data_dir)

    # Load full dataset with eval transform (we apply train transforms via Subset wrapper)
    full_ds = datasets.ImageFolder(data_dir, transform=get_eval_transform(input_size))

    total       = len(full_ds)
    n_train     = int(math.floor(ds_cfg["train_split"] * total))
    n_val       = int(math.floor(ds_cfg["val_split"]   * total))
    n_test      = total - n_train - n_val

    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    train_sub, val_sub, test_sub = random_split(
        full_ds, [n_train, n_val, n_test], generator=generator)

    # Wrap train subset to use augmented transform
    train_ds = _TransformSubset(train_sub, get_train_transform(input_size, aug_cfg))
    val_ds   = _TransformSubset(val_sub,   get_eval_transform(input_size))
    test_ds  = _TransformSubset(test_sub,  get_eval_transform(input_size))

    return train_ds, val_ds, test_ds, full_ds.classes, full_ds.class_to_idx


class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies a custom transform, overriding the parent's."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        # img is already a tensor (from ImageFolder transform) — we need PIL
        # Trick: get the original PIL image via the underlying dataset
        original_idx = self.subset.indices[idx]
        path, label  = self.subset.dataset.samples[original_idx]
        from PIL import Image
        pil_img = Image.open(path).convert("RGB")
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, label


def get_dataloaders(train_ds, val_ds, test_ds, tr_cfg: dict):
    bs          = tr_cfg["batch_size"]
    num_workers = min(tr_cfg.get("num_workers", 0), 4)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ── Display helpers ───────────────────────────────────────────────────────────

def print_dataset_summary(cfg: dict, train_ds, val_ds, test_ds, class_names):
    ds_cfg  = cfg["dataset"]
    total   = len(train_ds) + len(val_ds) + len(test_ds)

    # Count per class in each split
    def _class_counts(ds):
        labels = []
        for i in range(len(ds)):
            _, lbl = ds[i]
            labels.append(lbl)
        return Counter(labels)

    console.print()
    t = Table(
        title="[bold cyan]Dataset Split Summary[/bold cyan]",
        border_style="cyan",
        show_lines=True,
        min_width=80,
    )
    t.add_column("Split",    style="bold white",  width=12)
    t.add_column("Total",    style="yellow",       width=10, justify="right")
    t.add_column("% of All", style="dim",          width=12, justify="right")
    for cn in class_names:
        t.add_column(cn.title(), style="green",    width=10, justify="right")

    rows_data = [
        ("Train",     train_ds, ds_cfg["train_split"]),
        ("Validation", val_ds,  ds_cfg["val_split"]),
        ("Test",       test_ds, ds_cfg["test_split"]),
    ]
    for split_name, ds, frac in rows_data:
        counts = _class_counts(ds)
        row = [split_name, str(len(ds)), f"{frac*100:.0f}%"]
        for i in range(len(class_names)):
            row.append(str(counts.get(i, 0)))
        t.add_row(*row)

    t.add_section()
    total_per_class = _class_counts(train_ds)
    tc = Counter(total_per_class)
    for ds in [val_ds, test_ds]:
        tc += _class_counts(ds)
    row = ["TOTAL", str(total), "100%"] + [str(tc.get(i,0)) for i in range(len(class_names))]
    t.add_row(*row)

    console.print(t)

    # Augmentation table
    aug = ds_cfg["augmentation"]
    at = Table(title="[bold cyan]Training Data Augmentation[/bold cyan]",
               border_style="cyan", show_header=True,
               header_style="bold cyan", min_width=60)
    at.add_column("Augmentation",    style="key",   min_width=30)
    at.add_column("Value",           style="value", min_width=20)
    for k, v in aug.items():
        at.add_row(k.replace("_", " ").title(), str(v))
    at.add_row("Normalisation (mean)",  "[0.485, 0.456, 0.406]")
    at.add_row("Normalisation (std)",   "[0.229, 0.224, 0.225]")
    console.print(at)
    console.print()
