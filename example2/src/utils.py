"""
utils.py — Shared utilities: console styling, logging setup, seed, device selection.
"""

import os
import sys
import json
import logging
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.theme import Theme

# ── Global console with custom theme ─────────────────────────────────────────
THEME = Theme({
    "info":    "bold cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error":   "bold red",
    "title":   "bold white on blue",
    "section": "bold magenta",
    "key":     "bold cyan",
    "value":   "white",
    "dim":     "dim white",
})
console = Console(theme=THEME)


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logger(log_dir: str, name: str = "example2") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)

    return logger


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Device selection ──────────────────────────────────────────────────────────
def get_device(cfg_device: str = "auto") -> torch.device:
    if cfg_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name   = torch.cuda.get_device_name(0)
        else:
            device = torch.device("cpu")
            name   = "CPU"
    else:
        device = torch.device(cfg_device)
        name   = str(device)
    console.print(f"  [key]Device:[/key] [success]{name}[/success]")
    return device


# ── Config loader ─────────────────────────────────────────────────────────────
def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def save_config_snapshot(cfg: dict, out_dir: str):
    """Save a timestamped copy of the config alongside training results."""
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"config_{ts}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


# ── Banner ────────────────────────────────────────────────────────────────────
def print_banner():
    console.print()
    console.print(Panel.fit(
        Text.assemble(
            ("  Example 2 — Apple vs Pineapple CNN Classifier  \n", "bold white"),
            ("  Simple CNN for Binary Image Classification  ", "dim white"),
        ),
        border_style="green",
        padding=(1, 4),
    ))
    console.print()


# ── Section header ─────────────────────────────────────────────────────────────
def section(title: str):
    console.print()
    console.rule(f"[section]{title}[/section]")
    console.print()


# ── Key-value table ────────────────────────────────────────────────────────────
def kv_table(title: str, rows: list[tuple], accent: str = "cyan") -> Table:
    t = Table(title=title, border_style=accent, show_header=True,
              header_style=f"bold {accent}", min_width=60)
    t.add_column("Parameter", style="key",   min_width=30)
    t.add_column("Value",     style="value", min_width=25)
    for k, v in rows:
        t.add_row(str(k), str(v))
    return t


# ── Confirm helper ─────────────────────────────────────────────────────────────
def confirm(prompt_text: str, default: bool = True) -> bool:
    from rich.prompt import Confirm
    return Confirm.ask(f"[bold yellow]{prompt_text}[/bold yellow]", default=default)


# ── Path prompt ────────────────────────────────────────────────────────────────
def ask_path(prompt_text: str, must_exist: bool = True) -> Path:
    from rich.prompt import Prompt
    while True:
        raw = Prompt.ask(f"[bold cyan]{prompt_text}[/bold cyan]").strip().strip('"').strip("'")
        p = Path(raw)
        if must_exist and not p.exists():
            console.print(f"  [error]Path not found:[/error] {p}. Please try again.")
        else:
            return p
