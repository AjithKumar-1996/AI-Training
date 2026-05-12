#!/usr/bin/env python3
"""
main.py — Entry point for Example 2: Apple vs Pineapple CNN Classifier.

Usage:
    python main.py                   # uses config.json in same folder
    python main.py --config my.json  # custom config path

Presents a menu:
    1. Train a new model
    2. Retrain / resume from an existing model
    3. Run inference on an existing model
    4. Exit
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# ── Make sure we can import src/ regardless of CWD ───────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rich.console import Console
from rich.panel   import Panel
from rich.prompt  import Prompt, IntPrompt, Confirm
from rich.table   import Table
from rich.text    import Text

from src.utils   import (load_config, setup_logger, print_banner,
                          section, console, ask_path, confirm)
from src.train   import run_training
from src.inference import run_inference


# ═══════════════════════════════════════════════════════════════════════════════
# Menu
# ═══════════════════════════════════════════════════════════════════════════════
def show_menu() -> int:
    console.print()
    t = Table(border_style="green", show_header=False, min_width=55, padding=(0, 2))
    t.add_column(style="bold cyan",  width=5,  justify="center")
    t.add_column(style="bold white", min_width=40)
    t.add_row("1", "Train a new model from scratch")
    t.add_row("2", "Retrain / Resume from an existing checkpoint")
    t.add_row("3", "Run inference with an existing model")
    t.add_row("4", "Exit")
    console.print(t)
    console.print()
    choice = IntPrompt.ask(
        "[bold yellow]Select an option[/bold yellow]",
        choices=["1", "2", "3", "4"],
        default=1,
    )
    return choice


# ═══════════════════════════════════════════════════════════════════════════════
# Option 1 — Train new model
# ═══════════════════════════════════════════════════════════════════════════════
def flow_train_new(cfg: dict):
    section("TRAIN NEW MODEL")
    console.print("  [dim]A fresh model will be initialised with random weights.[/dim]")
    console.print(f"  [key]Dataset dir:[/key] [cyan]{cfg['dataset']['data_dir']}[/cyan]")
    run_training(cfg, resume_path=None)


# ═══════════════════════════════════════════════════════════════════════════════
# Option 2 — Retrain / Resume
# ═══════════════════════════════════════════════════════════════════════════════
def flow_retrain(cfg: dict):
    section("RETRAIN / RESUME FROM CHECKPOINT")

    # Find available checkpoints
    ck_dir = cfg["training"]["checkpoint"]["save_dir"]
    ck_dir_abs = SCRIPT_DIR / ck_dir
    existing = sorted(glob.glob(str(ck_dir_abs / "*.pth")))

    if existing:
        console.print("  [bold cyan]Available checkpoints:[/bold cyan]")
        t = Table(border_style="cyan", show_header=True,
                  header_style="bold cyan", min_width=70)
        t.add_column("#",    style="dim", width=5, justify="right")
        t.add_column("File", style="cyan", min_width=45)
        t.add_column("Size", style="dim", width=10, justify="right")
        for i, p in enumerate(existing, 1):
            sz = os.path.getsize(p) / 1024**2
            t.add_row(str(i), Path(p).name, f"{sz:.1f} MB")
        console.print(t)
        console.print()
        console.print("  Enter the checkpoint [bold]number[/bold] from the table above,")
        console.print("  or enter a [bold]full path[/bold] to a .pth file.")
        raw = Prompt.ask("[bold cyan]Checkpoint[/bold cyan]").strip().strip('"').strip("'")
        if raw.isdigit() and 1 <= int(raw) <= len(existing):
            resume_path = existing[int(raw) - 1]
        else:
            resume_path = raw
    else:
        console.print("  [warning]No checkpoints found in[/warning] "
                      f"[cyan]{ck_dir}[/cyan].")
        console.print("  Please enter the full path to a .pth checkpoint file:")
        resume_path = str(ask_path("Checkpoint path", must_exist=True))

    if not os.path.isfile(resume_path):
        console.print(f"  [error]File not found:[/error] {resume_path}")
        return

    console.print(f"  [success]Selected:[/success] [cyan]{Path(resume_path).name}[/cyan]")
    run_training(cfg, resume_path=resume_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Option 3 — Inference
# ═══════════════════════════════════════════════════════════════════════════════
def flow_inference(cfg: dict):
    section("INFERENCE")

    # Model selection
    ck_dir = cfg["training"]["checkpoint"]["save_dir"]
    best   = str(SCRIPT_DIR / ck_dir / "best_model.pth")
    if os.path.isfile(best):
        console.print(f"  [dim]Default best model found:[/dim] [cyan]{best}[/cyan]")
        use_best = Confirm.ask(
            "[bold yellow]Use best_model.pth?[/bold yellow]", default=True)
        if use_best:
            model_path = best
        else:
            model_path = str(ask_path("Model .pth path", must_exist=True))
    else:
        console.print("  [warning]No best_model.pth found.[/warning]")
        model_path = str(ask_path("Model .pth path", must_exist=True))

    # Input selection
    console.print()
    console.print("  Provide an [bold]image file[/bold] or a [bold]folder[/bold].")
    console.print("  For metrics (F1, confusion matrix), use an ImageFolder layout:")
    console.print("    input_folder/")
    console.print("      apple/  *.jpg")
    console.print("      pineapple/  *.jpg")
    console.print()
    input_path = str(ask_path("Input image or folder", must_exist=True))

    run_inference(cfg, model_path, input_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Example 2 — Apple vs Pineapple CNN")
    parser.add_argument("--config", default="config.json",
                        help="Path to config JSON (default: config.json)")
    args = parser.parse_args()

    # Change working dir to script location so relative paths in config work
    os.chdir(SCRIPT_DIR)

    # Load config
    cfg_path = args.config
    if not os.path.isfile(cfg_path):
        print(f"ERROR: Config file not found: {cfg_path}")
        sys.exit(1)
    cfg = load_config(cfg_path)

    # Setup logging
    logger = setup_logger(cfg["logging"]["log_dir"])

    print_banner()
    console.print(f"  [key]Config:[/key] [cyan]{cfg_path}[/cyan]")
    console.print(f"  [key]Classes:[/key] {cfg['model']['class_names']}")
    console.print(f"  [key]Input size:[/key] {cfg['model']['input_size']}")
    console.print(f"  [key]Dataset:[/key] [cyan]{cfg['dataset']['data_dir']}[/cyan]")

    while True:
        choice = show_menu()
        if choice == 1:
            flow_train_new(cfg)
        elif choice == 2:
            flow_retrain(cfg)
        elif choice == 3:
            flow_inference(cfg)
        elif choice == 4:
            console.print("[bold cyan]Goodbye![/bold cyan]")
            break

        console.print()
        if not Confirm.ask("[bold yellow]Return to main menu?[/bold yellow]", default=True):
            console.print("[bold cyan]Goodbye![/bold cyan]")
            break


if __name__ == "__main__":
    main()
