#!/usr/bin/env python3
"""
download_dataset.py — Downloads and organises the Apple vs Pineapple dataset.

Two options:
  Option A: Auto-download from Kaggle
            Requires:  pip install kaggle
                       Set KAGGLE_USERNAME / KAGGLE_KEY env variables
                       or place ~/.kaggle/kaggle.json

  Option B: Manual instructions (no dependencies)

  Option C: Generate synthetic placeholder images (for pipeline testing)

Run: python download_dataset.py
"""

import os
import sys
import shutil
import zipfile
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "apple_vs_pineapple"

try:
    from rich.console import Console
    from rich.panel   import Panel
    from rich.prompt  import Confirm, IntPrompt
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False
    class _Console:
        def print(self, *a, **kw): print(*a)
    console = _Console()


def print_manual_instructions():
    console.print()
    console.print("[bold cyan]Manual Download Instructions[/bold cyan]")
    console.print()
    console.print("Recommended dataset options:")
    console.print()
    console.print("  Option 1 — Kaggle: Fruits 360")
    console.print("    https://www.kaggle.com/datasets/moltean/fruits")
    console.print("    After extracting, copy only the 'Apple' and 'Pineapple' folders.")
    console.print()
    console.print("  Option 2 — Kaggle: Apple vs Pineapple")
    console.print("    Search 'apple pineapple classification' on kaggle.com")
    console.print()
    console.print("3. Organise images into this structure:")
    console.print()
    console.print(f"   {DATA_DIR}/")
    console.print("       apple/")
    console.print("           apple_0001.jpg")
    console.print("           apple_0002.jpg  ...")
    console.print("       pineapple/")
    console.print("           pineapple_0001.jpg  ...")
    console.print()
    console.print("4. Then run:  python main.py  and choose Option 1 (Train)")
    console.print()
    console.print("[dim]Tip: 200-1000 images per class is sufficient for this example.[/dim]")


def download_fruits360_via_kaggle():
    """Download Fruits 360 dataset using the Kaggle API."""
    import subprocess
    import tempfile

    console.print("\n[cyan]Downloading Fruits 360 via Kaggle API...[/cyan]")
    tmp = Path(tempfile.mkdtemp())

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "moltean/fruits",
             "--path", str(tmp), "--unzip"],
            check=True, capture_output=True, text=True
        )
        console.print("[green]Download complete.[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Kaggle download failed:[/red] {e.stderr}")
        return False
    except FileNotFoundError:
        console.print("[red]'kaggle' command not found.[/red]  "
                      "Install with:  pip install kaggle")
        return False

    # Organise — find apple and pineapple folders inside the Fruits 360 structure
    console.print("[cyan]Organising images...[/cyan]")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "apple").mkdir(exist_ok=True)
    (DATA_DIR / "pineapple").mkdir(exist_ok=True)

    moved = {"apple": 0, "pineapple": 0}
    # Fruits360 has folders like "Apple Red 1", "Pineapple" etc.
    for src_dir in tmp.rglob("*"):
        if not src_dir.is_dir():
            continue
        name_lower = src_dir.name.lower()
        if "apple" in name_lower and "pineapple" not in name_lower:
            for img in src_dir.glob("*.jpg"):
                dest = DATA_DIR / "apple" / f"apple_{moved['apple']:04d}.jpg"
                shutil.copy2(img, dest)
                moved["apple"] += 1
        elif "pineapple" in name_lower:
            for img in src_dir.glob("*.jpg"):
                dest = DATA_DIR / "pineapple" / f"pineapple_{moved['pineapple']:04d}.jpg"
                shutil.copy2(img, dest)
                moved["pineapple"] += 1

    shutil.rmtree(tmp, ignore_errors=True)
    console.print(f"  Apples: {moved['apple']:,}   Pineapples: {moved['pineapple']:,}")
    console.print(f"  Saved to: [cyan]{DATA_DIR}[/cyan]")
    return moved["apple"] > 0 and moved["pineapple"] > 0


def verify_dataset():
    """Check dataset structure and report counts."""
    console.print("\n[bold cyan]Verifying dataset...[/bold cyan]")
    if not DATA_DIR.exists():
        console.print(f"[red]Not found:[/red] {DATA_DIR}")
        return False

    total = 0
    for cls_dir in sorted(DATA_DIR.iterdir()):
        if cls_dir.is_dir():
            imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + \
                   list(cls_dir.glob("*.jpeg"))
            console.print(f"  [green]{cls_dir.name}:[/green]  {len(imgs):,} images")
            total += len(imgs)

    if total == 0:
        console.print("[red]No images found.[/red]")
        return False

    console.print(f"  [bold]Total: {total:,} images[/bold]")
    if total < 200:
        console.print("[yellow]Warning: Very few images. Consider using at least 200/class.[/yellow]")
    else:
        console.print("[green]Dataset looks good![/green]")
    return True


def generate_synthetic_demo(n_per_class: int = 200):
    """
    Generate coloured-noise placeholder images so the pipeline can be tested
    without a real dataset.  DO NOT use for actual model evaluation.

    Apple  — warm red/green tones
    Pineapple — warm yellow/brown tones with a slight green spike pattern
    """
    try:
        from PIL import Image, ImageDraw
        import numpy as np
    except ImportError:
        console.print("[red]Pillow not installed.[/red]  pip install Pillow")
        return

    console.print(f"\n[yellow]Generating {n_per_class} synthetic placeholder images per class...[/yellow]")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    np_rng = __import__("numpy").random.RandomState(42)

    class_specs = [
        # (class_name, base_colour_rgb, noise_range)
        ("apple",     [180,  60,  60], 50),   # red-dominated
        ("pineapple", [210, 170,  50], 50),   # yellow/golden-dominated
    ]

    for cls, mean_colour, noise in class_specs:
        cls_dir = DATA_DIR / cls
        cls_dir.mkdir(exist_ok=True)
        for i in range(n_per_class):
            base  = np_rng.randint(
                max(0, mean_colour[0]-noise), min(255, mean_colour[0]+noise+1),
                (128, 128), dtype=np.uint8)
            g_ch  = np_rng.randint(
                max(0, mean_colour[1]-noise), min(255, mean_colour[1]+noise+1),
                (128, 128), dtype=np.uint8)
            b_ch  = np_rng.randint(
                max(0, mean_colour[2]-noise), min(255, mean_colour[2]+noise+1),
                (128, 128), dtype=np.uint8)
            img_arr = np.stack([base, g_ch, b_ch], axis=-1).astype(np.uint8)
            Image.fromarray(img_arr).save(str(cls_dir / f"{cls}_{i:04d}.jpg"))
        console.print(f"  [green]{cls}:[/green] {n_per_class} images")

    console.print(f"  Saved to: [cyan]{DATA_DIR}[/cyan]")
    console.print("  [dim]Note: These are noise images for pipeline testing only.[/dim]")


def main():
    console.print()
    console.print("[bold green]Example 2 — Apple vs Pineapple Dataset Setup[/bold green]")
    console.print()

    # Check if already present
    if verify_dataset():
        console.print("\n[green]Dataset already present. You can run main.py.[/green]")
        return

    console.print("\nChoose how to get the dataset:\n")
    console.print("  [bold cyan]1[/bold cyan]  Download Fruits 360 via Kaggle API (requires kaggle account + API key)")
    console.print("  [bold cyan]2[/bold cyan]  Show manual download instructions")
    console.print("  [bold cyan]3[/bold cyan]  Generate synthetic placeholder images (for testing pipeline)")
    console.print()

    choice = input("Enter 1 / 2 / 3: ").strip()

    if choice == "1":
        ok = download_fruits360_via_kaggle()
        if not ok:
            print_manual_instructions()
    elif choice == "2":
        print_manual_instructions()
    elif choice == "3":
        n = input("Images per class (default 200): ").strip()
        n = int(n) if n.isdigit() else 200
        generate_synthetic_demo(n)
    else:
        console.print("[red]Invalid choice.[/red]")

    verify_dataset()


if __name__ == "__main__":
    main()
