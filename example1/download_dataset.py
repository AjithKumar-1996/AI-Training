#!/usr/bin/env python3
"""
download_dataset.py — Downloads and organises the Cats vs Dogs dataset.

Two options:
  Option A: Auto-download Microsoft's Kaggle Cats-vs-Dogs subset (~800 MB)
            Requires:  pip install kaggle
                       Set KAGGLE_USERNAME / KAGGLE_KEY env variables
                       or place ~/.kaggle/kaggle.json

  Option B: Manual instructions (no dependencies)

Run: python download_dataset.py
"""

import os
import sys
import shutil
import zipfile
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "cats_and_dogs"

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
    console.print("1. Go to:  https://www.kaggle.com/datasets/salader/dogs-vs-cats")
    console.print("   OR:     https://www.microsoft.com/en-us/download/details.aspx?id=54765")
    console.print()
    console.print("2. Download and extract the zip.")
    console.print()
    console.print("3. Organise images into this structure:")
    console.print()
    console.print(f"   {DATA_DIR}/")
    console.print("       cat/")
    console.print("           cat.0.jpg")
    console.print("           cat.1.jpg  ...")
    console.print("       dog/")
    console.print("           dog.0.jpg  ...")
    console.print()
    console.print("4. Then run:  python main.py  and choose Option 1 (Train)")
    console.print()
    console.print("[dim]Tip: 1000-5000 images per class is sufficient for this example.[/dim]")


def download_via_kaggle():
    """Download cats-vs-dogs using the Kaggle API."""
    import subprocess
    import tempfile

    console.print("\n[cyan]Downloading via Kaggle API...[/cyan]")
    tmp = Path(tempfile.mkdtemp())

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "salader/dogs-vs-cats",
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

    # Organise images
    console.print("[cyan]Organising images...[/cyan]")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "cat").mkdir(exist_ok=True)
    (DATA_DIR / "dog").mkdir(exist_ok=True)

    moved = {"cat": 0, "dog": 0}
    for img in tmp.rglob("*.jpg"):
        name = img.name.lower()
        if name.startswith("cat"):
            shutil.copy2(img, DATA_DIR / "cat" / img.name)
            moved["cat"] += 1
        elif name.startswith("dog"):
            shutil.copy2(img, DATA_DIR / "dog" / img.name)
            moved["dog"] += 1

    shutil.rmtree(tmp, ignore_errors=True)
    console.print(f"  Cats: {moved['cat']:,}   Dogs: {moved['dog']:,}")
    console.print(f"  Saved to: [cyan]{DATA_DIR}[/cyan]")
    return True


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
        console.print("[yellow]Warning: Very few images. Consider using at least 500/class.[/yellow]")
    else:
        console.print("[green]Dataset looks good![/green]")
    return True


def generate_synthetic_demo(n_per_class: int = 100):
    """
    Generate coloured-noise placeholder images so the pipeline can be tested
    without a real dataset.  DO NOT use for actual model evaluation.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        console.print("[red]Pillow not installed.[/red]  pip install Pillow")
        return

    console.print(f"\n[yellow]Generating {n_per_class} synthetic placeholder images per class...[/yellow]")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for cls, mean_colour in [("cat", [200, 130, 80]), ("dog", [100, 140, 180])]:
        cls_dir = DATA_DIR / cls
        cls_dir.mkdir(exist_ok=True)
        for i in range(n_per_class):
            noise = np.random.randint(0, 60, (224, 224, 3), dtype=np.uint8)
            base  = np.array(mean_colour, dtype=np.uint8).reshape(1, 1, 3)
            img   = np.clip(base + noise - 30, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(str(cls_dir / f"{cls}_{i:04d}.jpg"))
        console.print(f"  [green]{cls}:[/green] {n_per_class} images")

    console.print(f"  Saved to: [cyan]{DATA_DIR}[/cyan]")
    console.print("  [dim]Note: These are noise images for pipeline testing only.[/dim]")


def main():
    console.print()
    console.print("[bold blue]Example 1 — Dataset Setup[/bold blue]")
    console.print()

    # Check if already present
    if verify_dataset():
        console.print("\n[green]Dataset already present. You can run main.py.[/green]")
        return

    console.print("\nChoose how to get the dataset:\n")
    console.print("  [bold cyan]1[/bold cyan]  Download via Kaggle API (requires kaggle account + API key)")
    console.print("  [bold cyan]2[/bold cyan]  Show manual download instructions")
    console.print("  [bold cyan]3[/bold cyan]  Generate synthetic placeholder images (for testing pipeline)")
    console.print()

    choice = input("Enter 1 / 2 / 3: ").strip()

    if choice == "1":
        ok = download_via_kaggle()
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
