# Example 2 — Apple vs Pineapple CNN Image Classifier

Simple CNN trained from scratch for binary image classification.
This example demonstrates how a CNN distinguishes between two visually distinct fruits — 
an Apple (round, smooth, red/green) and a Pineapple (oval, textured, yellow/brown with a crown).

## Folder Structure

```
example2/
├── main.py                  ← Entry point (run this)
├── config.json              ← All configurable parameters
├── download_dataset.py      ← Dataset setup helper
├── src/
│   ├── model.py             ← SimpleCNN definition
│   ├── dataset.py           ← Loading, splitting, augmentation
│   ├── train.py             ← Training loop, early stopping, checkpoints
│   ├── inference.py         ← Inference, metrics, annotations
│   └── utils.py             ← Console, logging, device helpers
├── data/
│   └── apple_vs_pineapple/
│       ├── apple/       *.jpg
│       └── pineapple/   *.jpg
├── checkpoints/             ← Saved .pth files
├── results/                 ← Training curves, confusion matrix
└── logs/                    ← Training logs
```

## Quick Start

### 1. Use the same virtual environment as Example 1
```bash
# From example1/ directory — venv already has torch, torchvision, rich, etc.
# Activate venv first (PowerShell):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
..\example1\venv\Scripts\activate

# OR call venv Python directly (no activation needed):
..\example1\venv\Scripts\python.exe main.py
```

### 2. Get the dataset
```bash
python download_dataset.py
```

Choose:
- **Option 1** — Kaggle Fruits 360 (recommended, ~90 MB)
- **Option 2** — Manual download instructions
- **Option 3** — Synthetic images for pipeline testing

### 3. Run
```bash
python main.py
```

Select from the menu:
- **1** — Train new model
- **2** — Retrain / resume from checkpoint
- **3** — Inference on saved model

## Config Reference (`config.json`)

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| model | class_names | ["apple", "pineapple"] | Class labels |
| model | conv_blocks | 4 blocks | Conv layers |
| model | fc_hidden_units | 512 | FC hidden layer size |
| model | dropout_rate | 0.5 | Dropout probability |
| training | batch_size | 8 | Mini-batch size (MX150 safe) |
| training | num_epochs | 50 | Max training epochs |
| training | learning_rate | 0.001 | Initial LR |
| training | early_stopping.patience | 10 | Epochs without improvement before stopping |
| dataset | data_dir | data/apple_vs_pineapple | Dataset folder |

## What the CNN Learns

| CNN Layer | What it detects for Apple | What it detects for Pineapple |
|-----------|--------------------------|-------------------------------|
| Conv Block 1 | Smooth curves, stem indent | Diamond skin texture, sharp lines |
| Conv Block 2 | Round shape, smooth skin colour | Oval/cylinder shape |
| Conv Block 3 | Red/green colour regions | Yellow/brown base colour |
| Conv Block 4 | Overall fruit silhouette | Crown spikes at top |
| FC Head | Combines all features → verdict | Combines all features → verdict |
