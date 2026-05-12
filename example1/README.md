# Example 1 — Cat vs Dog CNN Image Classifier

Simple CNN trained from scratch for binary image classification.

## Folder Structure

```
example1/
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
│   └── cats_and_dogs/
│       ├── cat/  *.jpg
│       └── dog/  *.jpg
├── checkpoints/             ← Saved .pth files
├── results/                 ← Training curves, confusion matrix
└── logs/                    ← Training logs
```

## Quick Start

### 1. Get the dataset
```bash
python download_dataset.py
```

### 2. Run
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
| model | conv_blocks | 4 blocks | Conv layers — edit to add/remove layers |
| model | fc_hidden_units | 512 | FC hidden layer size |
| model | dropout_rate | 0.5 | Dropout probability |
| training | batch_size | 32 | Mini-batch size |
| training | num_epochs | 50 | Max training epochs |
| training | learning_rate | 0.001 | Initial LR |
| training | early_stopping.patience | 10 | Epochs without improvement before stopping |
| training | checkpoint.save_every_n_epochs | 5 | Periodic checkpoint frequency |
| dataset | train_split | 0.70 | Fraction used for training |
| dataset | val_split | 0.15 | Fraction used for validation |
