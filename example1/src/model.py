"""
model.py — SimpleCNN definition built from config.json.

Architecture:
  N x [Conv2d -> BatchNorm -> ReLU -> MaxPool]  (feature extractor)
  Flatten
  Linear -> ReLU -> Dropout
  Linear (output logits, one per class)

Each layer is fully described so training.py can print its purpose to the user.
"""

import math
import torch
import torch.nn as nn
from rich.table import Table
from rich.console import Console
from rich.panel import Panel

console = Console()


# ── Layer-level description catalogue ────────────────────────────────────────
LAYER_DOCS = {
    "Conv2d": (
        "Learnable 2-D convolution filter. Slides a {ks}x{ks} kernel over the input "
        "to detect local spatial patterns (edges, textures, shapes). "
        "Input channels: {ic} -> Output feature maps: {oc}. "
        "Padding={pad} keeps spatial size the same."
    ),
    "BatchNorm2d": (
        "Batch Normalisation. Normalises each feature map to zero mean and unit variance "
        "across the mini-batch. Stabilises training, allows higher learning rates, "
        "and acts as a mild regulariser."
    ),
    "ReLU": (
        "Rectified Linear Unit activation: f(x)=max(0,x). Introduces non-linearity "
        "so the network can learn complex functions. Fast and avoids vanishing gradients."
    ),
    "MaxPool2d": (
        "Max Pooling ({ps}x{ps}, stride={ps}). Takes the maximum value in each window, "
        "halving H and W. Reduces computation, provides translation invariance, "
        "and helps prevent overfitting."
    ),
    "Flatten": (
        "Converts the 3-D feature maps [C, H, W] into a 1-D vector so fully-connected "
        "layers can process them."
    ),
    "Linear_fc": (
        "Fully-Connected (Dense) layer: {ic} -> {oc} units. Combines all extracted "
        "features and learns high-level abstract relationships."
    ),
    "Dropout": (
        "Dropout (p={p}). During training, randomly zeroes {pct}% of neurons. "
        "Prevents co-adaptation of features and is the primary overfitting defence."
    ),
    "Linear_out": (
        "Output layer: {ic} -> {oc} logits (one per class). No activation here — "
        "CrossEntropyLoss applies Softmax internally for numerical stability."
    ),
}


class ConvBlock(nn.Module):
    """Conv2d + optional BN + ReLU + MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 padding: int, use_bn: bool, pool_size: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              padding=padding, bias=not use_bn)
        self.bn   = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size, stride=pool_size)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class SimpleCNN(nn.Module):
    """
    Configurable Simple CNN for binary (or multi-class) image classification.
    Architecture is fully driven by the 'model' section of config.json.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg        = cfg
        self.num_classes = cfg["num_classes"]
        self.class_names = cfg["class_names"]
        in_ch           = cfg["in_channels"]
        h, w            = cfg["input_size"]

        # ── Build convolutional blocks ────────────────────────────────────────
        conv_layers = []
        for blk in cfg["conv_blocks"]:
            conv_layers.append(ConvBlock(
                in_ch       = in_ch,
                out_ch      = blk["out_channels"],
                kernel_size = blk["kernel_size"],
                padding     = blk["padding"],
                use_bn      = blk["use_batchnorm"],
                pool_size   = blk["pool_size"],
            ))
            in_ch = blk["out_channels"]
        self.features = nn.Sequential(*conv_layers)

        # ── Compute flattened feature size ────────────────────────────────────
        n_pools      = len(cfg["conv_blocks"])
        feat_h       = h // (2 ** n_pools)
        feat_w       = w // (2 ** n_pools)
        flat_size    = in_ch * feat_h * feat_w
        self._flat   = flat_size         # expose for summary

        # ── Classifier head ───────────────────────────────────────────────────
        fc_units = cfg["fc_hidden_units"]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg["dropout_rate"]),
            nn.Linear(fc_units, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    # ── Convenience helpers ───────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_summary(self) -> list[dict]:
        """Returns a structured list describing every layer — used by training.py."""
        cfg     = self.cfg
        in_ch   = cfg["in_channels"]
        h, w    = cfg["input_size"]
        rows    = []
        block_num = 0

        for blk in cfg["conv_blocks"]:
            block_num += 1
            out_ch = blk["out_channels"]
            ks     = blk["kernel_size"]
            pad    = blk["padding"]
            ps     = blk["pool_size"]
            out_h  = h // ps
            out_w  = w // ps

            rows.append({
                "id":    f"Conv Block {block_num}",
                "type":  "Conv2d",
                "shape": f"[{in_ch}, {h}, {w}] -> [{out_ch}, {h}, {w}]",
                "params": out_ch * in_ch * ks * ks + (out_ch if not blk["use_batchnorm"] else 0),
                "why": LAYER_DOCS["Conv2d"].format(ks=ks, ic=in_ch, oc=out_ch, pad=pad),
            })
            if blk["use_batchnorm"]:
                rows.append({
                    "id":    f"  BatchNorm {block_num}",
                    "type":  "BatchNorm2d",
                    "shape": f"[{out_ch}, {h}, {w}] (unchanged)",
                    "params": out_ch * 2,
                    "why": LAYER_DOCS["BatchNorm2d"],
                })
            rows.append({
                "id":    f"  ReLU {block_num}",
                "type":  "ReLU",
                "shape": f"[{out_ch}, {h}, {w}] (unchanged)",
                "params": 0,
                "why": LAYER_DOCS["ReLU"],
            })
            rows.append({
                "id":    f"  MaxPool {block_num}",
                "type":  "MaxPool2d",
                "shape": f"[{out_ch}, {h}, {w}] -> [{out_ch}, {out_h}, {out_w}]",
                "params": 0,
                "why": LAYER_DOCS["MaxPool2d"].format(ps=ps),
            })
            in_ch = out_ch
            h, w  = out_h, out_w

        fc_units = cfg["fc_hidden_units"]
        flat_size = self._flat

        rows.append({
            "id":    "Flatten",
            "type":  "Flatten",
            "shape": f"[{in_ch}, {h}, {w}] -> [{flat_size}]",
            "params": 0,
            "why": LAYER_DOCS["Flatten"],
        })
        rows.append({
            "id":    "FC Hidden",
            "type":  "Linear",
            "shape": f"[{flat_size}] -> [{fc_units}]",
            "params": flat_size * fc_units + fc_units,
            "why": LAYER_DOCS["Linear_fc"].format(ic=flat_size, oc=fc_units),
        })
        rows.append({
            "id":    "  ReLU",
            "type":  "ReLU",
            "shape": f"[{fc_units}] (unchanged)",
            "params": 0,
            "why": LAYER_DOCS["ReLU"],
        })
        rows.append({
            "id":    "  Dropout",
            "type":  "Dropout",
            "shape": f"[{fc_units}] (unchanged)",
            "params": 0,
            "why": LAYER_DOCS["Dropout"].format(
                p=cfg["dropout_rate"], pct=int(cfg["dropout_rate"]*100)),
        })
        rows.append({
            "id":    "Output (Logits)",
            "type":  "Linear",
            "shape": f"[{fc_units}] -> [{self.num_classes}]",
            "params": fc_units * self.num_classes + self.num_classes,
            "why": LAYER_DOCS["Linear_out"].format(ic=fc_units, oc=self.num_classes),
        })
        return rows


def build_model(cfg: dict) -> "SimpleCNN":
    return SimpleCNN(cfg)


def print_model_layers(model: SimpleCNN):
    """Pretty-print each layer with shape flow and purpose."""
    from rich.console import Console
    console = Console()

    console.print()
    t = Table(
        title="[bold cyan]CNN Architecture — Layer by Layer[/bold cyan]",
        border_style="cyan",
        show_lines=True,
        min_width=120,
    )
    t.add_column("#",       style="dim",        width=18)
    t.add_column("Type",    style="bold cyan",  width=14)
    t.add_column("Shape Flow",     style="green",       width=30)
    t.add_column("Trainable\nParams", style="yellow",   width=12, justify="right")
    t.add_column("Purpose", style="white",      min_width=40)

    for row in model.layer_summary():
        t.add_row(
            row["id"],
            row["type"],
            row["shape"],
            f"{row['params']:,}" if row["params"] else "—",
            row["why"][:120],        # truncate to fit terminal
        )

    console.print(t)

    total  = model.count_parameters()
    console.print()
    kv = Table(border_style="cyan", show_header=False, min_width=50)
    kv.add_column(style="key",   min_width=30)
    kv.add_column(style="value", min_width=18)
    kv.add_row("Total Trainable Parameters",   f"{total:,}")
    kv.add_row("Model Size (approx.)",
               f"{total * 4 / 1024**2:.2f} MB  (float32)")
    kv.add_row("Input Tensor Shape",
               f"[B, {model.cfg['in_channels']}, "
               f"{model.cfg['input_size'][0]}, {model.cfg['input_size'][1]}]")
    kv.add_row("Output Tensor Shape",
               f"[B, {model.num_classes}]  (raw logits)")
    console.print(kv)


def load_model(path: str, cfg: dict, device: torch.device) -> SimpleCNN:
    model = build_model(cfg)
    ckpt  = torch.load(path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    return model
