#!/usr/bin/env python3
"""
Create interpretability figures from saved attention records.

Input files are produced by model_training.py and stored as:
  attn_records/<mode>/*.npz

Each .npz contains:
  - attn: (N, P, L) float16/float32, cross-attention averaged over heads
  - token_ids: (N, L) int
  - labels: (N,) int
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer


DEFAULT_BERT_DIR = "/home/aispeech/codes/zxy/TPANet-main/TPANet-main2/models/bert-base-uncased"
DEFAULT_MODES = ("original", "generic", "shuffle", "random")
NUMERIC_TOKEN_RE = re.compile(r"^(?:##)?\d+$")
SEMANTIC_TOKENS = {
    "statistics",
    "minimum",
    "maximum",
    "median",
    "trend",
    "upward",
    "downward",
    "classify",
    "emotion",
    "data",
    "value",
    "overall",
    "provided",
    "underlying",
}
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--records_root",
        default=str(Path(__file__).resolve().parents[1] / "attn_records"),
        help="Root directory containing per-mode attention record folders.",
    )
    ap.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parents[1] / "attn_figures"),
        help="Directory to save plotted figures and summary.",
    )
    ap.add_argument(
        "--bert_dir",
        default=DEFAULT_BERT_DIR,
        help="Local bert-base-uncased directory used for token id decoding.",
    )
    ap.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated prompt modes to compare, e.g. original,generic,shuffle,random.",
    )
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=30,
        help="Maximum token positions to display on the x-axis.",
    )
    ap.add_argument(
        "--class_names",
        default="",
        help="Optional comma-separated class names matching sorted label ids.",
    )
    return ap.parse_args()


def parse_modes(raw_modes):
    return [m.strip() for m in raw_modes.split(",") if m.strip()]


def load_mode_records(mode_dir):
    files = sorted(mode_dir.glob("*.npz"))
    attn_blocks = []
    token_blocks = []
    label_blocks = []
    for fp in files:
        with np.load(fp, allow_pickle=False) as data:
            attn_blocks.append(data["attn"].astype(np.float32))
            token_blocks.append(data["token_ids"].astype(np.int32))
            label_blocks.append(data["labels"].astype(np.int64))
    if not attn_blocks:
        return None
    return {
        "files": [str(f) for f in files],
        "attn": np.concatenate(attn_blocks, axis=0),
        "token_ids": np.concatenate(token_blocks, axis=0),
        "labels": np.concatenate(label_blocks, axis=0),
    }


def make_label_names(unique_labels, raw_class_names):
    if raw_class_names:
        names = [x.strip() for x in raw_class_names.split(",")]
        if len(names) == len(unique_labels):
            return {int(lbl): names[i] for i, lbl in enumerate(unique_labels)}
    return {int(lbl): f"Class {int(lbl)}" for lbl in unique_labels}


def tokens_for_mode(mode_data, tokenizer, max_tokens):
    token_ids = mode_data["token_ids"][0][:max_tokens].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return [t.replace("##", "") for t in tokens]


def set_token_ticks(ax, tokens):
    if not tokens:
        return
    stride = 2 if len(tokens) > 10 else 1
    ticks = list(range(0, len(tokens), stride))
    labels = [tokens[i] for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)


def get_map_range(mode_to_data, modes, labels, max_tokens):
    values = []
    for mode in modes:
        data = mode_to_data.get(mode)
        if data is None:
            continue
        for lbl in labels:
            mask = data["labels"] == lbl
            if np.any(mask):
                mean_map = data["attn"][mask].mean(axis=0)[:, :max_tokens]
                values.append(mean_map)
    if not values:
        return 0.0, 1.0
    stacked = np.concatenate([v.reshape(-1) for v in values], axis=0)
    return float(np.min(stacked)), float(np.max(stacked))


def plot_mode_class_heatmaps(mode_to_data, modes, labels, label_name_map, tokenizer, max_tokens, out_path):
    n_rows = len(modes)
    n_cols = len(labels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * max(1, n_cols), 2.8 * max(1, n_rows)),
        squeeze=False,
        constrained_layout=True,
    )
    vmin, vmax = get_map_range(mode_to_data, modes, labels, max_tokens)
    last_im = None
    for row, mode in enumerate(modes):
        data = mode_to_data.get(mode)
        ref_tokens = tokens_for_mode(data, tokenizer, max_tokens) if data is not None else []
        for col, lbl in enumerate(labels):
            ax = axes[row, col]
            if data is None:
                ax.text(0.5, 0.5, "No records", ha="center", va="center")
                ax.set_axis_off()
                continue
            mask = data["labels"] == lbl
            if not np.any(mask):
                ax.text(0.5, 0.5, "No samples", ha="center", va="center")
                ax.set_axis_off()
                continue
            mean_map = data["attn"][mask].mean(axis=0)[:, :max_tokens]
            last_im = ax.imshow(mean_map, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(f"{mode} | {label_name_map[int(lbl)]}")
            ax.set_xlabel("Prompt token")
            ax.set_ylabel("EEG patch")
            set_token_ticks(ax, ref_tokens)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.01)
    fig.suptitle("Cross-attention Heatmap (mode x class, avg over heads and samples)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_delta_original_random(mode_to_data, labels, label_name_map, tokenizer, max_tokens, out_path):
    if "original" not in mode_to_data or "random" not in mode_to_data:
        return
    original = mode_to_data["original"]
    random_mode = mode_to_data["random"]
    fig, axes = plt.subplots(
        1,
        len(labels),
        figsize=(4.2 * max(1, len(labels)), 3.2),
        squeeze=False,
        constrained_layout=True,
    )
    ref_tokens = tokens_for_mode(original, tokenizer, max_tokens)
    delta_maps = []
    valid_labels = []
    for lbl in labels:
        mask_o = original["labels"] == lbl
        mask_r = random_mode["labels"] == lbl
        if np.any(mask_o) and np.any(mask_r):
            m_o = original["attn"][mask_o].mean(axis=0)[:, :max_tokens]
            m_r = random_mode["attn"][mask_r].mean(axis=0)[:, :max_tokens]
            delta_maps.append(m_o - m_r)
            valid_labels.append(lbl)
    if not delta_maps:
        plt.close(fig)
        return
    vmax = max(float(np.max(np.abs(x))) for x in delta_maps)
    last_im = None
    for col, lbl in enumerate(labels):
        ax = axes[0, col]
        if lbl not in valid_labels:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center")
            ax.set_axis_off()
            continue
        idx = valid_labels.index(lbl)
        dmap = delta_maps[idx]
        last_im = ax.imshow(dmap, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"original - random | {label_name_map[int(lbl)]}")
        ax.set_xlabel("Prompt token")
        ax.set_ylabel("EEG patch")
        set_token_ticks(ax, ref_tokens)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
    fig.suptitle("Attention Delta Map")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def token_group(token):
    if token in SPECIAL_TOKENS:
        return "special"
    lower = token.lower()
    if lower in {".", ",", "-", ":"} or NUMERIC_TOKEN_RE.fullmatch(lower):
        return "numeric"
    if lower in SEMANTIC_TOKENS:
        return "semantic"
    return "other"


def compute_group_scores(data, tokenizer):
    scores = {"semantic": [], "numeric": [], "special": [], "other": []}
    attn = data["attn"]
    token_ids = data["token_ids"]
    for i in range(attn.shape[0]):
        vec = attn[i].mean(axis=0)  # (L,)
        tokens = tokenizer.convert_ids_to_tokens(token_ids[i].tolist())
        group_mass = {"semantic": 0.0, "numeric": 0.0, "special": 0.0, "other": 0.0}
        for t_idx, token in enumerate(tokens):
            group = token_group(token)
            group_mass[group] += float(vec[t_idx])
        for key in scores:
            scores[key].append(group_mass[key])
    return scores


def plot_group_boxplots(mode_to_data, modes, tokenizer, out_path):
    groups = ["semantic", "numeric"]
    mode_to_scores = {}
    for mode in modes:
        if mode in mode_to_data:
            mode_to_scores[mode] = compute_group_scores(mode_to_data[mode], tokenizer)

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(5.2 * len(groups), 4.0),
        squeeze=False,
        constrained_layout=True,
    )
    for idx, group in enumerate(groups):
        ax = axes[0, idx]
        series = []
        labels = []
        for mode in modes:
            if mode in mode_to_scores and mode_to_scores[mode][group]:
                series.append(mode_to_scores[mode][group])
                labels.append(mode)
        if not series:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center")
            ax.set_axis_off()
            continue
        try:
            ax.boxplot(series, tick_labels=labels, showmeans=True)
        except TypeError:
            ax.boxplot(series, labels=labels, showmeans=True)
        ax.set_title(f"{group} token attention mass")
        ax.set_ylabel("Attention mass")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Token Group Attention Distribution")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    modes = parse_modes(args.modes)
    records_root = Path(args.records_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir, local_files_only=True)

    mode_to_data = {}
    for mode in modes:
        mode_dir = records_root / mode
        loaded = load_mode_records(mode_dir)
        if loaded is not None:
            mode_to_data[mode] = loaded

    if not mode_to_data:
        raise RuntimeError(
            f"No attention record files found under {records_root}. "
            "Run model_training.py for at least one prompt mode first."
        )

    all_labels = sorted(
        set(int(x) for data in mode_to_data.values() for x in np.unique(data["labels"]).tolist())
    )
    label_name_map = make_label_names(all_labels, args.class_names)

    plot_mode_class_heatmaps(
        mode_to_data=mode_to_data,
        modes=modes,
        labels=all_labels,
        label_name_map=label_name_map,
        tokenizer=tokenizer,
        max_tokens=max(1, int(args.max_tokens)),
        out_path=out_dir / "attn_mode_class_heatmaps.png",
    )

    plot_delta_original_random(
        mode_to_data=mode_to_data,
        labels=all_labels,
        label_name_map=label_name_map,
        tokenizer=tokenizer,
        max_tokens=max(1, int(args.max_tokens)),
        out_path=out_dir / "attn_delta_original_minus_random.png",
    )

    plot_group_boxplots(
        mode_to_data=mode_to_data,
        modes=modes,
        tokenizer=tokenizer,
        out_path=out_dir / "attn_token_group_boxplots.png",
    )

    summary = {
        "records_root": str(records_root),
        "modes_requested": modes,
        "modes_loaded": sorted(mode_to_data.keys()),
        "class_ids": [int(x) for x in all_labels],
        "class_name_map": {str(k): v for k, v in label_name_map.items()},
        "samples_per_mode": {mode: int(data["attn"].shape[0]) for mode, data in mode_to_data.items()},
        "files_per_mode": {mode: data["files"] for mode, data in mode_to_data.items()},
        "outputs": {
            "mode_class_heatmaps": str(out_dir / "attn_mode_class_heatmaps.png"),
            "delta_original_minus_random": str(out_dir / "attn_delta_original_minus_random.png"),
            "token_group_boxplots": str(out_dir / "attn_token_group_boxplots.png"),
        },
    }
    with open(out_dir / "attn_interpretability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
