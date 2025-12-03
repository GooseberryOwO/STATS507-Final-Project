from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["test", "val"])
    p.add_argument("--topk_pairs", type=int, default=15)
    p.add_argument("--examples", type=int, default=4)
    return p.parse_args()


def load_trainer_state(run_dir: Path):
    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        print(f"[make_figures] trainer_state.json not found at {state_path}, skip learning curves.")
        return None
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_curves(state: dict):
    hist = state.get("log_history", [])
    train_rows = [h for h in hist if "loss" in h and "eval_loss" not in h]
    eval_rows = [h for h in hist if "eval_loss" in h]

    train_step = [r.get("epoch", r.get("step", i)) for i, r in enumerate(train_rows)]
    train_loss = [r["loss"] for r in train_rows]

    eval_step = [r.get("epoch", r.get("step", i)) for i, r in enumerate(eval_rows)]
    eval_loss = [r.get("eval_loss") for r in eval_rows]
    eval_acc = [r.get("eval_accuracy") for r in eval_rows]

    return (train_step, train_loss), (eval_step, eval_loss, eval_acc)


def plot_learning_curves(run_dir: Path, out_dir: Path):
    state = load_trainer_state(run_dir)
    if state is None:
        return None

    (tr_x, tr_l), (ev_x, ev_l, ev_a) = extract_curves(state)

    plt.figure()
    if tr_x:
        plt.plot(tr_x, tr_l, label="train loss")
    if ev_x:
        plt.plot(ev_x, ev_l, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("learning curves")
    plt.legend()
    out_path = out_dir / "learning_curves.pdf"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def ensure_label_names(pred_df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    """
    兼容新的 save_predictions.py（只有 label / pred），
    自动从 config 里读 id2label，补上 label_name / pred_name / top1_correct
    """
    df = pred_df.copy()
    if "top1_correct" not in df.columns:
        if "label" in df.columns and "pred" in df.columns:
            df["top1_correct"] = (df["label"] == df["pred"])
        else:
            raise ValueError("predictions parquet must contain 'label' and 'pred' columns.")
    if "label_name" in df.columns and "pred_name" in df.columns:
        return df

    config = AutoConfig.from_pretrained(run_dir)
    id2label = config.id2label 

    def map_id_to_name(x):
        try:
            return id2label[int(x)]
        except Exception:
            return str(x)

    df["label_name"] = df["label"].map(map_id_to_name)
    df["pred_name"] = df["pred"].map(map_id_to_name)
    return df


def plot_top_confusions(pred_df: pd.DataFrame, out_dir: Path, topk: int):
    df = ensure_label_names(pred_df, out_dir.parent)

    wrong = df[~df["top1_correct"]].copy()
    if wrong.empty:
        plt.figure()
        plt.text(0.5, 0.5, "no errors", ha="center", va="center")
        plt.axis("off")
        out_path = out_dir / "top_confusions.pdf"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path

    pairs = (
        wrong.groupby(["label_name", "pred_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(topk)
    )

    labels = (pairs["label_name"] + " -> " + pairs["pred_name"]).tolist()
    counts = pairs["count"].to_numpy()
    y = np.arange(len(labels))[::-1]

    plt.figure(figsize=(6, 0.3 * len(labels) + 2))
    plt.barh(y, counts[::-1])
    plt.yticks(y, labels[::-1], fontsize=7)
    plt.xlabel("count")
    plt.title("top confusion pairs")
    out_path = out_dir / "top_confusions.pdf"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = plot_learning_curves(run_dir, out_dir)

    pred_path = run_dir / f"predictions_{args.split}.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"missing {pred_path}, run save_predictions.py first")
    pred_df = pd.read_parquet(pred_path)

    conf = plot_top_confusions(pred_df, out_dir, topk=args.topk_pairs)

    if curves is not None:
        print("saved:", curves)
    else:
        print("learning curves skipped (no trainer_state.json)")
    print("saved:", conf)


if __name__ == "__main__":
    main()
