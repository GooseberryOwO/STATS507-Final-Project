
from __future__ import annotations
import argparse
import os
import json

import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def get_dataset(split: str):
    """
    Food-101 官方只有 train / validation，没有 test。
    我们约定:
      split == "test" -> 用 validation
      split == "val"  -> 也是 validation
      split == "train"-> train
    """
    ds = load_dataset("food101")
    if split == "train":
        return ds["train"]
    else:
        if "test" in ds:
            return ds["test"]
        else:
            return ds["validation"]


def make_collate_fn(processor):
    def collate_fn(batch):
        images = []
        labels = []
        for ex in batch:
            img = ex["image"]
            if hasattr(img, "mode") and img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
            labels.append(ex["label"])
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        labels = torch.tensor(labels, dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}
    return collate_fn


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    processor = AutoImageProcessor.from_pretrained(args.run_dir)
    model = AutoModelForImageClassification.from_pretrained(args.run_dir)
    model.to(device)
    model.eval()

    ds = get_dataset(args.split)

    from torch.utils.data import DataLoader

    collate_fn = make_collate_fn(processor)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_logits = []
    all_labels = []

    print(f"Running inference on split='{args.split}' with {len(ds)} examples...")
    for batch in tqdm(loader):
        labels = batch["labels"]
        pixel_values = batch["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits.detach().cpu()

        all_logits.append(logits)
        all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    preds = logits.argmax(axis=-1)
    top5_idx = np.argsort(logits, axis=-1)[:, -5:]

    acc = (preds == labels).mean()
    top5_acc = np.mean(
        [label in row for label, row in zip(labels, top5_idx)]
    )

    print(f"Accuracy: {acc:.4f}, Top-5 accuracy: {top5_acc:.4f}")

    summary = {
        "split": args.split,
        "n": int(len(labels)),
        "accuracy": float(acc),
        "top5_accuracy": float(top5_acc),
    }
    os.makedirs(args.run_dir, exist_ok=True)
    summary_path = os.path.join(args.run_dir, f"pred_summary_{args.split}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary to:", summary_path)

    df = pd.DataFrame(
        {
            "label": labels,
            "pred": preds,
        }
    )
    pred_path = os.path.join(args.run_dir, f"predictions_{args.split}.parquet")
    df.to_parquet(pred_path)
    print("Saved predictions to:", pred_path)


if __name__ == "__main__":
    main()
