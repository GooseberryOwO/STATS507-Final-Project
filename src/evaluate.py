from __future__ import annotations
import argparse
import os
import json

import numpy as np
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer


def load_food101_splits(val_split: float, seed: int):
    ds = load_dataset("food101")
    train_ds = ds["train"]
    if "test" in ds:
        test_ds = ds["test"]
    elif "validation" in ds:
        test_ds = ds["validation"]
    else:
        raise KeyError(f"no test or validation split in dataset: {list(ds.keys())}")
    split = train_ds.train_test_split(test_size=val_split, seed=seed, stratify_by_column="label")
    train_ds = split["train"]
    val_ds = split["test"]
    return train_ds, val_ds, test_ds


def build_preprocess(processor, image_size: int):
    def fn(examples):
        images = examples["image"]
        inputs = processor(images=images, return_tensors="pt")
        result = {"pixel_values": inputs["pixel_values"]}
        if "label" in examples:
            result["labels"] = examples["label"]
        return result
    return fn


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, top_k_accuracy_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    top5 = top_k_accuracy_score(labels, logits, k=5, labels=list(range(logits.shape[1])))
    return {"accuracy": acc, "top5_accuracy": top5}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["test", "val"])
    p.add_argument("--val_split", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=507)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    train_ds, val_ds, test_ds = load_food101_splits(val_split=args.val_split, seed=args.seed)
    ds = test_ds if args.split == "test" else val_ds
    processor = AutoImageProcessor.from_pretrained(args.run_dir)
    image_size = processor.size.get("shortest_edge", processor.size.get("height", 224))
    tfm = build_preprocess(processor, image_size=image_size)
    ds = ds.with_transform(tfm)
    model = AutoModelForImageClassification.from_pretrained(args.run_dir)
    targs = TrainingArguments(
        output_dir=os.path.join(args.run_dir, "_eval_tmp"),
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, compute_metrics=compute_metrics)
    metrics = trainer.evaluate(ds)
    out_path = os.path.join(args.run_dir, f"metrics_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
