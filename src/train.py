from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import inspect


@dataclass
class DataBundle:
    train: Any
    val: Any
    test: Any
    label_names: list[str]


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_food101(val_split: float, seed: int) -> DataBundle:
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
    label_names = train_ds.features["label"].names
    return DataBundle(train=train_ds, val=val_ds, test=test_ds, label_names=label_names)


from typing import Dict, Any
from transformers import AutoImageProcessor

def build_preprocess(processor: AutoImageProcessor, image_size: int, train: bool):
    def fn(examples: Dict[str, Any]):
        images = examples["image"]
        new_images = []
        for img in images:
            if hasattr(img, "mode") and img.mode != "RGB":
                img = img.convert("RGB")
            new_images.append(img)

        inputs = processor(images=new_images, return_tensors="pt")
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
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--val_split", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=507)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--early_stopping_patience", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    data = load_food101(val_split=args.val_split, seed=args.seed)
    label_names = data.label_names
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    image_size = processor.size.get("shortest_edge", processor.size.get("height", 224))
    train_tfm = build_preprocess(processor, image_size=image_size, train=True)
    eval_tfm = build_preprocess(processor, image_size=image_size, train=False)
    train_ds = data.train.with_transform(train_tfm)
    val_ds = data.val.with_transform(eval_tfm)

    args_tr = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    logging_steps=50,
    fp16=args.fp16,
    report_to="none",
    remove_unused_columns=False,
    )



    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(val_ds)

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    import json
    with open(os.path.join(args.output_dir, "metrics_val.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
