from __future__ import annotations
import argparse
import csv

from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="nutrition_lookup.csv")
    return p.parse_args()


def main():
    args = parse_args()
    ds = load_dataset("food101")
    train_ds = ds["train"]
    label_names = train_ds.features["label"].names
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "kcal", "protein", "fat", "carbs"])
        for name in label_names:
            writer.writerow([name, "", "", "", ""])


if __name__ == "__main__":
    main()
