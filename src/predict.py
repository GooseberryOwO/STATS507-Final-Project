from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--nutrition_csv", type=str, default="nutrition_lookup.csv")
    p.add_argument("--top_k", type=int, default=5)
    return p.parse_args()


def load_nutrition(csv_path: str):
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        name = str(row["class_name"])
        mapping[name] = {
            "kcal": float(row.get("kcal", np.nan)),
            "protein": float(row.get("protein", np.nan)),
            "fat": float(row.get("fat", np.nan)),
            "carbs": float(row.get("carbs", np.nan)),
        }
    return mapping


def main():
    args = parse_args()
    model = AutoModelForImageClassification.from_pretrained(args.run_dir)
    processor = AutoImageProcessor.from_pretrained(args.run_dir)
    img = Image.open(args.image).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = logits.softmax(dim=-1).cpu().numpy()[0]
    id2label = model.config.id2label
    topk_idx = np.argsort(-probs)[: args.top_k]
    preds = []
    for i in topk_idx:
        label = id2label.get(str(int(i)), id2label.get(int(i), str(int(i))))
        preds.append({"id": int(i), "label": label, "prob": float(probs[i])})
    nutrition_map = load_nutrition(os.path.join(os.path.dirname(args.run_dir), args.nutrition_csv))
    top1 = preds[0]
    nut = nutrition_map.get(top1["label"])
    result = {"predictions": preds, "nutrition": nut}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import torch
    main()
