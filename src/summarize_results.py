from __future__ import annotations
import argparse
import json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resnet_metrics", type=str, required=False)
    p.add_argument("--vit_metrics", type=str, required=False)
    p.add_argument("--out", type=str, default="table_perf.tex")
    return p.parse_args()


def load_metrics(path: str):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    resnet = load_metrics(args.resnet_metrics)
    vit = load_metrics(args.vit_metrics)
    lines = []
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Model & Top-1 Acc. & Top-5 Acc.\\\\")
    lines.append("\\midrule")
    if resnet:
        a = resnet.get("accuracy", "")
        t5 = resnet.get("top5_accuracy", "")
        lines.append(f"ResNet-50 & {a:.3f} & {t5:.3f}\\\\")
    if vit:
        a = vit.get("accuracy", "")
        t5 = vit.get("top5_accuracy", "")
        lines.append(f"ViT-Base/16 & {a:.3f} & {t5:.3f}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
