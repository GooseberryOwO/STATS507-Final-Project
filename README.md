# STATS 507 Final Project  
## Food Image Classification with Vision Transformers (Food-101)

**Author:** Yuze Jin  
**Course:** STATS 507 — Data Science & Analytics Using Python

---

## Overview
This project fine-tunes a Vision Transformer (ViT-Base/16) on the Food-101 dataset using Hugging Face tools.  
The model predicts food categories from images and maps predictions to approximate nutrition profiles using USDA data.

**Final Test Performance**
- Top-1 Accuracy: 87.05%
- Top-5 Accuracy: 97.50%

---

## Repository Structure
src/ # Training and evaluation scripts
runs/vit_colab/ # Predictions, metrics, confusion figure (no weights)
final_report.tex # LaTeX report
README.md # This file

---

## How to Run

### Install dependencies
```bash
pip install torch torchvision transformers datasets accelerate pandas matplotlib

Train model
python src/train.py --model_name google/vit-base-patch16-224-in21k \
  --output_dir runs/vit_local --epochs 1 --lr 5e-5 --batch_size 16 \
  --weight_decay 0.05 --val_split 0.10 --seed 507 --fp16

Generate predictions
python src/save_predictions.py --run_dir runs/vit_local --split test
Create confusion figure
bash
复制代码
python src/make_figures.py --run_dir runs/vit_local --split test
