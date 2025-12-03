# STATS 507 Final Project  
## Food Image Classification with Vision Transformers (Food-101)

**Author:** Yuze Jin  54587468
**Course:** STATS 507 — Data Science & Analytics Using Python

---

## Overview
This project implements an end-to-end food image classification system using the **Food-101** dataset and a **Vision Transformer (ViT-Base/16)** fine-tuned with the Hugging Face ecosystem.  
The goal is to classify food images and map predicted labels to approximate nutritional profiles derived from **USDA FoodData Central**, demonstrating how computer vision models can support lightweight nutrition-aware applications.

The final fine-tuned model achieves:

- **Top-1 Accuracy:** 87.05%  
- **Top-5 Accuracy:** 97.50%  
- **Dataset:** Food-101 test split (25,250 images)

Misclassification analysis and confusion-pair visualization are included to better understand model behavior.

This repository contains all code, configuration files, and evaluation artifacts required to reproduce the results.

---

## Repository Structure
src/ # Python source code (training, prediction, figures)

STATS_507_Final_Project.pdf/ # Summary report

result/ # Generated outputs (predictions + metrics + figures)

final.ipynb # Colab notebook used for experimentation

requirements.txt # Python dependencies

README.md # This file

---

## How to Run

### Install dependencies
```bash
pip install torch torchvision transformers datasets accelerate pandas matplotlib
```


Train model
```bash
python src/train.py --model_name google/vit-base-patch16-224-in21k \
  --output_dir runs/vit_local --epochs 1 --lr 5e-5 --batch_size 16 \
  --weight_decay 0.05 --val_split 0.10 --seed 507 --fp16
```

Generate predictions
```bash
python src/save_predictions.py --run_dir runs/vit_local --split test
Create confusion figure
```

Create confusion figure
```bash
python src/make_figures.py --run_dir runs/vit_local --split test
```
