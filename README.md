# STATS 507 Final Project  
### Nutrition-Aware Food Image Classification Using Vision Transformers

**Author:** Yuze Jin  
**UMID:** 54587468  
**Course:** STATS 507 — Data Science and Analytics Using Python  
**Instructor:** Prof. Brady Neal  
**University of Michigan**

---

## 📌 Project Overview

This project implements an end-to-end food image classification system using the **Food-101** dataset and a **Vision Transformer (ViT-Base/16)** fine-tuned with the Hugging Face ecosystem.  
The goal is to classify food images and map predicted labels to approximate nutritional profiles derived from **USDA FoodData Central**, demonstrating how computer vision models can support lightweight nutrition-aware applications.

The final fine-tuned model achieves:

- **Top-1 Accuracy:** 87.05%  
- **Top-5 Accuracy:** 97.50%  
- **Dataset:** Food-101 test split (25,250 images)

Misclassification analysis and confusion-pair visualization are included to better understand model behavior.

This repository contains all code, configuration files, and evaluation artifacts required to reproduce the results.

---

## 📁 Repository Structure

project/
│── final_report.tex # LaTeX version of the paper
│── final_report.pdf # Compiled report (if included)
│── README.md # This file
│
├── src/ # Python source code
│ ├── train.py # Fine-tunes ViT on Food-101
│ ├── save_predictions.py # Runs inference and saves predictions
│ ├── make_figures.py # Generates confusion-pair plots
│
├── runs/
│ └── vit_colab/
│ ├── predictions_test.parquet
│ ├── pred_summary_test.json
│ ├── top_confusions.pdf
│ ├── config.json
│ ├── preprocessor_config.json
│ └── (model weights removed for size limits)
│
└── figures/ (optional)


Large model weight files (e.g., `pytorch_model.bin`) are excluded from GitHub due to the 100MB file limit.

---

## ⚙️ Environment Setup

This project uses Python 3.9+ and the following dependencies:

```bash
pip install torch torchvision
pip install transformers datasets accelerate
pip install pandas matplotlib pyarrow
