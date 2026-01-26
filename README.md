# Precision Agriculture: MobileViT-Based Classification of Sugarcane Leaf Diseases

This repository contains the official implementation of the paper **"Precision Agriculture: MobileViT-Based Classification of Sugarcane Leaf Diseases"**, submitted to *The Visual Computer*.

This project introduces a deep learning approach using **MobileViT-v2-200** to classify 11 categories of sugarcane health conditions with high precision (98.57%), optimized for mobile and resource-constrained devices.

## 📂 Repository Structure

- `dataset/`: Directory structure for images (see requirements below).
- `modules/`: Helper scripts for preprocessing, training, and reporting.
- `plots/`: Generated visualization results (Confusion Matrices, t-SNE, UMAP).
- `models/`: Folder to save trained `.pt` checkpoints.
- `Main.py`: The central pipeline script.
- `GradCamViz.py`, `TSNEViz.py`, `UMAPViz.py`: Visualization tools.

## ⚙️ Requirements

To run this code, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt