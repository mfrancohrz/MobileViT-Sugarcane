# Precision Agriculture: MobileViT-Based Classification of Sugarcane Leaf Diseases
[![DOI](https://zenodo.org/badge/1142228645.svg)](https://doi.org/10.5281/zenodo.18378229)

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
```
## 🚀 Usage
1. Dataset Setup
Ensure your dataset is organized as follows inside the `dataset/` folder:

```text
dataset/
├── Healthy/
├── Mosaic/
├── Rust/
... (11 classes total)
```
2. Training & Evaluation
To run the full pipeline (training, evaluation, and report generation), execute:
```Bash
python Main.py --input ./dataset --output ./Output --test_split 15
```
Note: The script will automatically handle data augmentation and split the dataset into Train/Val/Test.

3. Visualization Tools
You can generate specific visualizations using the provided scripts:

Grad-CAM (Explainability):
```Bash
python GradCamViz.py --image "./Output/Test/BrownRust/sample1.jpg" --model "models/MobileViT-v2-200.pt"
```
t-SNE Embeddings:
```Bash
python TSNEViz.py --dataset "./Output/Test"
```
UMAP Embeddings:
```Bash
python UMAPViz.py --dataset "./Output/Test"
```
## 📊 Results
The proposed model achieves state-of-the-art performance:

Accuracy: 98.57%

F1-Score: 98.54%

(See the plots/ folder for Confusion Matrices and detailed metrics).

## 🔗 Citation
If you use this code or the proposed methodology in your research, please cite our manuscript:

Note: This manuscript is currently under review at The Visual Computer. Citation details will be updated upon publication.

DOI: 10.5281/zenodo.18378230
