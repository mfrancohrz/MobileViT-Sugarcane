import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

class TorchDisplayer:
    def __init__(self, nameModel, model, test_loader, class_names):
        self.nameModel = nameModel
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plot_dir = os.path.join("graficos", self.nameModel)
        os.makedirs(self.plot_dir, exist_ok=True)

    def _savePlot(self, fig, filename):
        path = os.path.join(self.plot_dir, filename + ".png")
        fig.savefig(path)
        print(f"✅ Saved plot: {path}")

    def plotConfusionMatrix(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix - {self.nameModel}")
        self._savePlot(fig, "confusion_matrix")
        plt.show()

    def plotNormalizedConfusionMatrix(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds).astype('float')
        cm_normalized = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
        cm_normalized = np.nan_to_num(cm_normalized)

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Normalized Confusion Matrix - {self.nameModel}")
        self._savePlot(fig, "normalized_confusion_matrix")
        plt.show()