"""
    Script: TorchDisplayer.py
    Description: Visualization utility for PyTorch models. Generates and saves 
                 Standard and Normalized Confusion Matrices.
    
    Authors: Miguel Franco-Hernández, Vladimir Mejía-Domínguez, Yakdiel Rodriguez-Gallo
    Version: 1.0.0
"""

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
        
        # Output directory changed to English
        self.plot_dir = os.path.join("plots", self.nameModel)
        os.makedirs(self.plot_dir, exist_ok=True)

    def _savePlot(self, fig, filename):
        """Saves the figure to the defined plot directory."""
        path = os.path.join(self.plot_dir, filename + ".png")
        fig.savefig(path, bbox_inches='tight', dpi=300) # Added dpi and bbox for publication quality
        print(f"✅ Saved plot: {path}")
        plt.close(fig) # Frees memory

    def _get_predictions(self):
        """Internal method to run inference and gather predictions."""
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
        
        return all_labels, all_preds

    def plotConfusionMatrix(self):
        """Generates and saves the Standard Confusion Matrix."""
        print(f"📊 Generating Confusion Matrix for {self.nameModel}...")
        y_true, y_pred = self._get_predictions()

        cm = confusion_matrix(y_true, y_pred)
        
        # Increased figure size for better readability in papers
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix - {self.nameModel}")
        
        # Rotate labels slightly if they are long
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        self._savePlot(fig, "confusion_matrix")
        # plt.show()

    def plotNormalizedConfusionMatrix(self):
        """Generates and saves the Normalized Confusion Matrix."""
        print(f"📊 Generating Normalized Confusion Matrix for {self.nameModel}...")
        y_true, y_pred = self._get_predictions()

        cm = confusion_matrix(y_true, y_pred).astype('float')
        
        # Normalize and handle division by zero with epsilon
        cm_normalized = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
        cm_normalized = np.nan_to_num(cm_normalized)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Normalized Confusion Matrix - {self.nameModel}")
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        self._savePlot(fig, "normalized_confusion_matrix")
        # plt.show()