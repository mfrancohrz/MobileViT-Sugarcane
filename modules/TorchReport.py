"""
    Script: TorchReport.py
    Description: Generates and saves a detailed classification report (Precision, Recall, F1-Score)
                 for PyTorch models in text format.
    
    Authors: Miguel Franco-Hernández, Vladimir Mejía-Domínguez, Yakdiel Rodriguez-Gallo
    Version: 1.0.0
"""

import os
import torch
from sklearn.metrics import classification_report
import pandas as pd

class TorchReport:
    def __init__(self, nameModel, test_loader, model, class_names):
        self.nameModel = nameModel
        self.test_loader = test_loader
        self.model = model
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)

    def _get_predictions(self):
        """Internal helper to run inference on the test set."""
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

    def generateReport(self, start_time, end_time):
        """Generates and saves the model report as a .txt file."""
        print(f"📝 Generating text report for {self.nameModel}...")
        
        y_true, y_pred = self._get_predictions()

        # Generate report as dictionary
        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        # Convert to DataFrame and format to 4 decimal places
        df_report = pd.DataFrame(report_dict).transpose()
        report_content = df_report.to_string(float_format="{:.4f}".format)

        test_size = len(self.test_loader.dataset)
        batch_size = self.test_loader.batch_size
        execution_time = end_time - start_time

        report_txt = (
            f"📄 Model Report: {self.nameModel}\n\n"
            f"🔹 Test Images: {test_size}\n"
            f"🔹 Batch Size: {batch_size}\n"
            f"🔹 Execution Time: {execution_time:.2f} seconds\n\n"
            f"📊 Classification Report:\n{report_content}\n"
        )

        report_path = os.path.join(self.report_dir, f"{self.nameModel}_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write(report_txt)

        print(f"✅ Report saved to {report_path}\n")