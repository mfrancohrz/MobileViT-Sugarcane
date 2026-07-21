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

        self.report_dir = "reportes"
        os.makedirs(self.report_dir, exist_ok=True)

    def generateReport(self, start_time, end_time):
        """Genera y guarda el reporte del modelo como archivo .txt (versión PyTorch)."""
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

        # Generar reporte como diccionario
        report_dict = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            output_dict=True
        )

        # Convertir a DataFrame y formatear a 4 decimales
        df_report = pd.DataFrame(report_dict).transpose()
        reporte = df_report.to_string(float_format="{:.4f}".format)

        test_size = len(self.test_loader.dataset)
        batch_size = self.test_loader.batch_size
        execution_time = end_time - start_time

        reporte_txt = (
            f"\U0001F4C4 Reporte del modelo: {self.nameModel}\n\n"
            f"\U0001F539 Test Images: {test_size}\n"
            f"\U0001F539 Batch Size: {batch_size}\n"
            f"\U0001F539 Execution Time: {execution_time:.2f} seconds\n\n"
            f"\U0001F4CA Classification Report:\n{reporte}\n"
        )

        report_path = os.path.join(self.report_dir, f"{self.nameModel}_report.txt")
        with open(report_path, 'w') as report_file:
            report_file.write(reporte_txt)

        print(f"✅ Reporte guardado en {report_path}\n")
