import os
import time
import glob
import shutil
import torch
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch import nn, optim
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import csv

class TrainerTorch:
    def __init__(self, train_dir, test_dir, model_name="mobilevitv2_200", num_classes=11, batch_size=16, image_size=256):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.image_size = image_size
        
        # ⚠️ Fijado a CPU para evitar saturación de la gráfica
        self.device = torch.device('cpu')
        
        self.nameModel = model_name 
        self.plot_dir = os.path.join("graficos", self.nameModel)
        
        self.models_dir = os.path.join("models", self.nameModel)
        self.report_dir = os.path.join("reportes", self.nameModel)
        
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        
        old_checkpoints = glob.glob(os.path.join("models", f"{self.nameModel}_*.pt"))
        for old_file in old_checkpoints:
            if os.path.isfile(old_file):
                try:
                    shutil.move(old_file, os.path.join(self.models_dir, os.path.basename(old_file)))
                    print(f"📁 Backup movido a su nueva subcarpeta: {os.path.basename(old_file)}")
                except Exception:
                    pass
                    
        old_reports = glob.glob(os.path.join("reportes", f"{self.nameModel}_*.txt"))
        for old_file in old_reports:
            if os.path.isfile(old_file):
                try:
                    new_name = os.path.basename(old_file).replace(f"{self.nameModel}_", "")
                    shutil.move(old_file, os.path.join(self.report_dir, new_name))
                    print(f"📄 Reporte movido y renombrado en su nueva subcarpeta: {new_name}")
                except Exception:
                    pass
        
        self.history_file = os.path.join(self.plot_dir, f"{self.nameModel}_history.csv")

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        train_data = ImageFolder(self.train_dir, transform=transform)
        test_data = ImageFolder(self.test_dir, transform=transform)

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        self.class_names = train_data.classes

    def build_model(self):
        model = timm.create_model(self.nameModel, pretrained=True, num_classes=self.num_classes)
        return model

    def get_latest_checkpoint(self):
        checkpoints = glob.glob(os.path.join(self.models_dir, f"{self.nameModel}_epoch_*.pt"))
        if not checkpoints:
            return None, 0
        
        latest_cp = max(checkpoints, key=lambda x: int(x.split('_epoch_')[-1].split('.pt')[0]))
        epoch = int(latest_cp.split('_epoch_')[-1].split('.pt')[0])
        return latest_cp, epoch

    def load_model(self, checkpoint_path=None):
        model = self.build_model().to(self.device)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.models_dir, f"{self.nameModel}_final.pt")
            
        if os.path.exists(checkpoint_path):
            # ⚠️ CORRECCIÓN: Obligar a cargar en el dispositivo actual (CPU)
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"✅ Modelo cargado desde {checkpoint_path}")
        else:
            print(f"⚠️ No se encontró checkpoint en {checkpoint_path}")
        return model

    def resume_training(self, total_epochs=55):
        latest_cp, start_epoch = self.get_latest_checkpoint()

        if latest_cp and start_epoch < total_epochs:
            print(f"🔄 Retomando entrenamiento desde la época {start_epoch} (Archivo: {latest_cp})...")
            self.train(epochs=total_epochs, continue_training=True, start_epoch=start_epoch, checkpoint_path=latest_cp)
        elif start_epoch >= total_epochs or os.path.exists(os.path.join(self.models_dir, f"{self.nameModel}_final.pt")):
            print(f"✅ El modelo {self.nameModel} ya ha completado su entrenamiento.")
            self.model = self.load_model()
        else:
            print(f"🆕 Iniciando entrenamiento para {self.nameModel} desde cero...")
            self.train(epochs=total_epochs, continue_training=False, start_epoch=0)

    def train(self, epochs=55, continue_training=False, start_epoch=0, checkpoint_path=None):
        self.prepare_data()

        if continue_training and checkpoint_path:
            model = self.load_model(checkpoint_path)
        else:
            model = self.build_model().to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        history_loss = []
        history_acc = []

        if continue_training and os.path.exists(self.history_file):
            try:
                df_hist = pd.read_csv(self.history_file)
                history_loss = df_hist['loss'].tolist()
                history_acc = df_hist['accuracy'].tolist()
                print(f"📊 Historial de gráficas recuperado: {len(history_loss)} épocas previas.")
            except Exception as e:
                print(f"⚠️ No se pudo leer el historial previo: {e}")
        elif not continue_training:
            with open(self.history_file, mode='w', newline='') as f:
                f.write("epoch,loss,accuracy\n")

        for epoch in range(start_epoch, epochs):
            current_lr = 0.001 if epoch < 50 else 0.0001
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            print(f"\n⏳ Época {epoch+1}/{epochs} | Learning Rate: {current_lr}")
            epoch_start = time.time()

            for images, labels in tqdm(self.train_loader, desc=f"Entrenando época {epoch+1}", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            acc = correct / total
            epoch_loss = running_loss / len(self.train_loader)
            
            history_loss.append(epoch_loss)
            history_acc.append(acc)

            with open(self.history_file, mode='a', newline='') as f:
                f.write(f"{epoch+1},{epoch_loss},{acc}\n")

            print(f"✅ Época {epoch+1} completada en {time.time() - epoch_start:.2f}s — Loss: {running_loss:.4f}, Accuracy: {acc:.4f}")

            if (epoch + 1) % 5 == 0:
                backup_path = os.path.join(self.models_dir, f"{self.nameModel}_epoch_{epoch+1}.pt")
                torch.save(model.state_dict(), backup_path)
                print(f"💾 Backup de seguridad guardado: {backup_path}")

        self.model = model
        final_path = os.path.join(self.models_dir, f"{self.nameModel}_final.pt")
        torch.save(model.state_dict(), final_path)
        print(f"🏁 Entrenamiento completado. Modelo final guardado en {final_path}")
        
        self.plot_training_metrics(history_loss, history_acc)
        self.evaluate(model)

    def save_model(self, model):
        os.makedirs(self.models_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.models_dir, f"{self.nameModel}_final.pt"))
        print(f"✅ Modelo guardado en {self.models_dir}/{self.nameModel}_final.pt")

    def plot_training_metrics(self, loss_list, acc_list):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(loss_list, label='Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2.plot(acc_list, label='Accuracy', color='green')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, "training_metrics.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📈 Gráfica de entrenamiento guardada en {save_path}")
        plt.show()

    def evaluate(self, model):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluando Test"):
                images = images.to(self.device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        report_dict = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            output_dict=True
        )

        df_report = pd.DataFrame(report_dict).transpose()
        reporte_formateado = df_report.to_string(float_format="{:.4f}".format)

        print("\n📋 Reporte de clasificación (TEST):\n")
        print(reporte_formateado)
        
        report_path = os.path.join(self.report_dir, "reporte_test.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Reporte de Clasificación (TEST) - {self.nameModel}\n\n")
            f.write(reporte_formateado)
        print(f"📄 Reporte de Test guardado en: {report_path}")
        
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(7, 7)) 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix (Test) - {self.nameModel}")
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "test_confusion_matrix.png"), bbox_inches='tight')
        print(f"✅ Matriz de confusión de TEST guardada.")
        plt.show()

        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Normalized Confusion Matrix (Test) - {self.nameModel}")
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "test_normalized_confusion_matrix.png"), bbox_inches='tight')
        print(f"✅ Matriz de confusión normalizada de TEST guardada.")
        plt.show()

    def evaluate_only(self):
        self.prepare_data()
        model = self.load_model()
        self.model = model
        self.evaluate(model)
        
    def generate_validation_results(self, df, class_names):
        class CustomDataset(Dataset):
            def __init__(self, dataframe, transform=None):
                self.dataframe = dataframe
                self.transform = transform
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
            def __len__(self):
                return len(self.dataframe)
            def __getitem__(self, idx):
                img_path = self.dataframe.iloc[idx]['filepath']
                label_name = self.dataframe.iloc[idx]['label']
                label_idx = self.class_to_idx[label_name]
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label_idx

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        validation_dataset = CustomDataset(df, transform=transform)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(validation_loader, desc="Evaluando Validación"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        reporte_formateado = df_report.to_string(float_format="{:.4f}".format)
        
        print("\n\n📋 Reporte de Clasificación (VALIDACIÓN):\n")
        print(reporte_formateado)

        report_path = os.path.join(self.report_dir, "reporte_validacion.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Reporte de Clasificación (VALIDACIÓN) - {self.nameModel}\n\n")
            f.write(reporte_formateado)
        print(f"📄 Reporte de Validación guardado en: {report_path}")

        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix (Val) - {self.nameModel}")
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "validation_confusion_matrix.png"), bbox_inches='tight')
        print(f"\n✅ Matriz de confusión de validación guardada.")
        plt.show()

        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Normalized Confusion Matrix (Val) - {self.nameModel}")
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "validation_normalized_confusion_matrix.png"), bbox_inches='tight')
        print(f"✅ Matriz de confusión normalizada de validación guardada.")
        plt.show()