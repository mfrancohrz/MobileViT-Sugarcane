"""
    Script: TrainerTorch.py
    Description: Handles the training, evaluation, and visualization pipeline 
                 specifically for the MobileViT architecture using PyTorch.
    
    Authors: Miguel Franco-Hernández, Vladimir Mejía-Domínguez, Yakdiel Rodriguez-Gallo
    Version: 1.0.0
"""

import os
import time
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
from PIL import Image
import pandas as pd
import numpy as np

class TrainerTorch:
    def __init__(self, train_dir, test_dir, num_classes=11, batch_size=16, image_size=256):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.image_size = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.nameModel = "MobileViT-v2-200"
        
        # Setup output directories
        self.plot_dir = os.path.join("plots", self.nameModel)
        self.model_dir = "models"
        
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_data(self):
        """Loads and transforms the datasets for training and testing."""
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
        """Initializes the MobileViT architecture with pretrained weights."""
        model = timm.create_model('mobilevitv2_200', pretrained=True)
        model.reset_classifier(self.num_classes)
        return model

    def load_model(self):
        """Loads a previously saved model checkpoint."""
        model = self.build_model().to(self.device)
        model_path = os.path.join(self.model_dir, f"{self.nameModel}.pt")
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✅ Model loaded successfully from checkpoint.")
        else:
            print(f"⚠️ Checkpoint not found at {model_path}")
            
        return model

    def train(self, epochs=50, lr=0.001, continue_training=False, start_epoch=0):
        """Main training loop."""
        self.prepare_data()

        if continue_training:
            model = self.load_model()
        else:
            model = self.build_model().to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        history_loss = []
        history_acc = []

        for i in range(epochs):
            epoch = start_epoch + i
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            print(f"\n⏳ Epoch {i+1}/{epochs} (Global Epoch: {epoch+1})")
            epoch_start = time.time()

            # Training loop with progress bar
            for images, labels in tqdm(self.train_loader, desc=f"Training epoch {epoch+1}", leave=False):
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
            avg_loss = running_loss / len(self.train_loader)
            
            history_loss.append(avg_loss)
            history_acc.append(acc)

            print(f"✅ Epoch {epoch+1} completed in {time.time() - epoch_start:.2f}s — Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

        self.model = model
        self.save_model(model)
        self.plot_training_metrics(history_loss, history_acc)
        self.evaluate(model)

    def save_model(self, model):
        """Saves the model state dictionary."""
        save_path = os.path.join(self.model_dir, f"{self.nameModel}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")

    def plot_training_metrics(self, loss_list, acc_list):
        """Generates and saves the training loss and accuracy plots."""
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
        plt.savefig(save_path)
        print(f"📈 Training metrics plot saved to {save_path}")
        # plt.show() # Uncomment if running in notebook

    def evaluate(self, model):
        """Evaluates the model on the test set and prints a classification report."""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        report_dict = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            output_dict=True
        )

        df_report = pd.DataFrame(report_dict).transpose()
        print("\n📋 Test Set Classification Report:\n")
        print(df_report.to_string(float_format="{:.4f}".format))

    def evaluate_only(self):
        """Wrapper to evaluate a loaded model without training."""
        self.prepare_data()
        model = self.load_model()
        self.model = model
        self.evaluate(model)
        
    def generate_validation_results(self, df, class_names):
        """
        Evaluates a specific DataFrame (Validation Set), prints the report,
        and generates confusion matrices.
        """
        
        # Internal Dataset class for DataFrame handling
        class CustomDataset(Dataset):
            def __init__(self, dataframe, transform=None, class_to_idx=None):
                self.dataframe = dataframe
                self.transform = transform
                self.class_to_idx = class_to_idx
                
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

        # Map class names to indices
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
        
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        validation_dataset = CustomDataset(df, transform=transform, class_to_idx=class_to_idx)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

        # --- Inference Loop ---
        self.model.eval()
        all_preds, all_labels = [], []
        
        print("\n🔄 Starting Validation Set Evaluation...")
        with torch.no_grad():
            for images, labels in tqdm(validation_loader, desc="Evaluating Validation Set"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # --- Text Report ---
        report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        print("\n\n📋 VALIDATION Set Classification Report:\n")
        print(df_report.to_string(float_format="{:.4f}".format))

        # --- Confusion Matrix Plot ---
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix (Validation) - {self.nameModel}")
        
        save_path_cm = os.path.join(self.plot_dir, "validation_confusion_matrix.png")
        plt.savefig(save_path_cm)
        print(f"\n✅ Validation Confusion Matrix saved to {save_path_cm}")
        # plt.show()

        # --- Normalized Confusion Matrix Plot ---
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Normalized Confusion Matrix (Validation) - {self.nameModel}")
        
        save_path_norm = os.path.join(self.plot_dir, "validation_normalized_confusion_matrix.png")
        plt.savefig(save_path_norm)
        print(f"✅ Validation Normalized Matrix saved to {save_path_norm}")
        # plt.show()