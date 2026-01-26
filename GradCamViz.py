"""
    Script: GradCamViz.py
    Description: Generates Grad-CAM visualizations to explain model predictions,
                 highlighting the regions of interest for specific diseases.
    
    Authors: Miguel Franco-Hernández, Vladimir Mejía-Domínguez, Yakdiel Rodriguez-Gallo
    Version: 1.0.0
"""

import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.datasets import ImageFolder
import os
import argparse

class GradCamVisualizer:
    def __init__(self, model_path, test_dir, num_classes=11):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.test_dir = test_dir
        self.num_classes = num_classes
        
        # Output directory
        self.output_dir = os.path.join("plots", "GradCAM")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._load_class_names()
        self._load_model()

    def _load_class_names(self):
        """Loads class names from the directory structure."""
        if os.path.exists(self.test_dir):
            self.class_names = ImageFolder(self.test_dir).classes
            print(f"✅ Loaded {len(self.class_names)} classes from {self.test_dir}")
        else:
            raise FileNotFoundError(f"❌ Dataset directory not found: {self.test_dir}")

    def _load_model(self):
        """Rebuilds MobileViT and loads the trained weights."""
        print(f"🔄 Loading model from {self.model_path}...")
        self.model = timm.create_model('mobilevitv2_200', pretrained=True)
        self.model.reset_classifier(self.num_classes)
        
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"❌ Model file not found at {self.model_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Identify target layer (Last Conv2d layer)
        conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
        if not conv_layers:
            raise ValueError("❌ No convolutional layers found in the model.")
        self.target_layer = conv_layers[-1]
        
        # Initialize GradCAM
        self.cam = GradCAM(model=self.model, target_layers=[self.target_layer])

    def visualize(self, img_path):
        """Runs Grad-CAM on a single image and saves the result."""
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            return

        # 1. Preprocessing
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        image_pil = Image.open(img_path).convert('RGB')
        input_tensor = transform(image_pil).unsqueeze(0).to(self.device)
        
        # Original class name from folder structure
        original_class = os.path.basename(os.path.dirname(img_path))

        # 2. Prediction
        outputs = self.model(input_tensor)
        pred_idx = outputs.argmax(dim=1).item()
        pred_class = self.class_names[pred_idx]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][pred_idx].item()

        # 3. Generate Heatmap
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]
        
        # Resize original image for visualization
        rgb_img = np.array(image_pil.resize((256, 256))) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # 4. Plotting
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        ax[0].imshow(rgb_img)
        ax[0].set_title(f"Original Class:\n{original_class}", fontsize=14)
        ax[0].axis('off')

        ax[1].imshow(rgb_img) # Showing original again or just text could work, but visual consistency is good
        ax[1].set_title(f"Predicted Class:\n{pred_class}\n(Conf: {confidence:.2f})", fontsize=14, color='green' if original_class == pred_class else 'red')
        ax[1].axis('off')

        ax[2].imshow(cam_image)
        ax[2].set_title("Grad-CAM Heatmap", fontsize=14)
        ax[2].axis('off')

        plt.tight_layout()
        
        # Save output
        filename = os.path.basename(img_path)
        save_path = os.path.join(self.output_dir, f"GradCAM_{filename}")
        plt.savefig(save_path)
        print(f"✅ Visualization saved to {save_path}")
        # plt.show() # Uncomment to view interactively
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualization for a specific image")
    
    # Default paths adapted to your project structure
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='models/MobileViT-v2-200.pt', help='Path to the .pt model file')
    parser.add_argument('--dataset', type=str, default='./Output/Test', help='Path to the Test dataset directory (for class names)')
    
    args = parser.parse_args()

    # Initialize and Run
    try:
        visualizer = GradCamVisualizer(model_path=args.model, test_dir=args.dataset)
        visualizer.visualize(args.image)
    except Exception as e:
        print(f"❌ Error: {e}")

    # Example Usage in Terminal:
    # python GradCamViz.py --image "./Output/Test/Yellow_Leaf/Yellow Leaf005.jpg"