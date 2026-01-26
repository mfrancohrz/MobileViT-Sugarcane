"""
    Script: TSNEViz.py
    Description: Generates t-SNE visualizations of the model's feature embeddings 
                 to demonstrate class separability.
    
    Authors: Miguel Franco-Hernández, Vladimir Mejía-Domínguez, Yakdiel Rodriguez-Gallo
    Version: 1.0.0
"""

import torch
import timm
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import seaborn as sns

def generate_tsne(model_path, test_dir, output_dir, num_classes=11):
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Dataset & Transforms (Matching training normalization)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"❌ Test directory not found: {test_dir}")

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = test_dataset.classes

    # 3. Load Model
    print(f"🔄 Loading model from {model_path}...")
    model = timm.create_model("mobilevitv2_200", pretrained=False, num_classes=num_classes)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"❌ Model file not found at {model_path}")
        
    model = model.to(device)
    model.eval()

    # 4. Extract Embeddings
    print("⏳ Extracting features...")
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            # Extract features before the classifier
            feats = model.forward_features(images)
            
            # Global Average Pooling if output is 4D [Batch, Channels, H, W]
            if feats.ndim == 4:
                feats = feats.mean(dim=(2, 3))
            
            embeddings.append(feats.cpu().numpy())
            labels.append(targets.numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    # 5. Apply t-SNE
    print("🔄 Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 6. Visualization using Seaborn for better style
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1],
        hue=[class_names[l] for l in labels],
        palette="tab20",
        s=60,
        alpha=0.8,
        edgecolor='k'
    )

    plt.title("t-SNE Feature Embeddings", fontsize=16)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Classes", borderaxespad=0.)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "tsne_embeddings.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ t-SNE plot saved to: {save_path}")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate t-SNE visualization")
    parser.add_argument('--model', type=str, default='models/MobileViT-v2-200.pt', help='Path to the .pt model file')
    parser.add_argument('--dataset', type=str, default='./Output/Test', help='Path to the Test dataset directory')
    parser.add_argument('--output', type=str, default='plots', help='Directory to save the plot')
    
    args = parser.parse_args()
    
    try:
        generate_tsne(args.model, args.dataset, args.output)
    except Exception as e:
        print(f"❌ Error: {e}")