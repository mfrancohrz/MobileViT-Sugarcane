"""
    Script: Preprocessing.py
    Description: Handles image transformations, resizing, and augmentation techniques.
    
    Authors: Miguel Franco-Hernández, Vladimir Mejía-Domínguez, Yakdiel Rodriguez-Gallo
    Version: 1.0.0
"""

import os
import cv2
import numpy as np
from scipy.ndimage import generic_filter

class Transformation:
    def __init__(self, outputPath):
        self.outputPath = outputPath

    def apply_transformations(self, inputImage, transformations, is_trainval=True):
        """
        Applies a list of transformations to the input image.
        If is_trainval is True, it also includes the base resized image in the output.
        """
        transformed_images = {}
        resized_img = self.resizedImage(inputImage, 224, 224)

        # Check if resized image is valid
        if resized_img is None or resized_img.size == 0:
            print("⚠️ Invalid resized image")
            return transformed_images
        
        # Only add the base resized image if it belongs to the Train/Val set
        if is_trainval:
            transformed_images["resizedImage"] = resized_img  

        for transform in transformations:
            # Dynamically call the method by name
            transformed_img = getattr(self, transform)(resized_img)
            
            if transformed_img is not None and transformed_img.size > 0:
                transformed_images[transform] = transformed_img
            else:
                print(f"⚠️ Transformation {transform} failed for the image")

        return transformed_images

    def resizedImage(self, img, width, height):
        """Resizes the image to the specified dimensions."""
        return cv2.resize(img, (width, height))

    def rotatedImage(self, img, angle=15):
        """Rotates the image by a specific angle."""
        height, width = img.shape[:2]
        center = (width / 2, height / 2)
        rotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(img, rotationMatrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    def equalizedImage(self, img):
        """Applies histogram equalization (works for Grayscale and RGB)."""
        if len(img.shape) == 2:
            return cv2.equalizeHist(img)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img

    def addNoise(self, img, intensity=0.5):
        """Adds Gaussian noise to the image."""
        mean, sigma = 0, max(1, intensity * 50)
        gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
        # Clip values to stay within valid image range [0, 255]
        return np.clip(img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

    def apply_variance_filter(self, img, window_size=3):
        """Applies a variance filter using a sliding window."""
        return generic_filter(img.astype(np.float32), np.var, size=window_size)

    def save_transformed_images(self, transformed_images, class_name, filename):
        """Saves the transformed images to the corresponding output directories."""
        for transform_name, image in transformed_images.items():
            transform_path = os.path.join(self.outputPath, transform_name, class_name)
            os.makedirs(transform_path, exist_ok=True)
            
            save_path = os.path.join(transform_path, filename)
            cv2.imwrite(save_path, image)
    
            # Optional: Comment out to reduce console spam if processing many images
            # print(f"✅ Saved transformation {transform_name}: {save_path}")
    
        # Print count of images in the folder (useful for debugging)
        for transform_name in transformed_images.keys():
            transform_path = os.path.join(self.outputPath, transform_name, class_name)
            num_images = len(os.listdir(transform_path))
            print(f"📂 Total images in {transform_name}/{class_name}: {num_images}")