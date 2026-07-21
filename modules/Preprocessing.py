"""
    Programa: Procesador de imagenes
    Descripcion: Se encarga de las transformaciones que sufren las imagenes
    Autor: Miguel Franco
"""
#Importando librerias a utilizar
import os
import cv2
import numpy as np
from scipy.ndimage import generic_filter

class Transformation:
    def __init__(self, outputPath):
        self.outputPath = outputPath

    def apply_transformations(self, inputImage, transformations, is_trainval=True):
        """
        Aplica las transformaciones a la imagen.
        Si la imagen proviene de TrainVal, también guarda la versión redimensionada.
        """
        transformed_images = {}
        resized_img = self.resizedImage(inputImage, 224, 224)

        # Verificar si la imagen redimensionada es válida
        if resized_img is None or resized_img.size == 0:
            print("⚠️ Imagen redimensionada no válida")
            return transformed_images
        
        # Solo agregar la imagen redimensionada si es TrainVal
        if is_trainval:
            transformed_images["resizedImage"] = resized_img  

        for transform in transformations:
            transformed_img = getattr(self, transform)(resized_img)
            if transformed_img is not None and transformed_img.size > 0:
                transformed_images[transform] = transformed_img
            else:
                print(f"⚠️ Transformación {transform} falló para la imagen")

        return transformed_images

    def resizedImage(self, img, width, height):
        return cv2.resize(img, (width, height))

    def rotatedImage(self, img, angle=15):
        height, width = img.shape[:2]
        center = (width / 2, height / 2)
        rotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(img, rotationMatrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    def equalizedImage(self, img):
        if len(img.shape) == 2:
            return cv2.equalizeHist(img)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img

    def addNoise(self, img, intensity=0.5):
        mean, sigma = 0, max(1, intensity * 50)
        gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

    def apply_variance_filter(self, img, window_size=3):
        return generic_filter(img.astype(np.float32), np.var, size=window_size)

    def save_transformed_images(self, transformed_images, class_name, filename):
        for transform_name, image in transformed_images.items():
            transform_path = os.path.join(self.outputPath, transform_name, class_name)
            os.makedirs(transform_path, exist_ok=True)
            
            save_path = os.path.join(transform_path, filename)  # ← Definir save_path
            cv2.imwrite(save_path, image)
    
            print(f"✅ Guardada transformación {transform_name}: {save_path}")
    
        # Imprimir cantidad de imágenes guardadas en cada carpeta de transformación
        for transform_name in transformed_images.keys():
            transform_path = os.path.join(self.outputPath, transform_name, class_name)
            num_images = len(os.listdir(transform_path))
            print(f"📂 Total imágenes en {transform_name}/{class_name}: {num_images}")
                   