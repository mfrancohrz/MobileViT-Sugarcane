"""
    Script: PreTrained.py
    Description: Handles data loading, splitting into Train/Validation sets, 
                 and creating data generators (DataLoaders) for Keras models.
                 Also provides the raw DataFrames required for PyTorch pipelines.
    
    Authors: Miguel Franco-Hernández, Vladimir Mejía-Domínguez, Yakdiel Rodriguez-Gallo
    Version: 1.0.0
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Pretrained:
    def __init__(self, path, *excluded_dirs):
        self.path = path
        self.excluded_dirs = list(excluded_dirs)
        # Debug: path verification removed for cleaner output
    
    def orderData(self, batch_size=32):
        """
        Scans the directory structure, creates a DataFrame of images, 
        and splits them into training and validation sets.
        
        Returns:
            trainingData: Keras ImageDataGenerator (for legacy models)
            validationData: Keras ImageDataGenerator (for legacy models)
            val_df: Pandas DataFrame containing validation paths (used for Torch evaluation)
        """
        data = []
        # Iterate through class folders (e.g., Healthy, Mosaic, Rust...)
        for class_name in os.listdir(self.path):
            if class_name in self.excluded_dirs:
                continue
            
            class_path = os.path.join(self.path, class_name)
            if os.path.isdir(class_path):
                # Iterate through transformation folders (e.g., rotated, equalized...)
                for category in os.listdir(class_path):
                    category_path = os.path.join(class_path, category)
                    if os.path.isdir(category_path):
                        # Filter only image files
                        files = [f for f in os.listdir(category_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
                        for file in files:
                            img_path = os.path.join(category_path, file)
                            data.append((img_path, category)) # Using folder name as label
        
        # Create DataFrames and split
        train_df, val_df = self._create_dataframes(data)
        
        # Create Keras Generators (Legacy support)
        trainingData = self._create_train_generator(train_df, batch_size)
        validationData = self._create_validation_generator(val_df, batch_size)
        
        return trainingData, validationData, val_df
    
    def _create_dataframes(self, data):
        """
        Converts list of paths to DataFrame and performs stratified split.
        """
        df = pd.DataFrame(data, columns=["filepath", "label"])
        
        # Note on split ratio 0.1765:
        # The dataset was initially split into 15% Test and 85% TrainVal.
        # To get a final distribution of 70% Train, 15% Val, 15% Test:
        # We need 15% of the ORIGINAL total from the current 85%.
        # Calculation: 15 / 85 ≈ 0.17647 -> 0.1765
        return train_test_split(df, test_size=0.1765, stratify=df["label"], random_state=42)
    
    def _create_train_generator(self, train_df, batch_size=32):
        """Creates Keras generator for training with augmentation."""
        train_datagen = ImageDataGenerator(
            rescale=1./255, 
            horizontal_flip=True, 
            rotation_range=20
        )
        return train_datagen.flow_from_dataframe(
            train_df, 
            x_col="filepath", 
            y_col="label", 
            target_size=(224, 224), 
            batch_size=batch_size, 
            class_mode="categorical"
        )
    
    def _create_validation_generator(self, val_df, batch_size=32):
        """Creates Keras generator for validation (rescale only)."""
        val_datagen = ImageDataGenerator(rescale=1./255)
        return val_datagen.flow_from_dataframe(
            val_df, 
            x_col="filepath", 
            y_col="label", 
            target_size=(224, 224), 
            batch_size=batch_size, 
            class_mode="categorical"
        )
    
class PostPreprocessor:
    def __init__(self, inputPath, outputPath, test_percentage):
        self.testPath = os.path.join(outputPath, "Test")
        self.test_percentage = test_percentage
        self.test_datagen = ImageDataGenerator(rescale=1./255)

    def getTestData(self, batch_size=32, target_size=(224, 224)):
        """
        Loads test images for model evaluation.
        Returns a Keras generator.
        """
        test_generator = self.test_datagen.flow_from_directory(
            self.testPath,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False  # Important for confusion matrix alignment
        )
        return test_generator