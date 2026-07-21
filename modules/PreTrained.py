import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Pretrained:
    def __init__(self, path, *excluded_dirs):
        self.path = path
        self.excluded_dirs = list(excluded_dirs)
        print(path)
    
    def orderData(self, batch_size=32): # 32
        data = []
        for class_name in os.listdir(self.path):
            if class_name in self.excluded_dirs:
                continue
            class_path = os.path.join(self.path, class_name)
            if os.path.isdir(class_path):
                for category in os.listdir(class_path):
                    category_path = os.path.join(class_path, category)
                    if os.path.isdir(category_path):
                        files = [f for f in os.listdir(category_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
                        for file in files:
                            img_path = os.path.join(category_path, file)
                            data.append((img_path, category))
        train_df, val_df = self._convertDataFrame(data)
        trainingData = self._trainData(train_df, batch_size)
        validationData = self._validationData(val_df, batch_size)
        return trainingData, validationData, val_df
    
    def _convertDataFrame(self, data):
        df = pd.DataFrame(data, columns=["filepath", "label"])
        return train_test_split(df, test_size=0.1765, stratify=df["label"], random_state=42)
    
    def _trainData(self, train_df, batch_size=32):#32
        train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20)
        return train_datagen.flow_from_dataframe(train_df, x_col="filepath", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="categorical")
    
    def _validationData(self, val_df, batch_size=32):#32
        val_datagen = ImageDataGenerator(rescale=1./255)
        return val_datagen.flow_from_dataframe(val_df, x_col="filepath", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="categorical")
    
class PostPreprocessor:
    def __init__(self, inputPath, outputPath, test_percentage):
        self.testPath = os.path.join(outputPath, "Test")
        self.test_percentage = test_percentage
        self.test_datagen = ImageDataGenerator(rescale=1./255)  # Normalización

    def getTestData(self, batch_size=32, target_size=(224, 224)):
        """Carga las imágenes de test para evaluar los modelos."""
        test_generator = self.test_datagen.flow_from_directory(
            self.testPath,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        return test_generator