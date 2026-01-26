"""
    Script: Main.py
    Description: 
        This script coordinates the entire image processing pipeline, including
        transformations, storage, and organization. Subsequently, it manages 
        the training of specified models. Upon completion, it generates various 
        plots to evaluate performance and produces a final report.
    
    Authors: Miguel Franco-Hernández, Vladimir Mejía-Domínguez, Yakdiel Rodriguez-Gallo
    Version: 1.0.0 (Release)
"""

from modules.Preprocessing import Transformation
from modules.DataPreprocessor import DataPreprocessor
from modules.PreTrained import Pretrained, PostPreprocessor
# from modules.Trainer import Trainer
# from modules.Displayer import Displayer
# from modules.Report import Report
import time
import cv2
import os
import argparse

class Main:
    def __init__(self, inputPath, outputPath, test_percentage, excluded_dirs=[]):
        self.processor = DataPreprocessor(inputPath, outputPath, test_percentage)
        self.transformation = Transformation(os.path.join(outputPath, "TrainVal"))
        self.pretrained = Pretrained(os.path.join(outputPath, "TrainVal"), os.path.join(outputPath, "Test"), *excluded_dirs)
        self.postprocessor = PostPreprocessor(inputPath, outputPath, test_percentage)
        self.testPath = os.path.join(outputPath, "Test")

    def getTestData(self):
        """Retrieves the test set."""
        return self.postprocessor.getTestData()

    def execute(self):
        """Executes image preprocessing and data augmentation."""
        self.processor.getTestData()
        transformations = ["rotatedImage", "equalizedImage", "addNoise", "apply_variance_filter"]

        # This loop applies data augmentation only to the Train/Val set (85%)
        print("🚀 Starting data augmentation...")
        for class_name in os.listdir(self.processor.originalTrainValPath):
            class_path = os.path.join(self.processor.originalTrainValPath, class_name)
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                transformed_images = self.transformation.apply_transformations(img, transformations, is_trainval=True)
                self.transformation.save_transformed_images(transformed_images, class_name, filename)
        print("✅ Data augmentation completed.")

    def getDatas(self):
        """Retrieves training and validation data."""
        return self.pretrained.orderData()

    def executeTraining(self, nameModel, trainingData, validationData, testData, val_df_filtered):
        """Trains and saves models, then generates plots and reports."""
        for name in nameModel:
            print(f"🚀 Processing {name}...")

            if name == "MobileViT-v2-200":
                # Lazy import to avoid dependency errors if Torch is not used for other models
                from modules.TrainerTorch import TrainerTorch
                
                torchTrainer = TrainerTorch(
                    train_dir=os.path.join(self.processor.originalTrainValPath),
                    test_dir=self.testPath,
                    num_classes=11
                )

                # Loads the model and prints the TEST set report
                print(f"📊 Evaluating {name} on Test Set...")
                torchTrainer.evaluate_only()

                # Generates the text report and plots for the VALIDATION set
                print(f"📊 Generating validation results for {name}...")
                torchTrainer.generate_validation_results(val_df_filtered, torchTrainer.class_names)

                print("\n✅ Reports and validation plots generated successfully.")
                continue

            # Legacy code for Keras models (Optional/Comparison)
            
            # trainer = Trainer(trainingData, validationData)
            # model = trainer.builderModel(name, load_existing=True)
            # if model is None:
            #     model = trainer.builderModel(name)

            # start_time = time.time()
            # history = trainer.train(model, 200)
            # end_time = time.time()
            # trainer.save_model(model, name)

        print("✅ All requested models have been processed.")


if __name__ == "__main__":
    # Setup Argument Parser for command line flexibility
    parser = argparse.ArgumentParser(description="Sugarcane Leaf Disease Classification Pipeline")
    parser.add_argument('--input', type=str, default='./dataset', help='Path to the input dataset')
    parser.add_argument('--output', type=str, default='./Output', help='Path to the output directory')
    parser.add_argument('--test_split', type=int, default=15, help='Percentage of data to use for testing')
    
    args = parser.parse_args()

    # Configuration
    inputPath = args.input
    outputPath = args.output
    test_percentage = args.test_split
    excluded_dirs = ["Original"]
    nameModel = ["MobileViT-v2-200"]

    # Ensure output directory exists
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    print(f"📂 Input Path: {inputPath}")
    print(f"📂 Output Path: {outputPath}")

    processor = Main(inputPath, outputPath, test_percentage, excluded_dirs)
    
    # Uncomment the following line to run data augmentation (only needed once)
    
    # processor.execute()
    
    # 1. Get the complete DataFrame with augmented data
    trainingData, validationData, val_df_complete = processor.getDatas()
    testData = processor.getTestData()

    # 2. Filter the validation DataFrame to use only clean images (no augmentation)
    # This regex looks for 'resizedImage' inside the path, handling both Windows (\) and Linux (/) separators
    print(f"\n- Original Validation Set Size (Augmented): {len(val_df_complete)}")
    
    # Assuming 'resizedImage' is the folder for clean images. 
    # Update regex if your folder structure is different.
    val_df_filtered = val_df_complete[val_df_complete['filepath'].str.contains(r'[/\\]resizedImage[/\\]', regex=True)]
    
    print(f"- Clean Validation Set Size (No Augmentation): {len(val_df_filtered)}")
    
    print("\n🚀 Data processing completed")
    
    # 3. Execute Training / Evaluation
    processor.executeTraining(nameModel, trainingData, validationData, testData, val_df_filtered)
    
    print("🚀 Evaluation process completed")