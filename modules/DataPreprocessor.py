"""
    Script: DataPreprocessor.py
    Description: Handles the initial partitioning of the dataset into Training/Validation 
                 and Testing sets based on a specified percentage.
    
    Authors: Miguel Franco-Hernández, Vladimir Mejía-Domínguez, Yakdiel Rodriguez-Gallo
    Version: 1.0.0
"""

import os
import shutil
import random

class DataPreprocessor:
    def __init__(self, inputPath, outputPath, test_percentage):
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.test_percentage = test_percentage
        
        # Define internal paths for organization
        self.testPath = os.path.join(self.outputPath, "Test")
        self.trainValPath = os.path.join(self.outputPath, "TrainVal")
        self.originalTrainValPath = os.path.join(self.trainValPath, "Original")
        
        self._create_base_folders()

    def _create_base_folders(self):
        """Creates the necessary directory structure for output."""
        os.makedirs(self.testPath, exist_ok=True)
        os.makedirs(self.trainValPath, exist_ok=True)
        os.makedirs(self.originalTrainValPath, exist_ok=True)

    def _get_class_folders(self):
        """Retrieves the list of class folders from the input directory."""
        return [d for d in os.listdir(self.inputPath) if os.path.isdir(os.path.join(self.inputPath, d))]

    def _split_files(self, class_input_path):
        """
        Shuffles and splits the files of a specific class into Test and Train/Val sets.
        Returns two lists: [test_files], [train_val_files].
        """
        files = [f for f in os.listdir(class_input_path) if os.path.isfile(os.path.join(class_input_path, f))]
        random.shuffle(files)
        
        # Calculate the split index
        test_count = int(len(files) * (self.test_percentage / 100))
        
        return files[:test_count], files[test_count:]

    def _copy_files(self, file_list, src_folder, dst_folder):
        """Copies a list of files from source to destination."""
        if not file_list:
            return
        os.makedirs(dst_folder, exist_ok=True)
        for file in file_list:
            shutil.copy2(os.path.join(src_folder, file), os.path.join(dst_folder, file))

    def getTestData(self):
        """
        Main execution method. Iterates through all classes, splits the data, 
        and moves images to their respective folders (Test vs Original Train/Val).
        """
        print(f"🔄 Splitting dataset: {self.test_percentage}% allocated for Testing.")
        
        for class_name in self._get_class_folders():
            class_input_path = os.path.join(self.inputPath, class_name)
            
            # Destination paths
            class_test_path = os.path.join(self.testPath, class_name)
            class_train_val_original_path = os.path.join(self.originalTrainValPath, class_name)
            
            # Split
            test_files, train_val_files = self._split_files(class_input_path)
            
            # Copy
            self._copy_files(test_files, class_input_path, class_test_path)
            self._copy_files(train_val_files, class_input_path, class_train_val_original_path)
            
            # Optional: Feedback per class
            # print(f"   Processed class '{class_name}': {len(test_files)} Test, {len(train_val_files)} Train/Val")