import os
import shutil
import random

class DataPreprocessor:
    def __init__(self, inputPath, outputPath, test_percentage):
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.test_percentage = test_percentage
        self.testPath = os.path.join(self.outputPath, "Test")
        self.trainValPath = os.path.join(self.outputPath, "TrainVal")
        self.originalTrainValPath = os.path.join(self.trainValPath, "Original")
        self._create_base_folders()

    def _create_base_folders(self):
        os.makedirs(self.testPath, exist_ok=True)
        os.makedirs(self.trainValPath, exist_ok=True)
        os.makedirs(self.originalTrainValPath, exist_ok=True)

    def _get_class_folders(self):
        return [d for d in os.listdir(self.inputPath) if os.path.isdir(os.path.join(self.inputPath, d))]

    def _split_files(self, class_input_path):
        files = [f for f in os.listdir(class_input_path) if os.path.isfile(os.path.join(class_input_path, f))]
        random.shuffle(files)
        test_count = int(len(files) * (self.test_percentage / 100))
        return files[:test_count], files[test_count:]

    def _copy_files(self, file_list, src_folder, dst_folder):
        if not file_list:
            return
        os.makedirs(dst_folder, exist_ok=True)
        for file in file_list:
            shutil.copy2(os.path.join(src_folder, file), os.path.join(dst_folder, file))

    def getTestData(self):
        for class_name in self._get_class_folders():
            class_input_path = os.path.join(self.inputPath, class_name)
            class_test_path = os.path.join(self.testPath, class_name)
            class_train_val_original_path = os.path.join(self.originalTrainValPath, class_name)
            test_files, train_val_files = self._split_files(class_input_path)
            self._copy_files(test_files, class_input_path, class_test_path)
            self._copy_files(train_val_files, class_input_path, class_train_val_original_path)