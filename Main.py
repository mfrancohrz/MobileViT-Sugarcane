"""
    Programa: Main.py
    Descripcion: 
        Este programa es el encargado de coordinar todo el proceso de procesamiento
        de imagenes, tanto sus transformaciones como su posterior guardado y
        almacenamiento. Posteriormente es el encargado de llevar a cabo el 
        entrenamiento de los modelos especificados, donde al final de su desarrollo
        debera proporcionar diferentes graficas que permitan la evoluacion de su
        desempeno y un reporte final
    Autor: Miguel Franco
    Version: 5.0.0
"""
from Modulos.Preprocessing import Transformation
from Modulos.DataPreprocessor import DataPreprocessor
from Modulos.PreTrained import Pretrained, PostPreprocessor
from Modulos.Trainer import Trainer
from Modulos.Displayer import Displayer
from Modulos.Report import Report
import time
import cv2
import os

class Main:
    def __init__(self, inputPath, outputPath, test_percentage, excluded_dirs=[]):
        self.processor = DataPreprocessor(inputPath, outputPath, test_percentage)
        self.transformation = Transformation(os.path.join(outputPath, "TrainVal"))
        self.pretrained = Pretrained(os.path.join(outputPath, "TrainVal"), os.path.join(outputPath, "Test"), *excluded_dirs)
        self.postprocessor = PostPreprocessor(inputPath, outputPath, test_percentage)
        self.testPath = os.path.join(outputPath, "Test")

    def getTestData(self):
        """Obtiene el conjunto de testeo."""
        return self.postprocessor.getTestData()

    def execute(self):
        """Ejecuta el preprocesamiento y transformación de imágenes."""
        self.processor.getTestData()
        transformations = ["rotatedImage", "equalizedImage", "addNoise", "apply_variance_filter"]

        for class_name in os.listdir(self.processor.originalTrainValPath):
            class_path = os.path.join(self.processor.originalTrainValPath, class_name)
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                transformed_images = self.transformation.apply_transformations(img, transformations, is_trainval=True)
                self.transformation.save_transformed_images(transformed_images, class_name, filename)

    def getDatas(self):
        """Obtiene los datos de entrenamiento y validación."""
        return self.pretrained.orderData()

    def executeTraining(self, nameModel, trainingData, validationData, testData, val_df_filtrado):
        """Entrena y guarda los modelos, luego genera las gráficas y reportes."""
        for name in nameModel:
            print(f"\n🚀 Procesando {name}...")

            # AQUI ESTA LA CORRECCION: Agregamos los nuevos modelos a la lista de PyTorch
            if name in ["mobilevitv2_100", "efficientnet_b0", "convnext_tiny", "efficientformer_l1", "edgenext_xx_small"]:
                from TrainerTorch import TrainerTorch
                
                torchTrainer = TrainerTorch(
                    train_dir=os.path.join(self.processor.originalTrainValPath),
                    test_dir=self.testPath,
                    model_name=name,
                    num_classes=11,
                    batch_size=8,
                    image_size=224
                )

                # Controla el flujo de entrenamiento (Inicia o retoma si hay backups)
                torchTrainer.resume_training(total_epochs=55)

                # Evalúa con la data de Test e imprime el reporte final
                torchTrainer.evaluate_only()

                # Genera y guarda reportes y matrices para la data de validación limpia
                torchTrainer.generate_validation_results(val_df_filtrado, torchTrainer.class_names)

                print(f"✅ Reportes y gráficas finalizados para {name}.")
                continue # Este continue es vital para que no pase a Keras

            # Este bloque inferior solo se ejecutará si pones modelos clásicos (Ej. DenseNet201)
            trainer = Trainer(trainingData, validationData)
            model = trainer.builderModel(name, load_existing=True)
            if model is None:
                model = trainer.builderModel(name)

            start_time = time.time()
            history = trainer.train(model, 200)
            end_time = time.time()
            trainer.save_model(model, name)

        print("\n✅ Todos los modelos han sido procesados satisfactoriamente.")


if __name__ == "__main__":
    inputPath = r'D:\Documentos\Sugarcane_Paper\Dataset'
    outputPath = r'D:\Documentos\Sugarcane_Paper\Output'
    test_percentage = 15
    excluded_dirs = ["Original"]
    
    nameModel = ["efficientformer_l1"]    # "edgenext_xx_small", "efficientformer_l1", "mobilevitv2_100", "efficientnet_b0", "convnext_tiny"

    processor = Main(inputPath, outputPath, test_percentage, excluded_dirs)
    
    trainingData, validationData, val_df_completo = processor.getDatas()
    testData = processor.getTestData()

    print(f"\n- Tamaño original del set de validación (con aumentos): {len(val_df_completo)}")
    val_df_filtrado = val_df_completo[val_df_completo['filepath'].str.contains(r'\\resizedImage\\', regex=True)]
    print(f"- Tamaño limpio del set de validación (sin aumentos): {len(val_df_filtrado)}")
    
    print("\n🚀 Procesamiento de datos completado")
    
    processor.executeTraining(nameModel, trainingData, validationData, testData, val_df_filtrado)
    
    print("🚀 Proceso de evaluación completado")