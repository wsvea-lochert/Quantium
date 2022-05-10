import os
import tensorflow as tf
from colorama import Fore, Back, Style
from Filch.FilchUtils import get_model, get_models_from_folder


class SiriusConverter:
    """
    Class to convert a model to a Tensorflow Lite model.
    """
    def __init__(self, directory: str, output_directory: str):
        self.directory = directory
        self.output_directory = output_directory

    def __convert_models(self):
        """
        Converts all models in the directory to Tensorflow Lite models.
        :return:
        """
        models = get_models_from_folder(self.directory)
        for model_name in models:
            model = get_model(f'{self.directory}{model_name}')
            converter = tf.lite.TFLiteConverter.from_keras_model(
                model)
            tflite_model = converter.convert()

            # check if file exists if not create it
            if not os.path.exists(f'{self.output_directory}{model_name}.tflite'):
                print(Fore.RED, f'{model_name} does not exist, creating it!')
                with open(f'{self.output_directory}{model_name}.tflite', 'w+') as f:
                    f.write(tflite_model)
            else:
                print(Fore.GREEN, f'{model_name} already exists, adding model information...')
                with open(f'{self.output_directory}{model_name}.tflite', 'wb') as f:
                    f.write(tflite_model)
