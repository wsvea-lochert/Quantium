import keras_tuner as kt
from colorama import Fore
from typing import Optional
from tensorflow import keras
from Dobby.DobbyDelivery import DobbyDelivery
from keras.callbacks import TensorBoard, EarlyStopping


class AlbusSearch:
    """
    AlbusSearch is a class that is used to search for the best model.
    """
    def __init__(self, model, json: str, kp_def: str, img_dir: str, log_dir: str, epochs: Optional[int] = 50):
        """
        Initializes the AlbusSearch class.
        :param model: Model builder collected from Albus.AlbusModels...
        :param json: Path to dataset json file.
        :param kp_def: Path to keypoint definition file.
        :param img_dir: Path to image directory.
        :param log_dir: Path to log directory.
        :param epochs: Number of epochs to train for.
        """
        self.model = model
        self.json = json
        self.kp_def = kp_def
        self.img_dir = img_dir
        self.epochs = epochs
        self.log_dir = log_dir
        self.data = DobbyDelivery(self.json, self.kp_def, self.img_dir)

    def search(self):
        self.__run_tuner(self.model)

    def __run_tuner(self, model):
        stop_early = EarlyStopping(monitor='val_loss', patience=3)

        tuner = kt.Hyperband(model,
                             objective='val_loss',
                             max_epochs=15,
                             factor=3,
                             directory='tmp/tb',
                             project_name='Quantium',
                             overwrite=True
                             )
        tuner.search(self.data.train_set, validation_data=self.data.val_set, epochs=self.epochs,
                     callbacks=[stop_early, TensorBoard(self.log_dir)])

        # Get the optimal hyperparameters and print to console.
        tuner.results_summary()
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(Fore.GREEN, best_hps)
