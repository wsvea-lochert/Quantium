from typing import Optional
# from keras.optimizer_v1 import Adam
from tensorflow import keras
from Dobby.DobbyDelivery import DobbyDelivery
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class DobbyTrainer:
    def __init__(self, json: str, kp_def: str, images: str, checkpoint_dir: str, log_dir: str, save_dir: str, name: str,
                 model, epochs: Optional[int] = 100, stopping_patience: Optional[int] = 10, lr_patience: Optional[int] = 3,
                 lr: Optional[float] = 0.025, compiler: Optional[bool] = True):
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.epochs = epochs
        self.stopping_patience = stopping_patience
        self.lr_patience = lr_patience
        data = DobbyDelivery(json, kp_def, images)
        self.train_data = data.train_set
        self.val_data = data.val_set
        self.model = model
        self.lr = lr
        self.compiler = compiler

    def train(self, ):
        print(f'name:{self.model_name}\nstopping: {self.stopping_patience}\nlr_patience: {self.lr_patience}\nLR: {self.lr}\nCompiler active: {self.compiler}')
        tensorboard_callback = TensorBoard(log_dir=f"{self.log_dir}{self.model_name}/")
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=self.stopping_patience)
        reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=self.lr_patience, verbose=1, mode="auto")
        checkpoint = ModelCheckpoint(filepath=f'{self.checkpoint_dir}{self.model_name}',
                                     monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        if self.compiler:
            self.model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(self.lr), metrics=['mae'])
        else:
            keras.backend.set_value(self.model.optimizer.learning_rate, self.lr)
        self.model.summary()
        self.model.fit(self.train_data, validation_data=self.val_data, epochs=self.epochs,
                       callbacks=[tensorboard_callback, early_stopping_callback, reduce_lr_callback,
                                  checkpoint, CustomLRCallback(self.lr, self.model_name)])

        self.model.save(f'{self.save_dir}{self.model_name}')


class CustomLRCallback(keras.callbacks.Callback):
    """Custom callback for learning rate scheduler.
    """

    def __init__(self, model_name):
        super(CustomLRCallback, self).__init__()
        self.model_name = model_name

    def on_epoch_begin(self, epoch, logs=None):
        # open a file and check if the learning rate is the same as the one in the file
        with open(f"{self.model_name}-lr.txt", "r") as f:
            learning_rate = f.read()
            # if not, set the learning rate to the one in the file
        self.model.optimizer.learning_rate = float(learning_rate)
        print(f"Epoch {epoch}: Learning rate changed to {self.model.optimizer.learning_rate} from file.")


