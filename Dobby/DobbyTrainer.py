from typing import Optional
# from keras.optimizer_v1 import Adam
from tensorflow import keras
from Dobby.DobbyDelivery import DobbyDelivery
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class DobbyTrainer:
    def __init__(self, json: str, kp_def: str, images: str, checkpoint_dir: str, log_dir: str, save_dir: str, name: str,
                 model, epochs: Optional[int] = 50, stopping_patience: Optional[int] = 3, lr_patience: Optional[int] = 2):
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

    def train(self, ):
        tensorboard_callback = TensorBoard(log_dir=f"{self.log_dir}{self.model_name}/")
        early_stopping_callback = EarlyStopping(monitor="val_mae", patience=3)
        reduce_lr_callback = ReduceLROnPlateau(monitor="val_mae", factor=0.1, patience=2, verbose=1, mode="auto")
        checkpoint = ModelCheckpoint(filepath=f'{self.checkpoint_dir}{self.model_name}',
                                     monitor='val_mae', verbose=1, save_best_only=True, mode='min')

        self.model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(0.001), metrics=['mae'])
        self.model.summary()
        self.model.fit(self.train_data, validation_data=self.val_data, epochs=self.epochs,
                      callbacks=[tensorboard_callback, early_stopping_callback, reduce_lr_callback,
                                 checkpoint])

        self.model.save(f'{self.save_dir}{self.model_name}')
