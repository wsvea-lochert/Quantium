import os
from colorama import Fore
from Filch.FilchUtils import get_model
from ModelBuilder.cnn import cnn
from ModelBuilder.resnet import resnet
from ModelBuilder.residual import residual
from ModelBuilder.mobilenet import mobilenet
from Dobby.DobbyTrainer import DobbyTrainer


class AlastorTrainer:
    def __init__(self, json_folder: str, image_folder: str, kp_def: str, checkpoint_dir: str, save_dir: str, log_dir: str):
        self.json_folder = json_folder
        self.image_folder = image_folder
        self.kp_def = kp_def
        self.checkpoint_dir = checkpoint_dir
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.splits = os.listdir(self.json_folder)

    def run(self):
        print(Fore.GREEN, "Running AlastorTrainer")
        for split in self.splits:
            print(Fore.GREEN, "Running AlastorTrainer on split:", split)
            if split.__contains__('1'):
                mobilenet_model = mobilenet()
                resnet_model = resnet()
                residual_model = residual()
                cnn_model = cnn()
            else:
                mobilenet_model = get_model(f'{self.checkpoint_dir}mobilenet-{split}/')
                resnet_model = get_model(f'{self.checkpoint_dir}resnet-{split}/')
                residual_model = get_model(f'{self.checkpoint_dir}residual-{split}/')
                cnn_model = get_model(f'{self.checkpoint_dir}cnn-{split}/')

            print(Fore.GREEN, f"Running DobbyTrainer, MobileNet-{split}")
            dobby_mobilenet = DobbyTrainer(json=f'{self.json_folder}{split}', kp_def=self.kp_def, images=self.image_folder,
                                           checkpoint_dir=self.checkpoint_dir, log_dir=self.log_dir, save_dir=self.save_dir,
                                           name=f'mobilenet-{split}', model=mobilenet_model)
            dobby_mobilenet.train()

            print(Fore.GREEN, f"Running DobbyTrainer, ResNet-{split}")
            dobby_resnet = DobbyTrainer(json=f'{self.json_folder}{split}', kp_def=self.kp_def, images=self.image_folder,
                                        checkpoint_dir=self.checkpoint_dir, log_dir=self.log_dir, save_dir=self.save_dir,
                                        name=f'resnet-{split}', model=resnet_model)
            dobby_resnet.train()

            print(Fore.GREEN, f"Running DobbyTrainer, Residual-{split}")
            dobby_cnn = DobbyTrainer(json=f'{self.json_folder}{split}', kp_def=self.kp_def, images=self.image_folder,
                                     checkpoint_dir=self.checkpoint_dir, log_dir=self.log_dir, save_dir=self.save_dir,
                                     name=f'cnn-{split}', model=cnn_model)
            dobby_cnn.train()

            print(Fore.GREEN, f"Running DobbyTrainer, Residual-{split}")
            dobby_residual = DobbyTrainer(json=f'{self.json_folder}{split}', kp_def=self.kp_def, images=self.image_folder,
                                          checkpoint_dir=self.checkpoint_dir, log_dir=self.log_dir, save_dir=self.save_dir,
                                          name=f'residual-{split}', model=residual_model)
            dobby_residual.train()

        print(Fore.GREEN, "AlastorTrainer finished")
