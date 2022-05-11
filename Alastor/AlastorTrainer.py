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
        self.splits.sort()
        for split in self.splits:
            split_name = str(os.path.splitext(split[0]))
            print(Fore.GREEN, "Running AlastorTrainer on split:", split)
            if '1' in split_name:
                mobilenet_model = mobilenet()
                resnet_model = resnet()
                residual_model = residual()
                cnn_model = cnn()
            else:
                mobilenet_model = get_model(f'{self.checkpoint_dir}mobilenet-{previous_split}/')
                resnet_model = get_model(f'{self.checkpoint_dir}resnet-{previous_split}/')
                residual_model = get_model(f'{self.checkpoint_dir}residual-{previous_split}/')
                cnn_model = get_model(f'{self.checkpoint_dir}cnn-{previous_split}/')

            print(Fore.GREEN, f"Running DobbyTrainer, MobileNet-{split}")
            dobby_mobilenet = DobbyTrainer(json=f'{self.json_folder}{split}', kp_def=self.kp_def, images=self.image_folder,
                                           checkpoint_dir=self.checkpoint_dir, log_dir=self.log_dir, save_dir=self.save_dir,
                                           name=f'mobilenet-{split_name}', model=mobilenet_model)
            dobby_mobilenet.train()

            print(Fore.GREEN, f"Running DobbyTrainer, ResNet-{split_name}")
            dobby_resnet = DobbyTrainer(json=f'{self.json_folder}{split}', kp_def=self.kp_def, images=self.image_folder,
                                        checkpoint_dir=self.checkpoint_dir, log_dir=self.log_dir, save_dir=self.save_dir,
                                        name=f'resnet-{split_name}', model=resnet_model)
            dobby_resnet.train()

            print(Fore.GREEN, f"Running DobbyTrainer, CNN-{split_name}")
            dobby_cnn = DobbyTrainer(json=f'{self.json_folder}{split}', kp_def=self.kp_def, images=self.image_folder,
                                     checkpoint_dir=self.checkpoint_dir, log_dir=self.log_dir, save_dir=self.save_dir,
                                     name=f'cnn-{split_name}', model=cnn_model)
            dobby_cnn.train()

            print(Fore.BLUE, f"Running DobbyTrainer, Residual-{split_name}")
            dobby_residual = DobbyTrainer(json=f'{self.json_folder}{split}', kp_def=self.kp_def, images=self.image_folder,
                                          checkpoint_dir=self.checkpoint_dir, log_dir=self.log_dir, save_dir=self.save_dir,
                                          name=f'residual-{split_name}', model=residual_model)
            dobby_residual.train()
            previous_split = split_name

        print(Fore.GREEN, "AlastorTrainer finished")
