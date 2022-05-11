import numpy as np
from Filch.FilchUtils import get_pose
from Dobby.DobbyDataset import DobbyDataset
from Filch.FilchUtils import get_train_params
from Dobby.DobbyAugmenter import get_augmentation_parameters, get_validation_augmentation_parameters
from colorama import Fore


class DobbyDelivery:
    def __init__(self, json: str, kp_definitions: str, img_dir: str):
        self.json = json
        self.kp_def = kp_definitions
        self.img_dir = img_dir
        self.images = []
        self.keypoints = []
        self.samples, self.json_dict, self.kp_def, self.colors, self.labels = get_train_params(self.json, self.kp_def)
        self.train_aug = get_augmentation_parameters()
        self.validation_aug = get_validation_augmentation_parameters()
        self.__get_images_and_keypoints()
        self.train_set, self.val_set = self.__deliver_datasets()

    def __get_images_and_keypoints(self):
        for sample in self.samples:
            data = get_pose(sample, self.json_dict, self.img_dir)
            image = data['img_data']
            keypoint = data['joints']

            self.images.append(image)
            self.keypoints.append(keypoint)

    def __deliver_datasets(self):
        np.random.shuffle(self.samples)
        train_keys, val_keys = (
            self.samples[int(len(self.samples) * 0.20):],
            self.samples[: int(len(self.samples) * 0.20)],
        )
        return DobbyDataset(train_keys, self.train_aug, self.json_dict, self.img_dir, 32, True), DobbyDataset(val_keys, self.validation_aug, self.json_dict, self.img_dir, 32, False)

    def __print_data_info(self):
        print(Fore.GREEN, f'Dataset delivered by DobbyDelivery service, current img_dir: {self.img_dir}, current json: {self.json}')
        print(Fore.CYAN, f"Total batches in training set: {len(self.train_set)}")
        print(Fore.CYAN, f"Total batches in validation set: {len(self.val_set)}")
