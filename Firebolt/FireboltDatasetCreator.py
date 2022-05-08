import json
from tqdm import tqdm
from PIL import Image
import imgaug.augmenters as iaa
from Filch.FilchUtils import get_pose
from Firebolt.FireboltUtils import get_json_dict, get_kpsoi


class FireboltDatasetCreator:
    """
    FireboltDataset class for creating final json file for dataset.
    """

    def __init__(self, img_dir: str, input_json: str, output_dir: str, output_json_path: str):
        """

        :param img_dir: Path to image directory.
        :param input_json: Path to input json file.
        :param output_dir: Path to output directory.
        :param output_json_path: Path to output json file.
        """
        self.img_dir = img_dir
        self.input_json = input_json
        self.output_dir = output_dir
        self.output_json_path = output_json_path
        self.json_dict = get_json_dict(self.input_json)
        self.samples = list(self.json_dict.keys())
        self.train_json = {}

    def create(self, augment: bool = False):
        """
        Creates the dataset.
        :param augment: Boolean to augment or not.
        :return:
        """
        self.__process(augment)

        with open(self.output_json_path, 'w') as outfile:
            json.dump(self.train_json, outfile, indent=2)

    def __process(self, aug: bool):
        """
        Processes the dataset.
        :param aug: Boolean to augment or not.
        """
        for sample in tqdm(self.samples):
            data = get_pose(sample, self.json_dict, self.img_dir)
            image = data["img_data"]
            keypoint = data["joints"]
            kps = get_kpsoi(keypoint, image.shape)

            if aug:
                seq = iaa.Sequential([
                    iaa.Sometimes(
                        0.5,
                        iaa.Multiply((0.90, 1.10)),  # change brightness, doesn't affect keypoints
                        iaa.Affine(rotate=(-20, 20), translate_px={"x": [-30, 30], "y": [-30, 30]}, shear=(-10, 10),
                                   scale=(0.80, 1.3))
                    ),
                    iaa.Sometimes(
                        0.1,
                        iaa.imgcorruptlike.MotionBlur(severity=1)
                        #    iaa.GaussianBlur(sigma=(0, 0.6))
                    ),
                    iaa.Sometimes(
                        0.1,
                        iaa.imgcorruptlike.Snow(severity=1)
                    ),
                    iaa.Sometimes(
                        0.1,
                        iaa.Rain(drop_size=(0.10, 0.20))
                    ),

                ], random_order=True)
                image_aug, kps_aug = seq(image=image, keypoints=kps)
                save_img = Image.fromarray(image_aug)
                save_img.save(f'{self.output_dir}{sample}')

                for i in range(15):  # Check if the keypoints are out of bounds.
                    if kps.keypoints[i].x == 0 and kps.keypoints[i].y == 0:
                        kps_aug.keypoints[i].x = 0
                        kps_aug.keypoints[i].y = 0

                    if kps_aug.keypoints[i].x < 0 or kps_aug.keypoints[i].y < 0:
                        kps_aug.keypoints[i].x = 0
                        kps_aug.keypoints[i].y = 0

                    if kps_aug.keypoints[i].x > 224 or kps_aug.keypoints[i].y > 224:
                        kps_aug.keypoints[i].x = 0
                        kps_aug.keypoints[i].y = 0

                self.__update_json(kps_aug, sample)

            else:
                save_img = Image.fromarray(image)
                save_img.save(f'{self.output_dir}{sample}')

                for i in range(15):  # Check if the keypoints are out of bounds.
                    if kps.keypoints[i].x == 0 and kps.keypoints[i].y == 0:
                        kps.keypoints[i].x = 0
                        kps.keypoints[i].y = 0

                    if kps.keypoints[i].x < 0 or kps.keypoints[i].y < 0:
                        kps.keypoints[i].x = 0
                        kps.keypoints[i].y = 0

                    if kps.keypoints[i].x > 224 or kps.keypoints[i].y > 224:
                        kps.keypoints[i].x = 0
                        kps.keypoints[i].y = 0

            self.__update_json(kps, sample)

    def __update_json(self, kps, sample):
        """
        Updates the json dictionary.
        :param kps: The keypoints for the current image
        :param sample: The name of the current image.
        :return:
        """
        self.train_json.update({sample: {'image_path': sample,
                                         'joints': [[kps.keypoints[0].x, kps.keypoints[0].y],
                                                    [kps.keypoints[1].x, kps.keypoints[1].y],
                                                    [kps.keypoints[2].x, kps.keypoints[2].y],
                                                    [kps.keypoints[3].x, kps.keypoints[3].y],
                                                    [kps.keypoints[4].x, kps.keypoints[4].y],
                                                    [kps.keypoints[5].x, kps.keypoints[5].y],
                                                    [kps.keypoints[6].x, kps.keypoints[6].y],
                                                    [kps.keypoints[7].x, kps.keypoints[7].y],
                                                    [kps.keypoints[8].x, kps.keypoints[8].y],
                                                    [kps.keypoints[9].x, kps.keypoints[9].y],
                                                    [kps.keypoints[10].x, kps.keypoints[10].y],
                                                    [kps.keypoints[11].x, kps.keypoints[11].y],
                                                    [kps.keypoints[12].x, kps.keypoints[12].y],
                                                    [kps.keypoints[13].x, kps.keypoints[13].y],
                                                    [kps.keypoints[14].x, kps.keypoints[14].y]
                                                    ]}

                                }
                               )
