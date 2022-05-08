import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import imgaug.augmenters as iaa
from Filch.FilchUtils import get_pose
from Firebolt.FireboltUtils import get_json_dict, get_kpsoi


class FireboltImageFlipper:
    """
    Class to flip images, poses and keep the original images, as well as expanding the output json file.
    """

    def __init__(self, img_dir: str, input_json: str, output_dir: str, output_json_path: str):
        """
        Initializes the class.
        :param img_dir: str Path to the directory containing the images.
        :param input_json: Path to the input json file.
        :param output_dir: Path to the image output directory.
        :param output_json_path: Path to the output json file.
        """

        self.img_dir = img_dir
        self.input_json = input_json
        self.output_dir = output_dir
        self.output_json_path = output_json_path
        self.json_dict = get_json_dict(self.input_json)
        self.samples = list(self.json_dict.keys())
        self.flip_json = {}

    def flip(self):
        """
        Flips the images, poses and keeps the original images.
        """

        self.__process()

        with open(self.output_json_path, 'w') as outfile:
            json.dump(self.flip_json, outfile, indent=2)

    def __process(self):
        """
        Processes the images and poses.
        """
        counter = 0

        for sample in tqdm(self.samples):
            data = get_pose(sample, self.json_dict, self.img_dir)
            image = data["img_data"]
            keypoint = data["joints"]
            kps = get_kpsoi(keypoint, image.shape)
            rot = np.rot90(image, k=1, axes=(1, 0))  # sometimes needed sometimes not.
            original = Image.fromarray(rot)

            original.save(self.output_dir + sample)

            self.flip_json.update({'image' + str(counter): {'image': str(sample),
                                                             'head': {'x': kps.keypoints[0].x, 'y': kps.keypoints[0].y},
                                                             'left_ankle': {'x': kps.keypoints[1].x, 'y': kps.keypoints[1].y},
                                                             'left_elbow': {'x': kps.keypoints[2].x, 'y': kps.keypoints[2].y},
                                                             'left_hip': {'x': kps.keypoints[3].x, 'y': kps.keypoints[3].y},
                                                             'left_knee': {'x': kps.keypoints[4].x, 'y': kps.keypoints[4].y},
                                                             'left_shoulder': {'x': kps.keypoints[5].x, 'y': kps.keypoints[5].y},
                                                             'left_wrist': {'x': kps.keypoints[6].x, 'y': kps.keypoints[6].y},
                                                             'neck': {'x': kps.keypoints[7].x, 'y': kps.keypoints[7].y},
                                                             'right_ankle': {'x': kps.keypoints[8].x, 'y': kps.keypoints[8].y},
                                                             'right_elbow': {'x': kps.keypoints[9].x, 'y': kps.keypoints[9].y},
                                                             'right_hip': {'x': kps.keypoints[10].x, 'y': kps.keypoints[10].y},
                                                             'right_knee': {'x': kps.keypoints[11].x, 'y': kps.keypoints[11].y},
                                                             'right_shoulder': {'x': kps.keypoints[12].x, 'y': kps.keypoints[12].y},
                                                             'right_writs': {'x': kps.keypoints[13].x, 'y': kps.keypoints[13].y},
                                                             'torso': {'x': kps.keypoints[14].x, 'y': kps.keypoints[14].y}
                                                             }
                                    })
            counter += 1

            self.__flipper(rot, kps, sample, counter)
            counter += 1

    def __flipper(self, image, kps, name: str, counter: int):
        """
        Flips the image and the keypoints.
        :param image: The image that is to be flipped.
        :param kps: Keypoints for the current image.
        :param name: Name of the current image being augmented.
        :return: New keypoints for the flipped image.
        """
        seq = iaa.Sequential([
            # flip image
            iaa.Fliplr(1.0),  # horizontally flip 100% of the images
        ], random_order=True)

        image_aug, kps_aug = seq(image=image, keypoints=kps)

        new_image_name = str(f'flip-{name}')

        self.flip_json.update({'image' + f'{str(counter)}': {'image': new_image_name,
                                                              'head': {'x': kps.keypoints[0].x, 'y': kps.keypoints[0].y},
                                                              'left_ankle': {'x': kps.keypoints[8].x, 'y': kps.keypoints[8].y},
                                                              'left_elbow': {'x': kps.keypoints[9].x, 'y': kps.keypoints[9].y},
                                                              'left_hip': {'x': kps.keypoints[10].x, 'y': kps.keypoints[10].y},
                                                              'left_knee': {'x': kps.keypoints[11].x, 'y': kps.keypoints[11].y},
                                                              'left_shoulder': {'x': kps.keypoints[12].x, 'y': kps.keypoints[12].y},
                                                              'left_wrist': {'x': kps.keypoints[13].x, 'y': kps.keypoints[13].y},
                                                              'neck': {'x': kps.keypoints[7].x, 'y': kps.keypoints[7].y},
                                                              'right_ankle': {'x': kps.keypoints[1].x, 'y': kps.keypoints[1].y},
                                                              'right_elbow': {'x': kps.keypoints[2].x,  'y': kps.keypoints[2].y},
                                                              'right_hip': {'x': kps.keypoints[3].x, 'y': kps.keypoints[3].y},
                                                              'right_knee': {'x': kps.keypoints[4].x, 'y': kps.keypoints[4].y},
                                                              'right_shoulder': {'x': kps.keypoints[5].x, 'y': kps.keypoints[5].y},
                                                              'right_writs': {'x': kps.keypoints[6].x, 'y': kps.keypoints[6].y},
                                                              'torso': {'x': kps.keypoints[14].x, 'y': kps.keypoints[14].y}
                                                              }
                                })

        save_img = Image.fromarray(image_aug)
        save_img.save(f'{self.output_dir}{new_image_name}')
