import os
import cv2
import json
import uuid
import numpy as np
from tqdm import tqdm
from random import randint
from typing import Optional
from Firebolt.FireboltUtils import get_json_dict, get_kpsoi
from Filch.FilchUtils import get_pose


class FireboltBackground:
    def __init__(self, img_dir: str, input_json: str, output_dir: str, output_json_path: str,
                 bg_dir: str, blanks_dir: Optional[str] = None, num_blanks: Optional[int] = None,
                 blank_bg: Optional[str] = None):
        """
        :param img_dir: directory with images
        :param input_json: Path to the input json file
        :param output_dir: Path to the output directory
        :param output_json_path: Path to the output json file
        :param bg_dir: Path to the backgrounds directory.
        """
        self.img_dir = img_dir
        self.input_json = input_json
        self.output_dir = output_dir
        self.output_json_path = output_json_path
        self.bg_dir = bg_dir
        self.json_dict = get_json_dict(self.input_json)
        self.samples = list(self.json_dict.keys())
        self.bg_dict = {}
        self.blanks_dir = blanks_dir
        self.num_blanks = num_blanks
        self.blank_bg = blank_bg
        self.image_counter = 0

    def swap(self):
        """
        Run all processing on images.
        :return:
        """
        self.__process()
        self.__create_blanks()

        with open(self.output_json_path, 'w') as outfile:
            json.dump(self.bg_dict, outfile, indent=2)

    def __process(self):
        """
        Process the images to create background swapped images.
        :return:
        """

        for sample in tqdm(self.samples):
            data = get_pose(sample, self.json_dict, self.img_dir)
            image = data["img_data"]
            keypoint = data["joints"]
            kps = get_kpsoi(keypoint, image.shape)

            for bg in os.listdir(self.bg_dir):
                new_image_name = str(os.path.splitext(bg)[0]) + '-' + str(sample)
                self.__swap_bg(sample, bg, new_image_name, False)

                self.bg_dict.update({'image' + str(self.image_counter): {'image': new_image_name,
                                                                         'head': {'x': kps.keypoints[0].x,
                                                                                  'y': kps.keypoints[0].y},
                                                                         'left_ankle': {'x': kps.keypoints[1].x,
                                                                                        'y': kps.keypoints[1].y},
                                                                         'left_elbow': {'x': kps.keypoints[2].x,
                                                                                        'y': kps.keypoints[2].y},
                                                                         'left_hip': {'x': kps.keypoints[3].x,
                                                                                      'y': kps.keypoints[3].y},
                                                                         'left_knee': {'x': kps.keypoints[4].x,
                                                                                       'y': kps.keypoints[4].y},
                                                                         'left_shoulder': {'x': kps.keypoints[5].x,
                                                                                           'y': kps.keypoints[5].y},
                                                                         'left_wrist': {'x': kps.keypoints[6].x,
                                                                                        'y': kps.keypoints[6].y},
                                                                         'neck': {'x': kps.keypoints[7].x,
                                                                                  'y': kps.keypoints[7].y},
                                                                         'right_ankle': {'x': kps.keypoints[8].x,
                                                                                         'y': kps.keypoints[8].y},
                                                                         'right_elbow': {'x': kps.keypoints[9].x,
                                                                                         'y': kps.keypoints[9].y},
                                                                         'right_hip': {'x': kps.keypoints[10].x,
                                                                                       'y': kps.keypoints[10].y},
                                                                         'right_knee': {'x': kps.keypoints[11].x,
                                                                                        'y': kps.keypoints[11].y},
                                                                         'right_shoulder': {'x': kps.keypoints[12].x,
                                                                                            'y': kps.keypoints[12].y},
                                                                         'right_writs': {'x': kps.keypoints[13].x,
                                                                                         'y': kps.keypoints[13].y},
                                                                         'torso': {'x': kps.keypoints[14].x,
                                                                                   'y': kps.keypoints[14].y}}
                                     })
                self.image_counter += 1

    def __swap_bg(self, img, background, name: str, blank: bool):
        """
        :param img: image name
        :param background: background image.
        :param blank: check if we are making blank images or not.
        :return:
        """
        if blank:
            image = cv2.imread(self.blanks_dir + img)
            background_image = cv2.imread(self.blank_bg + background)
        else:
            image = cv2.imread(self.img_dir + img)
            background_image = cv2.imread(self.bg_dir + background)

        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_copy = np.copy(image_color)
        image_copy = cv2.resize(image_copy, (224, 224))

        lower_green = np.array([0, 100, 0])  # [R value, G value, B value]
        upper_green = np.array([120, 255, 130])

        mask = cv2.inRange(image_copy, lower_green, upper_green)
        masked_image = np.copy(image_copy)
        masked_image[mask != 0] = [0, 0, 0]

        background_image_color = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

        # If the background image need to be resized uncomment this line.
        bg_size = randint(224, 600)
        background_image_resize = cv2.resize(background_image_color, (bg_size, bg_size))
        # Depending on the width of the background image change randint parameters
        background_image = np.roll(background_image_resize, 3 * randint(-bg_size, bg_size))

        crop_background = background_image[0:224, 0:224]

        crop_background[mask == 0] = [0, 0, 0]

        final_image = crop_background + masked_image
        final_image = cv2.resize(final_image, (224, 224))
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

        # write image to folder
        cv2.imwrite(self.output_dir + name, final_image)

    def __create_blanks(self):
        """
        This function creates blanks images.
        :returns: nothing
        """
        print(f'Creating {self.num_blanks} blank images, to balance dataset.')
        counter = 0
        backgrounds = os.listdir(self.bg_dir)
        blanks = os.listdir(self.blanks_dir)

        for i in range(self.num_blanks):
            bg = backgrounds[randint(0, len(backgrounds) - 1)]
            img = blanks[randint(0, len(blanks) - 1)]
            name = f'{uuid.uuid4()}-{img}'
            self.__swap_bg(img, bg, name, True)

            self.bg_dict.update(
                {'image' + f'{str(self.image_counter)}': {
                    'image': name,
                    'head': {'x': 0, 'y': 0},
                    'left_ankle': {'x': 0, 'y': 0},
                    'left_elbow': {'x': 0, 'y': 0},
                    'left_hip': {'x': 0, 'y': 0},
                    'left_knee': {'x': 0, 'y': 0},
                    'left_shoulder': {'x': 0, 'y': 0},
                    'left_wrist': {'x': 0, 'y': 0},
                    'neck': {'x': 0, 'y': 0},
                    'right_ankle': {'x': 0, 'y': 0},
                    'right_elbow': {'x': 0, 'y': 0},
                    'right_hip': {'x': 0, 'y': 0},
                    'right_knee': {'x': 0, 'y': 0},
                    'right_shoulder': {'x': 0, 'y': 0},
                    'right_writs': {'x': 0, 'y': 0},
                    'torso': {'x': 0, 'y': 0}
                    }
                 })
            counter += 1
            self.image_counter += 1
