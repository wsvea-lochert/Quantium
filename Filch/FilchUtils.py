import os
import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt


def get_model(path: str):
    """
    :param path: Path to model folder/file.
    :return:
    """
    model = keras.models.load_model(path)
    return model


def get_models_from_folder(path: str):
    """
    :param path: Path to models folder.
    :return: a list of model names from folder
    """
    models = []
    for file in os.listdir(path):
        models.append(file)

    if len(models) == 0:
        raise Exception("No models found in folder.")
    else:
        print(f'Found {len(models)} models in {path}')
        return models


def get_pose(name: str, json_dict: dict, img_dir: str):
    """
    Function for getting a image and its keypoints. Function for getting a image and its keypoints, Collected from https://keras.io/examples/vision/keypoint_detection/.
    :param name: image name
    :param json_dict: json dictionary
    :param img_dir: image directory
    :return: Image data.
    """
    data = json_dict[name]
    img_data = plt.imread(os.path.join(img_dir, data["image_path"]))

    if img_data.shape[-1] == 4:  #  If the image is RGBA convert it to RGB.
        img_data = img_data.astype(np.uint8)                                         #  Reading in the image data as an array.
        img_data = Image.fromarray(img_data)                                         #  Converting array to image.
        img_data = np.array(img_data.convert("RGB"))                                 #  Converting form RGBA to RGB.
    data["img_data"] = img_data                                                      #  Inserting the new image.

    return data


def get_train_params(json_file, kp_def_location):
    """"
    Get the training parameters from the json file.
    :param json_file: The json file containing the training parameters.
    :param kp_def_location: The location of the keypoint definition file.
    :return: The training parameters.
    """
    with open(json_file) as infile:
        json_dict = json.load(infile)

    for i in json_dict:
        for j in range(15):
            x = float(json_dict[i]['joints'][j][0])
            y = float(json_dict[i]['joints'][j][1])
            json_dict[i]['joints'][j] = [x, y]

    keypoint_def = pd.read_csv(kp_def_location)
    keypoint_def.head()

    colors = keypoint_def["Hex"].values
    colors = ['#' + color for color in colors]
    labels = keypoint_def["Name"].values.tolist()

    samples = list(json_dict.keys())
    return samples, json_dict, keypoint_def, colors, labels


def get_json_to_split(json_file: str):
    """"
        Get the training parameters from the json file.
        :param json_file: The json file containing the training parameters.
        :return: The training parameters.
        """
    with open(json_file) as infile:
        json_dict = json.load(infile)

    for i in json_dict:
        for j in range(15):
            x = float(json_dict[i]['joints'][j][0])
            y = float(json_dict[i]['joints'][j][1])
            json_dict[i]['joints'][j] = [x, y]

    samples = list(json_dict.keys())
    return samples, json_dict


def __rename_files(path: str, name: str):
    """
    :param path:
    :return:
    """
    file_list = os.listdir(path)
    print(file_list)
    for file_name in file_list:
        os.rename(path+file_name, path+f'{name}.jpg')  # TODO: check if this works correctly.


def load_image(image_path):
    """
    load image and make it ready for prediction.
    :param image_path: path to image
    :return: np.array of image
    """
    img_in = cv2.imread(image_path)
    image_color = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    image_resize = cv2.resize(image_color, (224, 224))
    img = np.array(image_resize).reshape(-1, 224, 224, 3)
    return img


def visualize_keypoints(keypoints, image_file, rot=False):
    """
    Function for visualizing keypoints prediction
    :param keypoints: predicted keypoints
    :param image_file: path to image
    :param rot: if image should be rotated
    :return: nothing
    """
    colours = ['F633FF', '00FF1B', '00FF1B', '00FF1B', '00FF1B', '00FF1B', '00FF1B', 'F633FF',
               'FF0000', 'FF0000', 'FF0000', 'FF0000', 'FF0000', 'FF0000', 'F633FF']
    colours = ['#' + color for color in colours]
    fig, ax = plt.subplots()

    for current_keypoint in keypoints:
        # img = Image.fromarray(images)
        image = plt.imread(image_file)
        """rotate image 45 degrees to the right"""
        # remove the next 3 lines if you don't want to rotate the image
        if rot:
            image = np.rot90(image, k=1, axes=(0, 1))
            image = np.flipud(image)
            image = np.fliplr(image)

        ax.imshow(image)

        current_keypoint = np.array(current_keypoint)
        # Since the last entry is the visibility flag, we discard it.
        current_keypoint = current_keypoint[:, :2]
        for idx, (x, y) in enumerate(current_keypoint):
            # ax.scatter([x*12.214], [y*16.285], c=colours[idx], marker="x", s=50, linewidths=5)
            ax.scatter([x], [y], c=colours[idx], marker="x", s=50, linewidths=5)
            # ax.scatter([x*2.23], [y*2.23], c=colours[idx], marker="x", s=50, linewidths=5)

    plt.tight_layout(pad=2.0)
    plt.show()
