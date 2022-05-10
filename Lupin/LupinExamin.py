import numpy as np
from colorama import Fore
import cv2
from typing import Optional
import matplotlib.pyplot as plt
from Filch.FilchUtils import get_train_params, get_pose, get_model, load_image


class LupinExamin:
    def __init__(self, json: str, img_dir: str, kp_def: str, model_dir: str, visualize: Optional[bool] = False, num_samples: Optional[int] = 500):
        """
        Initialize the class.
        :param json: path to json file.
        :param img_dir: path to image directory.
        :param kp_def: path to keypoint definition file.
        :param model_dir: path to models directory.
        """
        self.json = json
        self.img_dir = img_dir
        self.kp_def = kp_def
        self.model_dir = model_dir
        self.samples, self.json_dict, self.kp_def, self.colors, self.labels = get_train_params(self.json, self.kp_def)
        self.images = []
        self.gt_kps = []
        self. list_of_predictions = []
        self.list_of_gt = []
        self.visualize = visualize
        self.selected_samples = self.__get_test_samples()
        self.num_samples = num_samples

        self.head_errors = []
        self.left_shoulder_errors = []
        self.right_shoulder_errors = []
        self.left_elbow_errors = []
        self.right_elbow_errors = []
        self.left_wrist_errors = []
        self.right_wrist_errors = []
        self.left_hip_errors = []
        self.right_hip_errors = []
        self.neck_errors = []
        self.torso_errors = []
        self.left_knee_errors = []
        self.right_knee_errors = []
        self.left_ankle_errors = []
        self.right_ankle_errors = []

        self.head_gt = []
        self.left_shoulder_gt = []
        self.right_shoulder_gt = []
        self.left_elbow_gt = []
        self.right_elbow_gt = []
        self.left_wrist_gt = []
        self.right_wrist_gt = []
        self.left_hip_gt = []
        self.right_hip_gt = []
        self.neck_gt = []
        self.torso_gt = []
        self.left_knee_gt = []
        self.right_knee_gt = []
        self.left_ankle_gt = []
        self.right_ankle_gt = []

        self.head_distance = 0
        self.left_ankle_distance = 0
        self.left_elbow_distance = 0
        self.left_hip_distance = 0
        self.left_knee_distance = 0
        self.left_shoulder_distance = 0
        self.left_wrist_distance = 0
        self.neck_distance = 0
        self.right_ankle_distance = 0
        self.right_elbow_distance = 0
        self.right_hip_distance = 0
        self.right_knee_distance = 0
        self.right_shoulder_distance = 0
        self.right_wrist_distance = 0
        self.torso_distance = 0

    def __get_test_samples(self):
        num_samples = 500
        selected_samples = np.random.choice(self.samples, num_samples, replace=False)

        for sample in selected_samples:
            data = get_pose(sample, self.json_dict, self.img_dir)
            image = data["img_data"]
            keypoint = data["joints"]
            self.images.append(image)
            self.gt_kps.append(keypoint)

        return selected_samples

    def __predict(self, image_path: str, index: int, model):
        """
        Test the model on a single image.
        :param image_path: path to image.
        :param index: index of image in dataset.
        :return: prediction keypoints, and ground truth keypoints.
        """
        img_file = image_path
        img = load_image(image_path)
        predictions = model.predict(img).reshape(-1, 15, 2) * 224
        ground_truth = np.array(self.gt_kps[index])
        ground_truth.reshape(-1, 15, 2)

        if self.visualize:
            self.__visualize_keypoints(predictions, img_file, [ground_truth])
        return predictions, ground_truth

    def __run_test(self):
        """
        Run the test, and print the results.
        :return: nothing.
        """
        for i in range(len(self.selected_samples)):
            pred, gt = self.__predict(self.img_dir + self.selected_samples[i], i)
            self.list_of_predictions.append(pred)
            self.list_of_gt.append(gt)

        controlled_pred, controlled_gt = self.__check_if_valid()
        self.__append_errors(controlled_pred)
        self.__append_gt(controlled_gt)
        self.__calculate_distance()

    def __check_if_valid(self):
        """
        Check if the predictions are valid or if they are not usable to measure the error.
        :return: list of valid predictions, and list of valid ground truths.
        """
        p, g = [], []
        invalid = 0

        for i in range(len(self.list_of_predictions)):
            checker = False

            for j in range(15):
                # print(list_of_gt[i][j][0])
                if self.list_of_predictions[i][0][j][0] < 10 or self.list_of_predictions[i][0][j][1] < 10:  # TODO: set to 0.
                    checker = True
                    invalid += 1
                elif self.list_of_gt[i][j][0] < 1 or self.list_of_gt[i][j][1] < 1:  # TODO: set to 0.
                    checker = True
                    invalid += 1
            if not checker:
                p.append(self.list_of_predictions[i])
                g.append(self.list_of_gt[i])
        print(Fore.RED, f'{invalid} invalid samples found...')
        return p, g

    def __append_errors(self, predictions):
        """
        Append errors to the list.
        :param predictions: list of predictions.
        :return: nothing
        """
        for i in predictions:
            self.head_errors.append(i[0][0])
            self.left_ankle_errors.append(i[0][1])
            self.left_elbow_errors.append(i[0][2])
            self.left_hip_errors.append(i[0][3])
            self.left_knee_errors.append(i[0][4])
            self.left_shoulder_errors.append(i[0][5])
            self.left_wrist_errors.append(i[0][6])
            self.neck_errors.append(i[0][7])
            self.right_ankle_errors.append(i[0][8])
            self.right_elbow_errors.append(i[0][9])
            self.right_hip_errors.append(i[0][10])
            self.right_knee_errors.append(i[0][11])
            self.right_shoulder_errors.append(i[0][12])
            self.right_wrist_errors.append(i[0][13])
            self.torso_errors.append(i[0][14])

    def __append_gt(self, gt):
        """
        Append ground truth to the list.
        :param gt: list of ground truths.
        :return: Nothing.
        """
        for i in range(len(gt)):
            self.head_gt.append(gt[i][0])
            self.left_ankle_gt.append(gt[i][1])
            self.left_elbow_gt.append(gt[i][2])
            self.left_hip_gt.append(gt[i][3])
            self.left_knee_gt.append(gt[i][4])
            self.left_shoulder_gt.append(gt[i][5])
            self.left_wrist_gt.append(gt[i][6])
            self.neck_gt.append(gt[i][7])
            self.right_ankle_gt.append(gt[i][8])
            self.right_elbow_gt.append(gt[i][9])
            self.right_hip_gt.append(gt[i][10])
            self.right_knee_gt.append(gt[i][11])
            self.right_shoulder_gt.append(gt[i][12])
            self.right_wrist_gt.append(gt[i][13])
            self.torso_gt.append(gt[i][14])

    def __calculate_distance(self):
        """
        Calculate the distance between the ground truth and the predictions, and print it to the console.
        """

        for i in range(len(self.head_errors)):
            self.head_distance += np.sqrt((self.head_errors[i][0] - self.head_gt[i][0]) ** 2 + (self.head_errors[i][1] - self.head_gt[i][1]) ** 2)
            self.left_ankle_distance += np.sqrt((self.left_ankle_errors[i][0] - self.left_ankle_gt[i][0]) ** 2 + (self.left_ankle_errors[i][1] - self.left_ankle_gt[i][1]) ** 2)
            self.left_elbow_distance += np.sqrt((self.left_elbow_errors[i][0] - self.left_elbow_gt[i][0]) ** 2 + (self.left_elbow_errors[i][1] - self.left_elbow_gt[i][1]) ** 2)
            self.left_hip_distance += np.sqrt((self.left_hip_errors[i][0] - self.left_hip_gt[i][0]) ** 2 + (self.left_hip_errors[i][1] - self.left_hip_gt[i][1]) ** 2)
            self.left_knee_distance += np.sqrt((self.left_knee_errors[i][0] - self.left_knee_gt[i][0]) ** 2 + (self.left_knee_errors[i][1] - self.left_knee_gt[i][1]) ** 2)
            self.left_shoulder_distance += np.sqrt((self.left_shoulder_errors[i][0] - self.left_shoulder_gt[i][0]) ** 2 + (self.left_shoulder_errors[i][1] - self.left_shoulder_gt[i][1]) ** 2)
            self.left_wrist_distance += np.sqrt((self.left_wrist_errors[i][0] - self.left_wrist_gt[i][0]) ** 2 + (self.left_wrist_errors[i][1] - self.left_wrist_gt[i][1]) ** 2)
            self.neck_distance += np.sqrt((self.neck_errors[i][0] - self.neck_gt[i][0]) ** 2 + (self.neck_errors[i][1] - self.neck_gt[i][1]) ** 2)
            self.right_ankle_distance += np.sqrt((self.right_ankle_errors[i][0] - self.right_ankle_gt[i][0]) ** 2 + (self.right_ankle_errors[i][1] - self.right_ankle_gt[i][0]) ** 2)
            self.right_elbow_distance += np.sqrt((self.right_elbow_errors[i][0] - self.right_elbow_gt[i][0]) ** 2 + (self.right_elbow_errors[i][1] - self.right_elbow_gt[i][0]) ** 2)
            self.right_hip_distance += np.sqrt((self.right_hip_errors[i][0] - self.right_hip_gt[i][0]) ** 2 + (self.right_hip_errors[i][1] - self.right_hip_gt[i][0]) ** 2)
            self.right_knee_distance += np.sqrt((self.right_knee_errors[i][0] - self.right_knee_gt[i][0]) ** 2 + (self.right_knee_errors[i][1] - self.right_knee_gt[i][0]) ** 2)
            self.right_shoulder_distance += np.sqrt((self.right_shoulder_errors[i][0] - self.right_shoulder_gt[i][0]) ** 2 + (self.right_shoulder_errors[i][1] - self.right_shoulder_gt[i][0]) ** 2)
            self.right_wrist_distance += np.sqrt((self.right_wrist_errors[i][0] - self.right_wrist_gt[i][0]) ** 2 + (self.right_wrist_errors[i][1] - self.right_wrist_gt[i][0]) ** 2)
            self.torso_distance += np.sqrt((self.torso_errors[i][0] - self.torso_gt[i][0]) ** 2 + (self.torso_errors[i][1] - self.torso_gt[i][0]) ** 2)

            # print("Model: ", model_name)
            print(Fore.MAGENTA, "Average Head distance: ", self.head_distance / len(self.head_errors))
            print(Fore.MAGENTA, "Average Neck distance: ", self.neck_distance / len(self.left_hip_errors))

            print(Fore.RED, "Average Right_shoulder distance: ", self.right_shoulder_distance / len(self.left_hip_errors))
            print(Fore.RED, "Average Right_elbow distance: ", self.right_elbow_distance / len(self.left_hip_errors))
            print(Fore.RED, "Average Right_wrist distance: ", self.right_wrist_distance / len(self.left_hip_errors))

            print(Fore.GREEN, "Average Left_shoulder distance: ", self.left_shoulder_distance / len(self.left_hip_errors))
            print(Fore.GREEN, "Average Left_elbow distance: ", self.left_elbow_distance / len(self.left_ankle_errors))
            print(Fore.GREEN, "Average Left_wrist distance: ", self.left_wrist_distance / len(self.left_hip_errors))

            print(Fore.MAGENTA, "Average Torso distance: ", self.torso_distance / len(self.left_hip_errors))
            print(Fore.RED, "Average Right_hip distance: ", self.right_hip_distance / len(self.left_hip_errors))
            print(Fore.RED, "Average Right_knee distance: ", self.right_knee_distance / len(self.left_hip_errors))
            print(Fore.RED, "Average Right_ankle distance: ", self.right_ankle_distance / len(self.left_hip_errors))

            print(Fore.GREEN, "Average Left_hip distance: ", self.left_hip_distance / len(self.left_hip_errors))
            print(Fore.GREEN, "Average Left_knee distance: ", self.left_knee_distance / len(self.left_hip_errors))
            print(Fore.GREEN, "Average Left_Ankle distance: ", self.left_ankle_distance / len(self.left_ankle_errors))

    def __visualize_keypoints(self, keypoints, image_file: str, gt_keypoints, rot: Optional[bool] = False):
        """
        Visualize keypoints on image.
        :param keypoints: Predicted keypoints.
        :param image_file: path to image.
        :param gt_keypoints: Ground truth keypoints.
        :param rot: True if images needs to be rotated, false otherwise.
        :return:
        """
        # colours = ['F633FF', '00FF1B', '00FF1B', '00FF1B', '00FF1B', '00FF1B', '00FF1B', 'F633FF',
        #            'FF0000', 'FF0000', 'FF0000', 'FF0000', 'FF0000', 'FF0000', 'F633FF']
        colours = ['#' + color for color in self.colors]  # TODO: fix to the correct colors.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

        plt.rcParams.update({'font.size': 16,
                             'text.color': 'white', })

        image = plt.imread(image_file)
        """rotate image 90 degrees to the right"""
        # remove the next 3 lines if you don't want to rotate the image
        if rot:
            image = np.rot90(image, k=1, axes=(0, 1))
            image = np.flipud(image)
            image = np.fliplr(image)

        ax1.imshow(image)
        ax1.set_title('Prediction')
        ax2.imshow(image)
        ax2.set_title('Ground Truth')

        for current_keypoint in keypoints:
            current_keypoint = np.array(current_keypoint)
            # Since the last entry is the visibility flag, we discard it.
            current_keypoint = current_keypoint[:, :2]
            for idx, (x, y) in enumerate(current_keypoint):
                ax1.scatter([x], [y], c=colours[idx], marker="x", s=50, linewidths=5)

        for current_keypoint in gt_keypoints:
            current_keypoint = np.array(current_keypoint)
            # Since the last entry is the visibility flag, we discard it.
            current_keypoint = current_keypoint[:]
            for idx, (x, y) in enumerate(current_keypoint):
                ax2.scatter([x], [y], c=colours[idx], marker="x", s=50, linewidths=5)

        plt.tight_layout(pad=2.0)
        plt.show()
