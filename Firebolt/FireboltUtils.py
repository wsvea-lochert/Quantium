import json
from imgaug.augmentables import Keypoint, KeypointsOnImage


def get_json_dict(input_json: str):
    """
    Function for getting the json dictionary.
    :param input_json:
    :return: A json dictionary containing all image names and their keypoints.
    """

    with open(input_json) as infile:
        json_data = json.load(infile)

    json_dict = {}

    for i in range(len(json_data)):
        tmp_obj = json_data['image' + str(i)]
        y = {tmp_obj['image']: {'image_path': tmp_obj['image'],
                                'joints': [[tmp_obj['head']['x'], tmp_obj['head']['y']],
                                           [tmp_obj['left_ankle']['x'], tmp_obj['left_ankle']['y']],
                                           [tmp_obj['left_elbow']['x'], tmp_obj['left_elbow']['y']],
                                           [tmp_obj['left_hip']['x'], tmp_obj['left_hip']['y']],
                                           [tmp_obj['left_knee']['x'], tmp_obj['left_knee']['y']],
                                           [tmp_obj['left_shoulder']['x'], tmp_obj['left_shoulder']['y']],
                                           [tmp_obj['left_wrist']['x'], tmp_obj['left_wrist']['y']],
                                           [tmp_obj['neck']['x'], tmp_obj['neck']['y']],
                                           [tmp_obj['right_ankle']['x'], tmp_obj['right_ankle']['y']],
                                           [tmp_obj['right_elbow']['x'], tmp_obj['right_elbow']['y']],
                                           [tmp_obj['right_hip']['x'], tmp_obj['right_hip']['y']],
                                           [tmp_obj['right_knee']['x'], tmp_obj['right_knee']['y']],
                                           [tmp_obj['right_shoulder']['x'], tmp_obj['right_shoulder']['y']],
                                           [tmp_obj['right_writs']['x'], tmp_obj['right_writs']['y']],
                                           [tmp_obj['torso']['x'], tmp_obj['torso']['y']]
                                           ]}}
        json_dict.update(y)

    return json_dict


def get_kpsoi(keypoints, img_shape):
    kps = KeypointsOnImage([
        Keypoint(x=keypoints[0][0], y=keypoints[0][1]),
        Keypoint(x=keypoints[1][0], y=keypoints[1][1]),
        Keypoint(x=keypoints[2][0], y=keypoints[2][1]),
        Keypoint(x=keypoints[3][0], y=keypoints[3][1]),
        Keypoint(x=keypoints[4][0], y=keypoints[4][1]),
        Keypoint(x=keypoints[5][0], y=keypoints[5][1]),
        Keypoint(x=keypoints[6][0], y=keypoints[6][1]),
        Keypoint(x=keypoints[7][0], y=keypoints[7][1]),
        Keypoint(x=keypoints[8][0], y=keypoints[8][1]),
        Keypoint(x=keypoints[9][0], y=keypoints[9][1]),
        Keypoint(x=keypoints[10][0], y=keypoints[10][1]),
        Keypoint(x=keypoints[11][0], y=keypoints[11][1]),
        Keypoint(x=keypoints[12][0], y=keypoints[12][1]),
        Keypoint(x=keypoints[13][0], y=keypoints[13][1]),
        Keypoint(x=keypoints[14][0], y=keypoints[14][1])
    ], shape=img_shape)

    return kps
