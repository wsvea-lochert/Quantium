from typing import Optional
from Filch.FilchUtils import load_image, get_model, visualize_keypoints


def predict(image_path, model_path, visualize: Optional[bool] = False):
    """
    Predict the image using the model
    :param image_path: path to the image
    :param model_path: path to the model
    :param visualize: whether to visualize the keypoints
    :return: the predicted keypoints
    """
    image = load_image(image_path)
    model = get_model(model_path)
    prediction = model.predict(image).reshape(-1, 15, 2) * 224
    if visualize:
        visualize_keypoints(prediction, image_path, False)
    return prediction
