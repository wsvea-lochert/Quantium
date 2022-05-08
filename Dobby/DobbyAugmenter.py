import imgaug.augmenters as iaa


def get_augmentation_parameters():
    """
    Returns the augmentation parameters for the DobbyAugmentert.
    :return: Returns the augmentation parameters for the DobbyAugmentert.
    """
    return iaa.Sequential([
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


def get_validation_augmentation_parameters():
    return iaa.Sequential([
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
