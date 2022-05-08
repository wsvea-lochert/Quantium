import os
from PIL import Image


class FireboltResizer:
    def __init__(self, directory, output, size):
        self.directory = directory
        self.output = output
        self.size = size

    def resize(self):
        images = os.listdir(self.directory)
        for image in images:
            img = Image.open(self.directory + '/' + image)
            resized = img.resize((self.size, self.size))
            resized.save(self.output + '/' + image)


# def resize_images(directory, output, size):
#     ImageResizer(directory, output, size).resize()