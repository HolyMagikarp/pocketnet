import cv2 as cv
import numpy as np
import os
import json

NAME_ID_MAP = json.load(open('./names.json', 'r'))


BASE_DIR = "./dataset/"
SAVE_BASE_DIR = BASE_DIR + "preprocessed/"
VALID_EXTENSIONS = ['.jpg', '.png', '.jpeg']

def downsample(image, size):

    levels = int(np.floor(np.log2(image.shape[0] / size)))

    desired_size = int(size * (2 ** levels))

    resized = cv.resize(image, (desired_size, desired_size))

    current_image = resized

    for _ in range(levels):
        new_size = current_image.shape[0] // 2
        current_image = cv.pyrDown(current_image, None, (new_size, new_size))

    return current_image



# takes the largest centered square crop of the input image
def square(image):
    (height, width, _) = image.shape


    diff = max(height, width) - min(height, width)

    pad1 = diff // 2
    pad2 = diff - pad1

    if height > width:
        padded = cv.copyMakeBorder(image, 0, 0, pad1, pad2, cv.BORDER_CONSTANT )
    else:
        padded = cv.copyMakeBorder(image, pad1, pad2, 0, 0, cv.BORDER_CONSTANT )

    return padded


def preprocess(directory, size, source):
    input_path = BASE_DIR + source + '/' + directory

    if not os.path.exists(input_path):
        print("No directory: {}".format(input_path))
        return

    filenames = os.listdir(input_path)
    os.makedirs(SAVE_BASE_DIR + directory, exist_ok=True)

    total = len(filenames)
    counter = 0
    for filename in filenames:
        name, extension = os.path.splitext(filename)

        if extension not in VALID_EXTENSIONS:
            continue

        image = cv.imread(input_path + '/' + filename)

        if image is None:
            continue

        squared = square(image)
        mini = downsample(squared, size)

        cv.imwrite(SAVE_BASE_DIR + directory + '/' + filename, mini)

        counter += 1
        print("Preprocessed {}/{} from {}\n".format(counter, total, directory))

def create_reflection(image):
    pass

if __name__ == "__main__":


    for p in ['pikachu', 'gengar']:#list(NAME_ID_MAP.keys())[:30]:
        preprocess(p, 100, "bing")
