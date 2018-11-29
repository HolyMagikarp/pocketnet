import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import os

BASE_DIR = "./dataset/"
SAVE_BASE_DIR = BASE_DIR + "preprocessed/"
VALID_EXTENSIONS = ['.jpg', '.png', '.jpeg']

def downsample(image, size):

    levels = int(np.floor(np.log2(image.shape[0] / size)))

    desired_size = size * (2 ** levels)

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
        padded = cv.copyMakeBorder(image, 0, 0, pad1, pad2, cv.BORDER_REPLICATE)
    else:
        padded = cv.copyMakeBorder(image, pad1, pad2, 0, 0, cv.BORDER_REPLICATE)

    return padded


def preprocess(directory, size):
    input_path = BASE_DIR + directory
    filenames = os.listdir(input_path)
    os.makedirs(SAVE_BASE_DIR + directory, exist_ok=True)

    total = len(filenames)
    counter = 0
    for filename in filenames:
        name, extension = os.path.splitext(filename)

        if extension not in VALID_EXTENSIONS:
            continue

        image = cv.imread(input_path + '/' + filename)
        #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        squared = square(image)
        mini = downsample(squared, size)

        cv.imwrite(SAVE_BASE_DIR + directory + '/' + filename, mini)

        counter += 1
        print("Preprocessed {}/{} from {}\n".format(counter, total, directory))


if __name__ == "__main__":
    # image = cv.imread("/Users/dannyliu/Documents/repos/pocketnet/dataset/pikachu/00000000.jpg")
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()
    #
    # square = square(image)
    # plt.imshow(square)
    # plt.show()
    #
    #
    # mini = downsample(square, 100)
    # plt.imshow(mini)
    # plt.show()

    preprocess('bulbasaur', 100)
