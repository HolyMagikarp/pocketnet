import cv2 as cv
import numpy as np
import os
import json
import matplotlib.pyplot as plt

NAME_ID_MAP = json.load(open('./names.json', 'r'))

# preprocessed and augmented data is stored in /dataset/preprocessed/pokemon_name
# where images for each pokemon is stored in its own folder
BASE_DIR = "./dataset/"
SAVE_BASE_DIR = BASE_DIR + "preprocessed/"
VALID_EXTENSIONS = ['.jpg', '.png', '.jpeg']

# downsamples a square image  to the given size
# input image must be square
def downsample(image, size):

    # calculate how many levels to downsample in image pyramid
    levels = int(np.floor(np.log2(image.shape[0] / size)))
    desired_size = int(size * (2 ** levels))
    resized = cv.resize(image, (desired_size, desired_size))
    current_image = resized

    # downsample the image to the specified size
    for _ in range(levels):
        new_size = current_image.shape[0] // 2
        current_image = cv.pyrDown(current_image, None, (new_size, new_size))

    return current_image

# pads the shorter side of the image to return a square image
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

# preprocesses all images in the directory:
# /dataset/source/directory
# preprocess involves turning the image into a square image, and then
# downsampling it to the specified size
# the augment argument is a boolean indicating whether or not to
# generate augmented images in the preprocess step
def preprocess(directory, size, source, augment):
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
        images = augment_image(mini, augment)

        for i in images:
            name = filename.split('.')
            name = str(counter) + '.' + name[-1]
            cv.imwrite(SAVE_BASE_DIR + directory + '/' + name, i)

            counter += 1
        print("Preprocessed {}/{} from {}\n".format(counter, total, directory))

# helper for augmentation that creates a horizontal reflection
def create_reflections(image):
    return cv.flip(image, 1)

# helper for augmentation that adds salt and pepper noise to the image
# salt and pepper noise random pixels to black or white
# amount indicates the percent of pixels to turn into noise
def add_noise(image, amount):
    size = image.shape
    total = np.prod(size[:-1])
    coordinates = []
    for i in range(size[0]):
        for j in range(size[1]):
            coordinates.append((i, j))

    choices = np.random.choice(np.arange(total), int(amount * total), False)
    choices = np.array(coordinates)[choices]
    copy = np.zeros(image.shape)

    copy[:, :, :] = image
    for choice in choices:
        colour = np.random.choice([0, 255])
        copy[choice[0], choice[1], :] = colour

    return copy

# helper that returns a list of image augmentations including
# the original image
# proceed indicates whether or not to include augmented images in
# the return
def augment_image(image, proceed=False):
    results = []
    results.append(image)
    if proceed:
        reflection = create_reflections(image)
        results.append(reflection)
        results.append(add_noise(image, 0.10))
        results.append(add_noise(reflection, 0.10))

    return results

# running this file preprocesses and augment pokemon images in /dataset/bing and saves them in
# /dataset/preprocessed
# the for loop determines which pokemon to preprocess
# by default it preprocesses the first 10 pokemon, but can be changed
if __name__ == "__main__":

    # for loop can be changed to preprocess specific pokemon
    for p in list(NAME_ID_MAP.keys())[:10]:
    #for p in ['pikachu', 'bulbasaur', 'charmander', 'squirtle', 'zubat']:
        preprocess(p, 100, "bing", True)
