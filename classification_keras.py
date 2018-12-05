from keras import applications
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Activation, MaxPooling2D
from keras.models import model_from_json

from sklearn.model_selection import train_test_split
from sklearn import svm

import numpy as np
import os
import cv2 as cv
import json
import matplotlib.pyplot as plt
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32
TEST_BATCH_SIZE = 16
# this variable needs to be set to the number of different pokemons in the
# /dataset/preprocessed directory
NUM_CLASSES = 5

# NOT USED IN FINAL REPORT
# this is an alternative approach to transfer training using the keras api
# extracts features from images specified by the train_path and test_path
# the keras api allows several models trained on imagenet to be loaded and used
# this model performed very poorly and I could not figure out where I went wrong
def extract_features(train_path, test_path):
    train_gen, test_gen = create_data_generators(train_path, test_path)

    model = applications.VGG16(include_top=False, input_shape=(100, 100, 3), pooling='max', weights='imagenet')

    features = model.predict_generator(train_gen, steps=train_gen.n // BATCH_SIZE, verbose=1)

    np.save(
        open('saves/classification_train_features.npy', 'wb'),
        features)

    np.save(
        open('saves/classification_train_labels.npy', 'wb'),
        train_gen.classes
    )
    print("Saved training arrays\n")

# helper that creates a train and test data generator for the keras api
# all data are taken from the data_path, and are then split into 70-30 train-test sets
# the generator also adds augmentations of rotation and horizontal flips
# images produced by the generator is normalized to 0-1
def create_data_generators(data_path):

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.3,
        rotation_range=20,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(data_path,
        target_size=(100,100),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(100, 100),
        batch_size=TEST_BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        subset='validation'
    )

    return train_generator, validation_generator

# NOT USED IN THE FINAL REPORT
# a helper that splits features extracted by a pretrained network into train and test sets
def shuffle_and_split(features, labels, split):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=split, random_state=42)

    return x_train, x_test, y_train, y_test

# NOT USED IN THE FINAL REPORT
# trains a neural network classifier on the features extracted by a pretrained CNN
def train_model():
    features = np.load(open('saves/classification_train_features.npy', 'rb'))
    labels = np.load(open('saves/classification_train_labels.npy', 'rb'))[:features.shape[0]]

    num_classes = len(set(labels))

    x_train, x_test, y_train, y_test = shuffle_and_split(features, labels, 0.3)

    model = Sequential()
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=50,
        validation_data=(x_test, y_test)
    )

    model.save_weights('saves/classification_model.h5')

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("NN Accuracy: ", test_acc)

    test_svm_model(x_train, y_train, x_test, y_test)

# creates a custom CNN with 4 convolution layers and 2 fully connected layers
# trains the CNN using train and test data generators using images from the train_path directory
def train_custom_model(train_path):
    train_gen, test_gen = create_data_generators(train_path)

    model = Sequential()
    # convolution layer with 32 filters of size 3x3
    # takes imput from a 100x100 coloured image
    model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
    model.add(Activation('relu'))
    # max pools the results to be fed into the next layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # convolution layer with 64 filters of size 3x3
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    # max pools the results to be fed into the next layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # convolution layer with 128 filters of size 3x3
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    # max pools the results to be fed into the next layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # convolution layer with 128 filters of size 3x3
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    # max pools the results to be fed into the next layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # flattens the 3d convolution output
    model.add(Flatten())
    # fully connected layer of 128 units
    model.add(Dense(128, activation='relu'))
    # output layer
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # uses the Adam optimizer and categorical_crossentropy loss
    # because the data generators return one-hot encoded labels
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.n // BATCH_SIZE,
        epochs=50,
        validation_data=test_gen,
        validation_steps=test_gen.n // TEST_BATCH_SIZE
    )

    # save both the model and the weights
    model.save_weights('saves/custom_model.h5')
    model_json = model.to_json()
    with open('saves/custom_model.json', 'w') as json_file:
        json_file.write(model_json)

    return model

# NOT USED IN THE FINAL REPORT
# trains and tests a svm on features extracted by a pretrained network
# this svm also performed badly, despite the same strategy working in classification.py
def test_svm_model(x_train, y_train, x_test, y_test):

    classifier = svm.SVC(gamma='scale')
    classifier.fit(reshape_to_matrix(x_train), y_train)
    predictions = classifier.predict(reshape_to_matrix(x_test))
    correct = [i for i in range(len(predictions)) if predictions[i] == y_test[i]]

    print("SVM Accuracy: {}\n".format(len(correct) / len(predictions) * 100))
    return len(correct) / len(predictions) * 100

# NOT USED IN FINAL REPORT
def reshape_to_matrix(tensor):
    matrix = tensor.reshape((tensor.shape[0], np.prod(tensor.shape[1:])))
    return matrix

# NOT USED IN FINAL REPORT
def transer_training(train_path, test_path, get_features=False):

    if get_features:
        extract_features(train_path, test_path)

    train_model()


# loads the custom CNN saved in train_custom_model
# evaluates the model using test data from data_dir
def evaluate_model(data_dir):

    # load and compile model
    json_file = open('saves/custom_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights('saves/custom_model.h5')

    loaded_model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])


    train_gen, test_gen = create_data_generators(data_dir)

    loss, accuracy = loaded_model.evaluate_generator(test_gen, steps=test_gen.n // TEST_BATCH_SIZE)
    print("evaluation loss: {}, accuracy: {}".format(loss, accuracy))


# runs the full pipeline for the custom CNN model
# the global variable NUM_CLASSES must be set to the number of pokemon directories in /dataset/preprocessed
if __name__ == "__main__":

    # trains the CNN
    train_custom_model('dataset/preprocessed')
    # evaluates the model on a specific pokemon
    evaluate_model('dataset/preprocessed')