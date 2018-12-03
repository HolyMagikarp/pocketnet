from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn import svm


import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

BATCH_SIZE = 16

def extract_features(train_path, test_path):
    train_gen, test_gen = create_data_generators(train_path, test_path)

    model = applications.InceptionV3(include_top=False, weights='imagenet')

    features = model.predict_generator(train_gen, steps=train_gen.n // BATCH_SIZE, verbose=1)

    np.save(
        open('saves/classification_train_features.npy', 'wb'),
        features)

    np.save(
        open('saves/classification_train_labels.npy', 'wb'),
        train_gen.classes
    )
    print("Saved training arrays\n")
    test_features = model.predict_generator(test_gen, steps=test_gen.n // BATCH_SIZE, verbose=1)

    np.save(
        open('saves/classification_test_features.npy', 'wb'),
        test_features)

    np.save(
        open('saves/classification_test_labels.npy', 'wb'),
        test_gen.classes
    )
    print("Saved test arrays\n")
def create_data_generators(train_path, test_path):

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        zoom_range=0.2)

    train_generator = train_datagen.flow_from_directory(train_path,
        target_size=(100,100),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True)

    test_datagen = ImageDataGenerator()

    validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(100, 100),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True)

    return train_generator, validation_generator

def shuffle_and_split(features, labels, split):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=split, random_state=42)

    return x_train, x_test, y_train, y_test


def train_model():
    features = np.load(open('saves/classification_train_features.npy', 'rb'))
    labels = np.load(open('saves/classification_train_labels.npy', 'rb'))[:features.shape[0]]

    num_classes = np.max(labels) + 1

    x_train, x_test, y_train, y_test = shuffle_and_split(features, labels, 0.3)

    model = Sequential()
    model.add(Flatten(input_shape=features.shape[1:]))
    #model.add(Dense(2048, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(29, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=15,
        validation_data=(x_test, y_test)
    )

    model.save_weights('saves/classification_model.h5')

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("NN Accuracy: ", test_acc)

    #test_model(model, x_test, y_test)

    classifier = svm.SVC(gamma='scale')
    classifier.fit(x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])), y_train)
    predictions = classifier.predict(x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])))
    correct = [i for i in range(len(predictions)) if predictions[i] == y_test[i]]

    print("SVM Accuracy: {}\n".format(len(correct) / len(predictions) * 100))
    return len(correct) / len(predictions) * 100

def test_model(model, x_test, y_test):

    predictions = model.predict(
        x_test,
        verbose=1)

    predictions = np.argmax(predictions, axis=1)

    print(len(y_test), len(predictions))
    correct = [i for i in range(len(predictions)) if predictions[i] == y_test[i]]
    print(len(correct) / len(predictions))

def main(train_path, test_path, get_features=False):

    if get_features:
        extract_features(train_path, test_path)

    train_model()




if __name__ == "__main__":
    main('dataset/preprocessed', 'dataset/sprites', True)



