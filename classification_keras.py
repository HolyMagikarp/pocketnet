from keras import applications
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Activation, MaxPooling2D

from sklearn.model_selection import train_test_split
from sklearn import svm

import numpy as np

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

def extract_features(train_path, test_path):
    train_gen, test_gen = create_data_generators(train_path, test_path)

    model = applications.VGG16(include_top=False, input_shape=(100, 100, 3), pooling='max', weights='imagenet')

    #steps=train_gen.n // BATCH_SIZE
    features = model.predict_generator(train_gen, steps=train_gen.n // BATCH_SIZE, verbose=1)

    np.save(
        open('saves/classification_train_features.npy', 'wb'),
        features)

    np.save(
        open('saves/classification_train_labels.npy', 'wb'),
        train_gen.classes
    )
    print("Saved training arrays\n")

def create_data_generators(data_path):

    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.3)

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

def shuffle_and_split(features, labels, split):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=split, random_state=42)

    return x_train, x_test, y_train, y_test


def train_model():
    features = np.load(open('saves/classification_train_features.npy', 'rb'))
    labels = np.load(open('saves/classification_train_labels.npy', 'rb'))[:features.shape[0]]

    num_classes = len(set(labels))

    x_train, x_test, y_train, y_test = shuffle_and_split(features, labels, 0.3)

    model = Sequential()
    #model.add(Flatten(input_shape=features.shape[1:]))
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.5))
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

def train_custom_model(train_path):
    train_gen, test_gen = create_data_generators(train_path)
    num_classes = 10



    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(100, 100, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])


    model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.n // BATCH_SIZE,
        epochs=50,
        validation_data=test_gen,
        validation_steps=test_gen.n // TEST_BATCH_SIZE
    )

    model.save_weights('saves/custom_model.h5')

def test_svm_model(x_train, y_train, x_test, y_test):

    classifier = svm.SVC(gamma='scale')
    classifier.fit(reshape_to_matrix(x_train), y_train)
    predictions = classifier.predict(reshape_to_matrix(x_test))
    correct = [i for i in range(len(predictions)) if predictions[i] == y_test[i]]

    print("SVM Accuracy: {}\n".format(len(correct) / len(predictions) * 100))
    return len(correct) / len(predictions) * 100

def reshape_to_matrix(tensor):
    matrix = tensor.reshape((tensor.shape[0], np.prod(tensor.shape[1:])))
    return matrix

def main(train_path, test_path, get_features=False):

    if get_features:
        extract_features(train_path, test_path)

    train_model()




if __name__ == "__main__":
    #main('dataset/preprocessed')

    train_custom_model('dataset/preprocessed')

