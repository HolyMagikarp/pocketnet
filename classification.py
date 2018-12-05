import cv2 as cv
import numpy as np
import os
import re
import json
import tensorflow as tf
from sklearn import svm
from tensorflow.python.platform import gfile
from joblib import dump, load
import matplotlib.pyplot as plt

MODEL_DIR = 'graphs'
IMAGE_DIR = 'dataset/'

NAME_ID_MAP = json.load(open('./names.json', 'r'))

# loads the model graph called 'classify_image_graph_def.pb' stored in the graphs directory
def load_graph():
    with gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

# NOT USED IN FINAL REPORT
# helper that one hot encodes labels
def one_hot_labels(labels, classes):
    a = np.array(labels)
    b = np.zeros((len(labels), classes))
    b[np.arange(len(labels)), a] = 1

    return b

# helper that extracts features for all images located in
# /dataset/source/pokemon
def extract_pokemon_features(pokemon, source):
    data_dir = IMAGE_DIR + source + '/' + pokemon + '/'
    list_images = [data_dir + f for f in os.listdir(data_dir) if re.search('jpg|JPG|png|PNG', f)]

    features = extract_features(list_images)

    return features, list_images

# loads InceptionV3 along with Imagenet weights, and processes all images in list_images
# the features are taken as the output of the last pooling layer of the network
# returns a list of features that corresponds to the list_images
def extract_features(list_images):

    nb_features = 2048
    features = np.empty((len(list_images),nb_features))

    load_graph()

    with tf.Session() as sess:

        feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        total = len(list_images)
        for i, image in enumerate(list_images):
            if (i % 10 == 0):
                print('Progress {:.2f}%'.format(i / total * 100))

            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()

            predictions = sess.run(feature_tensor,
                {'DecodeJpeg/contents:0': image_data})

            features[i,:] = np.squeeze(predictions)

    print('Progress: 100.00%\n')
    return features

# trains a support vector machine classifier
# saves the classifier at 'saves/svm_classifier.joblib'
def train_svm_classifier(train_features, train_labels):
    classifier = svm.SVC(gamma='scale')
    classifier.fit(train_features, train_labels)

    dump(classifier, 'saves/svm_classifier.joblib')

    return classifier

# classifies the test_image with the svm classifier saved at
# 'saves/svm_classifier.joblib'
def svm_classify(test_image):

    classifier = load('saves/svm_classifier.joblib')
    test_feature = extract_features([test_image])
    guess = classifier.predict(test_feature)

    return classifier, guess

# computes the accuracy of the classifier
def svm_test(classifier, test_features, test_labels):
    predictions = classifier.predict(test_features)
    correct = [i for i in range(len(predictions)) if predictions[i] == test_labels[i]]

    print("Accuracy: {}\n".format(len(correct) / len(predictions) * 100))
    return len(correct) / len(predictions) * 100

# trains a feed forward neural network classifier
def train_nn_classifier(train_features, train_labels):
    num_classes = len(set(train_labels))

    labels = one_hot_labels(train_labels, num_classes)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(2048, activation=tf.nn.relu),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_features, train_labels, epochs=5, shuffle=True)

    return model

# computes the accuracy of the neural network classifier
def nn_test(model, test_features, test_labels):
    predictions = model.predict(test_features)
    correct = [i for i in range(len(predictions)) if np.argmax(predictions[i]) + 1 == test_labels[i]]

    print("Accuracy: {}\n".format(len(correct) / len(predictions) * 100))
    return len(correct) / len(predictions) * 100

# saves extracted features and labels into the saves directory
# this eliminates the need to re-extract the same features
# when retraining the classifiers
def save_data(train_x, train_y, test_x, test_y, filenames):
    np.save(
        open('saves/classification_v1_train_features.npy', 'wb'),
        train_x)

    np.save(
        open('saves/classification_v1_train_labels.npy', 'wb'),
        train_y
    )

    np.save(
        open('saves/classification_v1_test_features.npy', 'wb'),
        test_x
    )

    np.save(
        open('saves/classification_v1_test_labels.npy', 'wb'),
        test_y
    )

    np.save(
        open('saves/classification_v1_filenames.npy', 'wb'),
        filenames
    )

# loads the saved features and labels from save_data
def load_data():
    train_x = np.load(open('saves/classification_v1_train_features.npy', 'rb'))
    train_y = np.load(open('saves/classification_v1_train_labels.npy', 'rb'))

    test_x = np.load(open('saves/classification_v1_test_features.npy', 'rb'))
    test_y = np.load(open('saves/classification_v1_test_labels.npy', 'rb'))

    filenames = np.load(open('saves/classification_v1_filenames.npy', 'rb'))

    return train_x, train_y, test_x, test_y, filenames

# helper that creates training and test data by using
# the pretrained neural network to extract features for pokemon in targets
# and then splits the features by test_ratio into training and test data
# finally saves the data in the saves/ directory
def extract_and_save_features(targets, test_ratio):
    features, labels = [], []
    test_features, test_labels = [], []
    test_filenames = []
    i = 0
    for p in targets:

        print("Extracting features for {}".format(p))
        f, filenames = extract_pokemon_features(p, "preprocessed")
        l = [i] * f.shape[0]
        test_size = int(len(l) * test_ratio)

        features.append(f[test_size:, :])
        labels.append(l[test_size:])

        test_features.append(f[:test_size, :])
        test_filenames.append(filenames[:test_size])
        test_labels.append(l[:test_size])

        i += 1

    train_x = np.concatenate(features, axis=0)
    train_y = np.concatenate(labels, axis=0)
    test_x = np.concatenate(test_features, axis=0)
    test_y = np.concatenate(test_labels, axis=0)
    filenames = np.concatenate(test_filenames, axis=0)

    save_data(train_x, train_y, test_x, test_y, filenames)

# creates an image that shows 3 good predictions and 3 bad predictions of the svm classifier
# the image is called classification_summary and is saved in the base project directory
def show_results(x, y, filenames, classes):
    classifier = load('saves/svm_classifier.joblib')

    wrong = {}
    right = {}
    for i in range(len(y)):
        prediction = int(classifier.predict([x[i]])[0])

        if classes[prediction] not in wrong.keys():
            wrong[classes[prediction]] = []
            right[classes[prediction]] = []

        if prediction != y[i]:
            wrong[classes[prediction]].append((filenames[i], classes[y[i]]))
        else:
            right[classes[prediction]].append((filenames[i], classes[y[i]]))

    num_row = len(wrong.keys())
    num_col = 6

    plt.figure()
    row = 0
    for k in wrong.keys():
        col = 1

        for r in right[k][:3]:
            image = cv.imread(r[0])
            if image is None:
                break
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            position = row * num_col + col
            plt.axis('off')
            plt.subplot(num_row, num_col, position)
            plt.imshow(image)

            col += 1

        col = 4
        for r in wrong[k][:3]:
            image = cv.imread(r[0])
            if image is None:
                break
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            position = row * num_col + col
            plt.axis('off')
            plt.subplot(num_row, num_col, position)
            plt.imshow(image)

            col += 1
        row += 1

    plt.axis('off')
    plt.savefig('./classification_summary.png')
    plt.show()

    return wrong

# runs the full transfer training pipeline
# needs the preprocessed directory to be populated by subdirectories of preprocessed pokemon images
# need to clone the tensorflow/models directory into the base project directory
# found here: https://github.com/tensorflow/models
# then execute: python /models/tutorial/imagenet/classify_image.py -model_dir graphs
# this saves the InceptionV3 model and weights in the graphs directory
if __name__ == "__main__":

    # classes can be set to determine exactly which pokemon to classify
    # by default it classifies the first 10
    classes = list(NAME_ID_MAP.keys())[:10]
    #classes = ['pikachu', 'charmander', 'gastly', 'haunter', 'gengar', 'meowth']

    # extracts features with InceptionV3
    extract_and_save_features(classes, 0.3)

    # loads extracted data
    features, labels, test_features, test_labels, filenames = load_data()

    # trains the neural network classifier
    nn_model = train_nn_classifier(features, labels)
    accuracy = nn_test(nn_model, test_features, test_labels)

    # trains the svm classifier
    svm = train_svm_classifier(features, labels)
    accuracy = svm_test(svm, test_features, test_labels)

    # creates a summary of good and bad predictions using the svm classifier
    wrong = show_results(test_features, test_labels, filenames, classes)
    print("END\n")
