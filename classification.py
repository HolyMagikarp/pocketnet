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

def load_graph():
    with gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def one_hot_labels(labels, classes):
    a = np.array(labels)
    b = np.zeros((len(labels), classes))
    b[np.arange(len(labels)), a] = 1

    return b

def extract_pokemon_features(pokemon, source):
    data_dir = IMAGE_DIR + source + '/' + pokemon + '/'
    list_images = [data_dir + f for f in os.listdir(data_dir) if re.search('jpg|JPG|png|PNG', f)]

    features = extract_features(list_images)

    return features, list_images

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

def train_svm_classifier(train_features, train_labels):
    classifier = svm.SVC(gamma='scale')
    classifier.fit(train_features, train_labels)

    dump(classifier, 'saves/svm_classifier.joblib')

    return classifier

def svm_classify(test_image):

    classifier = load('saves/svm_classifier.joblib')
    test_feature = extract_features([test_image])
    guess = classifier.predict(test_feature)

    return classifier, guess

def svm_test(classifier, test_features, test_labels):
    predictions = classifier.predict(test_features)
    correct = [i for i in range(len(predictions)) if predictions[i] == test_labels[i]]

    print("Accuracy: {}\n".format(len(correct) / len(predictions) * 100))
    return len(correct) / len(predictions) * 100

def train_nn_classifier(train_features, train_labels):
    num_classes = len(set(train_labels))

    #labels = one_hot_labels(train_labels, num_classes)
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

def nn_test(model, test_features, test_labels):
    predictions = model.predict(test_features)
    correct = [i for i in range(len(predictions)) if np.argmax(predictions[i]) + 1 == test_labels[i]]

    print("Accuracy: {}\n".format(len(correct) / len(predictions) * 100))
    return len(correct) / len(predictions) * 100


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

def load_data():
    train_x = np.load(open('saves/classification_v1_train_features.npy', 'rb'))
    train_y = np.load(open('saves/classification_v1_train_labels.npy', 'rb'))

    test_x = np.load(open('saves/classification_v1_test_features.npy', 'rb'))
    test_y = np.load(open('saves/classification_v1_test_labels.npy', 'rb'))

    filenames = np.load(open('saves/classification_v1_filenames.npy', 'rb'))

    return train_x, train_y, test_x, test_y, filenames

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


def show_results(x, y, filenames, classes):
    classifier = load('saves/svm_classifier.joblib')

    wrong = {}
    right = {}
    for i in range(len(y)):
        prediction = classifier.predict([x[i]])[0]

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

if __name__ == "__main__":

    classes = list(NAME_ID_MAP.keys())[:10]
    #classes = ['pikachu', 'charmander', 'gastly', 'haunter', 'gengar', 'meowth']
    extract_and_save_features(classes, 0.3)

    features, labels, test_features, test_labels, filenames = load_data()

    #print("Training NN classifier")
    nn_model = train_nn_classifier(features, labels)
    accuracy = nn_test(nn_model, test_features, test_labels)

    print("Training SVM classifier")
    svm = train_svm_classifier(features, labels)
    accuracy = svm_test(svm, test_features, test_labels)

    wrong = show_results(test_features, test_labels, filenames, classes)
    print("END\n")
