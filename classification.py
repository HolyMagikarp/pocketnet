import keras
import numpy as np
import os
import re
import json
import tensorflow as tf
from sklearn import svm
from tensorflow.python.platform import gfile
from keras.applications import vgg16, inception_v3, resnet50, mobilenet

#inception_model = inception_v3.InceptionV3(weights='imagenet')


MODEL_DIR = 'graphs'
IMAGE_DIR = 'dataset/'

NAME_ID_MAP = json.load(open('./names.json', 'r'))

def load_graph():
    with gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def one_hot_labels(labels):
    a = np.array(labels)
    b = np.zeros((len(labels), np.max(a)))
    b[np.arange(len(labels)), a - 1] = 1

    return b

def extract_pokemon_features(pokemon, source):
    label = NAME_ID_MAP[pokemon]
    data_dir = IMAGE_DIR + source + '/' + pokemon + '/'
    list_images = [data_dir + f for f in os.listdir(data_dir) if re.search('jpg|JPG|png|PNG', f)]

    features = extract_features(list_images)
    labels = [label] * features.shape[0]

    return features, labels

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


    print('Progress: 100.00%')
    return features

def train_svm_classifier(train_features, train_labels):
    classifier = svm.SVC(gamma='scale')
    classifier.fit(train_features, train_labels)

    return classifier

def svm_classify(classifier, test_image):

    test_feature = extract_features([test_image])
    guess = classifier.predict(test_feature)

    print(guess)

def svm_test(classifier, test_features, test_labels):
    predictions = classifier.predict(test_features)
    correct = [i for i in range(len(predictions)) if predictions[i] == test_labels[i]]

    print("Accuracy: {}\n".format(len(correct) / len(predictions) * 100))
    return len(correct) / len(predictions) * 100

def train_nn_classifier(train_features, train_labels):
    num_classes = np.max(train_labels)

    labels = one_hot_labels(train_labels)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(2048, activation=tf.nn.relu),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    model.fit(train_features, labels, epochs=5)

    return model

def nn_test(model, test_features, test_labels):
    predictions = model.predict(test_features)
    labels = one_hot_labels(test_labels)
    correct = [i for i in range(len(predictions)) if np.argmax(predictions[i]) == np.argmax(labels[i])]

    print("Accuracy: {}\n".format(len(correct) / len(predictions) * 100))
    return len(correct) / len(predictions) * 100



if __name__ == "__main__":


    features, labels = [], []
    test_features, test_labels = [], []
    for p in list(NAME_ID_MAP.keys())[:50]:


        print("Extracting features for {}\n".format(p))
        f, l = extract_pokemon_features(p, "bing")

        features.append(f[5:, :])
        labels.append(l[5:])

        #t_f, t_l = extract_pokemon_features(p, "sprites")
        test_features.append(f[:5, :])
        test_labels.append(l[:5])

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    # print("Training NN classifier")
    #
    # nn_model = train_nn_classifier(features, labels)
    # accuracy = nn_test(nn_model, test_features, test_labels)

    print("Training SVM classifier")

    svm = train_svm_classifier(features, labels)
    accuracy = svm_test(svm, test_features, test_labels)

    print("END\n")
