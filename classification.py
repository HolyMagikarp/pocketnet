import keras
import numpy as np
import os
import re
import json
import tensorflow as tf
from tensorflow.python.platform import gfile
from keras.applications import vgg16, inception_v3, resnet50, mobilenet

#inception_model = inception_v3.InceptionV3(weights='imagenet')


MODEL_DIR = 'graphs'
IMAGE_DIR = 'dataset/preprocessed/'

NAME_ID_MAP = json.load(open('./names.json', 'r'))

def load_graph():
    with gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(pokemon):
    label = NAME_ID_MAP[pokemon]
    data_dir = IMAGE_DIR + pokemon + '/'
    list_images = [data_dir + f for f in os.listdir(data_dir) if re.search('jpg|JPG|png|PNG', f)]

    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    labels = []

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

            labels.append(label)

    print('Progress: 100.00%')
    return features, labels


if __name__ == "__main__":

    features, labels = [], []
    for p in ["pikachu", "bulbasaur"]:
        f, l = extract_features(p)
        features.append(f)
        labels.append(l)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(1)

