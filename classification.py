import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet

inception_model = inception_v3.InceptionV3(weights='imagenet')
