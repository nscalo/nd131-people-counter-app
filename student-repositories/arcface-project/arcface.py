import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
import os
import archs_face
from metrics_face import *

def obtain_arcface_model(args, arch=None, n_classes=1000):
    if not arch:
        arch = args.arch
    arcface_model = archs_face.__dict__[arch](args, n_classes)
    arcface_model.load_weights(os.path.join(args.folder, 'model_sm.hdf5'))
    # arcface_model = Model(inputs=arcface_model.input[0], outputs=arcface_model.layers[-3].output)

    return arcface_model