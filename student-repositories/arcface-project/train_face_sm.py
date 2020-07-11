import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import joblib
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import facenet
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler, TerminateOnNaN, LambdaCallback
import archs_face
from metrics import *
from scheduler import *
from keras.preprocessing import image
from PIL import Image
import pandas as pd
import re
import cv2
from glob import glob
from concurrent.futures import ThreadPoolExecutor

arch_names = archs_face.__dict__.keys()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg8',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: vgg8)')
    parser.add_argument('--num-features', default=5, type=int,
                        help='dimention of embedded features')
    parser.add_argument('--num-embedding', default=5, type=int,
                        help='dimention of embedded features')
    parser.add_argument('--num_images', default=0, type=int,
                        help='dimention of embedded features')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--nrof_classes', default=3139, type=int,
                        help='nrof_classes')
    parser.add_argument('--test_size', default=100, type=int,
                        help='test size')
    parser.add_argument('--steps_per_epoch', default=1, type=int,
                        help='steps_per_epoch')
    parser.add_argument('--scheduler', default='CosineAnnealing',
                        choices=['CosineAnnealing', 'None'],
                        help='scheduler: ' +
                            ' | '.join(['CosineAnnealing', 'None']) +
                            ' (default: CosineAnnealing)')
    parser.add_argument('--n_epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--workers', default=16, type=int, metavar='N',
                        help='number of workers')
    parser.add_argument('--epochs', default=35, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min-lr', default=1e-3, type=float,
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--validation_set_split_ratio', type=float,
        help='The ratio of the total dataset to use for validation', default=0.0)
    parser.add_argument('--min_nrof_val_images_per_class', type=float,
        help='Classes with fewer images will be removed from the validation set', default=0)
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned face patches.',
        default='')
    args = parser.parse_args()

    return args

def preprocess_images(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def main():
    args = parse_args()

    # add model name to args
    args.name = 'mnist_%s_%dd' %(args.arch, args.num_features)

    os.makedirs('models/%s' %args.name, exist_ok=True)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    IMG_HEIGHT = 160
    IMG_WIDTH = 160

    classes, nrof_classes = facenet.get_dataset(args.data_dir)
    path_exp = os.path.expanduser(args.data_dir)
    test_exp = os.path.expanduser(args.test_data_dir)

    if args.nrof_classes:
        nrof_classes = args.nrof_classes

    image_paths = [img_path for i in range(nrof_classes) for img_path in glob(os.path.join(path_exp, classes[i], "*.jpg"))[2:9]]
    image_paths = np.array(image_paths).flatten()
    test_image_paths = [img_path for i in range(nrof_classes) for img_path in glob(os.path.join(path_exp, classes[i], "*.jpg"))[0:1]]
    test_image_paths = np.array(test_image_paths).flatten()
    image_paths = image_paths[:].tolist()
    test_image_paths = test_image_paths[:].tolist()
    real_classes = np.array(list(map(lambda a: a.split("/")[4], image_paths)))
    test_real_classes = np.array(list(map(lambda a: a.split("/")[4], test_image_paths)))

    train_set = image_paths
    test_set = test_image_paths

    def path_to_tensor(img):
        face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
        faces_multi = face_cascade.detectMultiScale(img, 1.1, 4)
        faces = [face.astype(np.int64).tolist() for face in faces_multi]
        return faces

    def resize(yes_face, img):
        return cv2.resize(img[yes_face[1]:yes_face[1]+yes_face[3],yes_face[0]:yes_face[0]+yes_face[2]],(IMG_HEIGHT,IMG_WIDTH))

    def paths_to_tensor(executor, img_paths):
        def img_to_tensor(img_path, ii):
            img = image.load_img(img_path)
            gray = np.asarray(img.convert('L'))
            img = np.asarray(img)
            faces = path_to_tensor(gray)
            if len(faces) > 0:
                img = resize(faces[0], img)
                return np.expand_dims(img,0), ii
            else:
                return False
        list_tensors = []
        list_indices = []
        for result in tqdm(executor.map(img_to_tensor, img_paths, range(len(img_paths)))):
            if result is not False:
                list_tensors.append(result[0])
                list_indices.append(result[1])
        return np.vstack(list_tensors), list_indices

    y_train = real_classes
    y_test = test_real_classes

    y_values = pd.get_dummies(y_train).values
    y_test_values = pd.get_dummies(y_test).values

    if args.optimizer == 'SGD':
        optimizer = SGD(lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = Adam(lr=args.lr)

    model = archs_face.__dict__[args.arch](args, len(np.unique(y_train)))
    model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
    model.summary()

    model.load_weights(os.path.join('models', args.name, 'model_sm.hdf5'))

    callbacks = [
        ModelCheckpoint(os.path.join('models', args.name, 'model_sm.hdf5'),
            verbose=1, save_best_only=False, period=1, monitor='val_acc'),
        CSVLogger(os.path.join('models', args.name, 'log.csv')),
        TerminateOnNaN()]

    if args.scheduler == 'CosineAnnealing':
        callbacks.append(CosineAnnealingScheduler(T_max=args.n_epochs, eta_max=args.lr, eta_min=args.min_lr, verbose=1))

    if 'face' in args.arch:
        print("Training started")
        train_executor = ThreadPoolExecutor(max_workers=2)
        X_train, train_non_tensors = paths_to_tensor(train_executor, train_set)
        test_executor = ThreadPoolExecutor(max_workers=2)
        X_test, test_non_tensors = paths_to_tensor(test_executor, test_set)
        model.fit([X_train, 
        y_values[train_non_tensors]], 
        y_values[train_non_tensors],
        epochs=args.n_epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        validation_data=([X_test, 
        y_test_values[test_non_tensors]], 
        y_test_values[test_non_tensors]),
        callbacks=callbacks, verbose=1)

            # model.load_weights(os.path.join('models', args.name, 'model.hdf5'))

    # model.load_weights(os.path.join('models', args.name, 'model.hdf5'))
    # X_test = paths_to_tensor(test_set)
    # if 'face' in args.arch:
    #     score = model.evaluate([X_test[:args.batch_size], y_test[:args.batch_size]], y_test[:args.batch_size], verbose=1)
    # else:
    #     score = model.evaluate(X_test, y_test, verbose=1)

    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])


if __name__ == '__main__':
    main()
