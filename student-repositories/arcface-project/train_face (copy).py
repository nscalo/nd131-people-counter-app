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
from glob import glob

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
    parser.add_argument('--num_images', default=0, type=int,
                        help='dimention of embedded features')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
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

    image_paths = [img_path for i in range(nrof_classes) for img_path in glob(os.path.join(path_exp, classes[i], "*.jpg"))]
    image_paths = np.array(image_paths).flatten()
    np.random.shuffle(image_paths)
    image_paths = image_paths[:args.num_images]
    real_classes = np.array(list(map(lambda a: a.split("/")[2], image_paths.tolist())))

    train_set, test_set, class_indices = facenet.split_dataset(image_paths,
    args.validation_set_split_ratio, args.min_nrof_val_images_per_class, 'SPLIT_CLASSES')

    split = int(round(len(real_classes)*(1-args.validation_set_split_ratio)))

    def path_to_tensor(img_path):
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        return np.expand_dims(np.asarray(img),axis=0)

    def paths_to_tensor(img_paths):
        list_of_tensors = list(map(lambda x: path_to_tensor(x), img_paths))
        return np.vstack(list_of_tensors)

    # train_filenames = list(map(lambda x: x.split("/")[0], train_data_gen.filenames))
    # test_filenames = list(map(lambda x: x.split("/")[0], test_data_gen.filenames))
    y_train = real_classes[class_indices[0:split].tolist()]
    y_test = real_classes[class_indices[split:-1].tolist()]

    y_values = pd.get_dummies(y_train).values
    y_test_values = pd.get_dummies(y_test).values

    if args.optimizer == 'SGD':
        optimizer = SGD(lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = Adam(lr=args.lr)

    model = archs_face.__dict__[args.arch](args, nrof_classes)
    model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
    model.summary()

    callbacks = [
        ModelCheckpoint(os.path.join('models', args.name, 'model.hdf5'),
            verbose=1, save_best_only=False, period=1),
        CSVLogger(os.path.join('models', args.name, 'log.csv')),
        TerminateOnNaN()]

    if args.scheduler == 'CosineAnnealing':
        callbacks.append(CosineAnnealingScheduler(T_max=args.n_epochs, eta_max=args.lr, eta_min=args.min_lr, verbose=1))

    if 'face' in args.arch:
        print("Training started")
        # callbacks.append(LambdaCallback(on_batch_end=lambda batch, logs: print('W has nan value!!') if np.sum(np.isnan(model.layers[-4].get_weights()[0])) > 0 else 0))
        for epoch in range(args.epochs):
            for idx in range(0,len(y_train),args.batch_size):
                X_train = paths_to_tensor(train_set[idx:idx+args.batch_size])
                X_test = paths_to_tensor(test_set[idx:idx+args.batch_size])
                model.fit([X_train,y_values[idx:idx+args.batch_size]], y_values[idx:idx+args.batch_size],
                    epochs=args.n_epochs,
                    steps_per_epoch=args.steps_per_epoch,
                    workers=args.workers,
                    validation_data=None,
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
