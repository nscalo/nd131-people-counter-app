#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv
import cv2

from tests_common import NewOpenCVTests
from dotenv import load_dotenv
from keras.models import Sequential, load_model, Model

from metrics_face import ArcFace
import object_detection
import sys
import argparse
import score
import time
import logging

logging.basicConfig(filename="logs/keras/main_lognorm.log", filemode='w', level=logging.INFO)

load_dotenv(".env")

def parse_args():
    parser = argparse.ArgumentParser("Parse args for videoio inference for keras")
    parser.add_argument("--input_weights", required=True, default="", type=str)
    parser.add_argument("--batch_size", required=True, default=64, type=int)
    parser.add_argument("--input_file", required=False, default="", type=str)
    parser.add_argument("--name", required=True, default="", type=str)
    parser.add_argument("--method", required=True, default="", type=str)
    parser.add_argument("--callback", required=True, default="", type=str)
    parser.add_argument("--face_detector_weights", required=False, default="", type=str)
    parser.add_argument("--face_detector_model", required=False, default="", type=str)

    return parser.parse_args()

class Bindings(NewOpenCVTests):

    def check_name(self, name):
        self.assertFalse(name == None)
        self.assertFalse(name == "")

    def test_registry(self):
        self.check_name(cv.videoio_registry.getBackendName(cv.CAP_ANY))
        self.check_name(cv.videoio_registry.getBackendName(cv.CAP_FFMPEG))
        self.check_name(cv.videoio_registry.getBackendName(cv.CAP_OPENCV_MJPEG))
        backends = cv.videoio_registry.getBackends()
        for backend in backends:
            self.check_name(cv.videoio_registry.getBackendName(backend))

    def build_data_sess(self, in_blob_name="data", out_blob_name="softmaxout", need_reshape=False):
        arcface_model = load_model(self.args.input_weights, custom_objects={'ArcFace': ArcFace})
        if self.args.method == "pnorm":
            self.arcface_model = Model(inputs=arcface_model.input[0], 
            outputs=arcface_model.layers[self.config_dict['ARCFACE_POOLING_LAYER_INDEX']].output)
        elif self.args.method == "lognorm":
            self.arcface_model = Model(inputs=arcface_model.input[0], 
        outputs=arcface_model.layers[self.config_dict['ARCFACE_PREBATCHNORM_LAYER_INDEX']].output)

    def face_detector(self, images, image_ids, batch_size):
        return object_detection.main(self.param_dict, images)

    def measure(self, features, risk_difference=0.05, significant=1, to_significant=5):
        measure_scores = []
        for ii in range(0,len(features),2):
            risk_vector1 = score.process_outputs(features[ii], significant, to_significant)
            risk_vector2 = score.process_outputs(features[ii+1], significant, to_significant)
            measure_scores.append(
                score.face_recognize_risk(risk_difference, risk_vector1, risk_vector2)
            )
        return measure_scores

    def preprocess_conventional_box_images(self, images, bbox, image_source):
        img = np.zeros((len(bbox),160,160,3))
        for ii,box in enumerate(bbox):
            left, top, width, height = box
            if width > 40 and height > 40:
                i = images[top:top+height,left:left+width]
                i = cv2.resize(i, (160,160))
                img[ii] = i
        return img

    def preprocess_grayscale_box_images(self, images, bbox, image_source):
        img = np.zeros((len(bbox),160,160,1))
        for ii,box in enumerate(bbox):
            left, top, width, height = box
            if width > 40 and height > 40:
                i = images[top:top+height,left:left+width]
                i = cv2.resize(i, (160,160))
                img[ii] = np.expand_dims(i,2)
        return img
    
    def initialize(self, args):
        self.args = args
        self.config_dict = dict()
        self.config_dict['ARCFACE_PREBATCHNORM_LAYER_INDEX']=-3
        self.config_dict['ARCFACE_POOLING_LAYER_INDEX']=-4
        self.param_dict = {
            "framework": "tensorflow",
            "thr": 0.4,
            "model": self.args.face_detector_weights,
            "backend": 3,
            "async": self.args.batch_size,
            "target": 0,
            "classes": "",
            "nms": 0.65,
            "motion_tracker": False,
            "inputs": "",
            "input": None,
            "scale": 1.0,
            "mean": 1.0,
            "rgb": False,
            "config": self.args.face_detector_model,
        }
    
    def test_video(self, args, input_file):
        
        self.initialize(args)
        
        self.build_data_sess(need_reshape=True)
        cap = cv.VideoCapture(input_file)

        counter = 0
        measures = []
        frames = []
        input_shape = (160,160)
        
        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            if counter == 0:
                counter += 1
                continue

            frames.append(frame)

            if counter % args.batch_size == 0:

                start_time = time.time()
                b, co, cl = [], [], []

                for frame in frames:
                    boxes, confidences, classIds = self.face_detector(frame, None, self.args.batch_size)
                    b.append(boxes)
                    co.append(confidences)
                    cl.append(classIds)

                images_sync_list = []
                for ii, boxes in enumerate(b):
                    frame = frames[ii]
                    if len(boxes) > 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if args.method == "pnorm":
                            images_sync = self.preprocess_conventional_box_images(frame, boxes, None)
                        elif args.method == "lognorm":
                            images_sync = self.preprocess_grayscale_box_images(frame, boxes, None)
                        if images_sync is not None:
                            images_sync_list.append(images_sync)
                
                if len(images_sync_list) > 0:
                    images_sync_list = np.vstack(images_sync_list)
                    end_time = time.time()
                    logging.info("""Frame preprocessing time: {t}""".format(t=(end_time - start_time)))

                    start_time = time.time()
                    
                    features = self.arcface_model.predict(images_sync_list, verbose=1)

                    end_time = time.time()
                    logging.info("""Frame inference time: {t}, {p} per face-image""".format(t=(end_time - start_time), 
                    p=(end_time - start_time)/len(images_sync_list)))

                    start_time = time.time()
                    measures = self.measure(features)
                    end_time = time.time()

                    logging.info("""Frame post-process time: {t}, {p} per face-image""".format(t=(end_time - start_time), 
                    p=(end_time - start_time)/len(images_sync_list)))

                frames = []
                
            counter += 1
        
        logging.info(str(measures))

    def test_lognorm_model(self, args):
        self.initialize(args)
        self.build_data_sess(need_reshape=True)
        img_path = "dnn/vgg_face.jpg"

        frame = cv.imread(img_path).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        images_sync = cv2.resize(frame, (160,160))

        images_sync = np.expand_dims(images_sync,2)
        images_sync = np.expand_dims(images_sync,0)
        images_sync = np.vstack([images_sync]*100)

        # boxes, confidences, classIds = self.face_detector(frame, None, self.args.batch_size)
        # if args.method == "pnorm":
        #     images_sync = self.preprocess_conventional_box_images(frame, boxes, None)
        # elif args.method == "lognorm":
        #     images_sync = self.preprocess_grayscale_box_images(frame, boxes, None)
        #     print(images_sync.shape)

        # images_sync = np.expand_dims(images_sync, 0)
        features = self.arcface_model.predict(images_sync, verbose=1)
        risk_difference=1e-3
        m = self.measure(features, risk_difference=risk_difference, 
        significant=1, to_significant=10)
        m1 = np.mean(m)
        print("base score: ", str(m1))
        diffs = []
        for i in range(20):
            images_sync2 = np.clip(images_sync + np.random.randint(-180,180,images_sync.shape),0,255).astype(np.uint8)
            features = self.arcface_model.predict(images_sync2, verbose=1)
            m = self.measure(features, risk_difference=risk_difference, 
            significant=1, to_significant=10)
            m2 = np.mean(m)
            try:
                print("score. " + str(m2))
                print("diff score. " + str(abs(m2 - m1)))
                diffs.append(abs(m2 - m1))
            except Exception as e:
                print(e.args)

        print(np.mean(diffs))

if __name__ == '__main__':
    args = parse_args()
    # Bindings.bootstrap()
    test = Bindings()
    test.__getattribute__(args.callback)(args)