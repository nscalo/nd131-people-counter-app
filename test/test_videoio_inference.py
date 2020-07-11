#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests
from dotenv import load_dotenv

import sys
import argparse
import score
import time
import caffe
import logging

logging.basicConfig(filename="logs/softmax/main_log.log", filemode='w', level=logging.INFO)

load_dotenv(".env")

def parse_args():
    parser = argparse.ArgumentParser("Parse args for videoio inference for caffe")
    parser.add_argument("--input_graph", required=True, default="", type=str)
    parser.add_argument("--input_weights", required=True, default="", type=str)
    parser.add_argument("--batch_size", required=True, default=64, type=int)
    parser.add_argument("--input_file", required=True, default="", type=str)
    parser.add_argument("--name", required=True, default="", type=str)
    parser.add_argument("--callback", required=True, default="", type=str)
    parser.add_argument("--is_softmax", required=False, default=False, type=bool)
    parser.add_argument("-thr", "--threshold", help="", type=float, default=0.4)
    parser.add_argument("-ic", "--is_conf", help="", type=bool, default=False)
    parser.add_argument("-it", "--is_threshold", help="", type=bool, default=False)

    return parser.parse_args()

def preprocessing(frame, net_input_shape):
    p_frame = cv.resize(frame, (net_input_shape[0], net_input_shape[1]))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)

    return p_frame

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
        caffe.set_mode_cpu()
        self.model = str(self.args.input_graph)
        self.weights = str(self.args.input_weights)

        self.network = caffe.Net(self.model, self.weights, caffe.TEST)
        self.in_blob_name = in_blob_name
        self.out_blob_name = out_blob_name
        self.need_reshape = need_reshape

    def build_sess_data_opencv(self, in_blob_name="data", out_blob_name="detection_out", need_reshape=False):
        caffe.set_mode_cpu()
        self.model = str(self.args.input_graph)
        self.weights = str(self.args.input_weights)

        self.network = cv.dnn_DetectionModel(self.weights, self.model)
        self.in_blob_name = in_blob_name
        self.out_blob_name = out_blob_name
        self.need_reshape = need_reshape

    def preprocess_bounding_box_ssd(self, images, result, confidence_level=0.48):
        boxes = []
        confs = []
        height, width, _ = images[0].shape
        for box in result[0][0]: # Output shape is 1x1x100x7
            conf = box[2]
            if conf >= confidence_level:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                boxes.append([xmin, ymin, xmax, ymax])
                confs.append(conf)
        return boxes, confs

    def get_output(self, input_blob):
        if self.need_reshape:
            self.network.blobs[self.in_blob_name].reshape(*input_blob.shape)
        
        return self.network.forward_all(**{self.in_blob_name: input_blob})[self.out_blob_name]

    def get_output_opencv(self, input_blob):
        if self.need_reshape:
            self.network.blobs[self.in_blob_name].reshape(*input_blob.shape)
        
        self.network.setInput(input_blob.astype(np.int8), self.in_blob_name)
        return self.network.forward()

    def calculate_threshold(self, frame, result, args):
        r = (result.flatten() > args.threshold).astype(bool)
        return np.sum(r)
    
    def test_video(self, args, input_file):
        self.args = args
        if args.is_softmax:
            self.build_data_sess(need_reshape=True)
            weights = np.arange(1000)
            alpha = 1e3
        else:
            self.build_sess_data_opencv(need_reshape=False)
        cap = cv.VideoCapture(input_file)

        counter = 0
        frames = []
        scores = []
        input_shape = (224,224)
        ix = []
        
        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            if counter == 0:
                counter += 1
                continue

            frames.append(frame)

            if (counter) % args.batch_size == 0:

                start_time = time.time()
                p_frames = []
                scores = []

                for frame in frames:
                    p_frame = preprocessing(frame, input_shape)
                    p_frames.append(p_frame)
                
                end_time = time.time()
                logging.info("""Frame preprocessing time: {t}""".format(t=(end_time - start_time)))

                start_time = time.time()
                
                p_frames = np.vstack(p_frames)
                if args.is_softmax:
                    output = self.get_output(p_frames)
                else:
                    output = self.get_output_opencv(p_frames)

                end_time = time.time()
                logging.info("""Frame inference time: {t}""".format(t=(end_time - start_time)))

                start_time = time.time()
                if args.is_softmax:
                    # s = score.lognorm(weights, output[0,:,0,0], alpha)
                    for ii,frame in enumerate(frames):
                        s = self.calculate_threshold(frame, output[ii,:,0,0], args)
                        scores.append(s)
                else:
                    boxes, confs = self.preprocess_bounding_box_ssd(frames, output)
                    if len(confs) > 0:
                        s = np.mean(confs), len(confs) / len(output[0][0])
                        scores.append(s)
                end_time = time.time()

                logging.info("""Frame post-process time: {t}""".format(t=(end_time - start_time)))
                
                frames = []

                im = np.mean(scores)
                ix.append(im)
                logging.info(im)

            counter += 1
    
        print("scores: ", ix, "scores mean: ", np.mean(ix))
        

if __name__ == '__main__':
    args = parse_args()
    # Bindings.bootstrap()
    test = Bindings()
    test.__getattribute__(args.callback)(args, args.input_file)