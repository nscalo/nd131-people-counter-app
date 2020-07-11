"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import numpy as np
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from concurrent.futures import ThreadPoolExecutor

import logging

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """

    model_desc = """
    Path to an xml file with a trained model.
    """
    video_desc = """
    Path to image or video file
    """
    cpu_extension_desc = """
    MKLDNN (CPU)-targeted custom layers.
    Absolute path to a shared library with the
    kernels impl.
    """
    device_desc = """
    Specify the target device to infer on: 
    CPU, GPU, FPGA or MYRIAD is acceptable. Sample 
    will look for a suitable plugin for device 
    specified (CPU by default)
    """
    prob_threshold_desc = """
    Probability threshold for detections filtering
    (0.5 by default)
    """
    color_desc = """
    color for painting
    """
    thickness_desc = """
    thickness for painting
    """
    batch_size_desc = """
    batch_size for inference
    """
    mode_desc = """
    mode for async, sync
    """

    parser = ArgumentParser()
    parser.add_argument("-m", help=model_desc, required=True, type=str)
    parser.add_argument("-i", help=video_desc, required=True, type=str)
    parser.add_argument("-l", help=cpu_extension_desc, required=False, type=str,
            default=None)
    parser.add_argument("-d", help=device_desc, type=str, default="CPU")
    parser.add_argument('-xp', required=False, type=str)
    parser.add_argument("-pt", help=prob_threshold_desc, type=float, default=0.5)
    parser.add_argument("-c", help=color_desc, type=tuple, default=(0,255,0))
    parser.add_argument("-th", help=thickness_desc, type=int, default=2)
    parser.add_argument("-bt", "--batch_size", help=batch_size_desc, type=int, default=16)
    parser.add_argument("-mode", "--mode", help=mode_desc, type=str, default='async')
    parser.add_argument("-out", "--output_log", help=mode_desc, type=str, default='main.log')
    # parser.add_argument("-o", "--output_video", help=mode_desc, type=str, default="output_video.avi")
    parser.add_argument("-thr", "--threshold", help=mode_desc, type=float, default=0.4)
    parser.add_argument("-ic", "--is_conf", help="", type=bool, default=False)
    parser.add_argument("-asp", "--aspect", help="", type=bool, default=False)
    parser.add_argument("-it", "--is_threshold", help="", type=bool, default=False)
    parser.add_argument("-gs", "--grayscale", help="", type=bool, default=False)
    
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client(client_id="people_counter")
    client.reinitialise(client_id="people_counter", clean_session=True, userdata=None)
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def preprocessing(frame, net_input_shape, grayscale=False):
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    if grayscale:
        p_frame = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
        p_frame = np.expand_dims(p_frame, 2)
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)

    return p_frame

def calculate_threshold(frame, result, args, width, height):
    r = (result.flatten() > args.threshold).astype(bool)
    return np.sum(r)

def draw_boxes(frame, result, args, width, height):
    confs = []
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.pt:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            confs.append(conf)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, args.th)
    
    return frame, confs

def infer_on_batch_result(infer_network, frames, args, width, height, request_id=0, 
conf=False, threshold=False, aspect=False):
    ### TODO: Get the results of the inference request ###
    results = infer_network.extract_output(request_id=request_id)
    output_shape = 200
    thrs = []
    confidences = []

    if aspect:
        return [], [], []

    ### TODO: Update the frame to include detected bounding boxes
    for ii, frame in enumerate(frames):
        if threshold:
            thr = calculate_threshold(frame, results[ii,:,:,:], args, width, height)
            thrs.append(thr)
        else:
            frame, confs = draw_boxes(frame, results[ii,:,:,:], args, width, height)
        frames[ii] = frame
        confidences = np.append(confidences, confs).tolist()
    
    # Write out the frame
    
    return frames, thrs, confidences

def infer_on_multi_result(infer_network, frames, args, width, height, request_id=0, 
conf=False, threshold=False, aspect=False):
    ### TODO: Get the results of the inference request ###
    results = infer_network.extract_output(request_id=request_id)
    output_shape = 200
    thrs = []
    confidences = []

    if aspect:
        return [], [], []

    ### TODO: Update the frame to include detected bounding boxes
    for ii, frame in enumerate(frames):
        if threshold:
            thr = calculate_threshold(frame, results[:,:,output_shape*ii:output_shape*ii+output_shape,:], 
            args, width, height)
            thrs.append(thr)
        else:
            frame, confs = draw_boxes(frame, results[:,:,output_shape*ii:output_shape*ii+output_shape,:], 
        args, width, height)
        frames[ii] = frame
        confidences = np.append(confidences, confs).tolist()
    
    # Write out the frame
    
    return frames, thrs, confidences

def infer_on_result(infer_network, frame, args, width, height, request_id=0, 
conf=False, threshold=False, aspect=False):
    ### TODO: Get the results of the inference request ###
    results = infer_network.extract_output(request_id=request_id)
    output_shape = 200
    thrs = []
    confidences = []

    if aspect:
        return [], [], []

    ### TODO: Update the frame to include detected bounding boxes
    if threshold:
        thr = calculate_threshold(frame, results[:,:,:,:], 
        args, width, height)
        thrs.append(thr)
    else:
        frame, confs = draw_boxes(frame, results, args, width, height)
        confidences = np.append(confidences, confs).tolist()
    
    # Write out the frame
    
    return frame, thrs, confidences

    ### TODO: Extract any desired stats from the results ###
    ### TODO: Calculate and send relevant information on ###
    ### current_count, total_count and duration to the MQTT server ###
    ### Topic "person": keys of "count" and "total" ###
    ### Topic "person/duration": key of "duration" ###

def sync_async(infer_network, p_frames, t='sync', request_id=20):
    if t == "async":
        for ii, p_frame in enumerate(p_frames[0:16]):
            infer_network.async_inference(p_frame, request_id=ii+request_id)
    elif t == "sync":
        infer_network.sync_inference(p_frames[16:])

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()

    CPU_EXTENSION = args.l

    def exec_f(l):
        pass

    infer_network.load_core(args.m, args.d, cpu_extension=CPU_EXTENSION, args=args)

    if "MYRIAD" in args.d:
        infer_network.feed_custom_layers(args, {'xml_path': args.xp}, exec_f)

    if "CPU" in args.d:
        infer_network.feed_custom_parameters(args, exec_f)

    infer_network.load_model(args.m, args.d, cpu_extension=CPU_EXTENSION, args=args)
    # Set Probability threshold for detections

    args = build_argparser().parse_args()

    frames = []

    # b = args.batch_size
    # args.batch_size = 1

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.m, args.d, cpu_extension=args.l, 
    args=args)

    # args.batch_size = b

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)
    width = int(cap.get(3))
    height = int(cap.get(4))

    # out = cv2.VideoWriter(args.output_video, 
    # cv2.VideoWriter_fourcc(*'MJPG'), 30, (width,height))

    def call_infer_network(infer_network, frame, args, width, height, request_id, conf=False, threshold=False, aspect=False):
        if infer_network.wait(request_id=request_id) == 0:
            frame, thrs, confidences = infer_on_result(infer_network, frame, args, width, height, request_id, 
            conf=conf, threshold=threshold, aspect=args.aspect)
        return frame, thrs, confidences

    def call_sync_infer_network(infer_network, frames, args, width, height, request_id, conf=False, threshold=False, aspect=False):
        frames, thrs, confidences = infer_on_multi_result(infer_network, frames, args, width, height, request_id, 
        conf=conf, threshold=threshold, aspect=args.aspect)
        return frames, thrs, confidences

    def call_sync_batch_infer_network(ingfer_network, frames, args, width, height, request_id, conf=False, threshold=False, aspect=False):
        frames, thrs, confidences = infer_on_batch_result(infer_network, frames, args, width, height, request_id, 
        conf=conf, threshold=threshold, aspect=args.aspect)
        return frames, thrs, confidences

    def call_sync_async_infer_network(infer_network, frames, args, width, height, request_id, conf=False, threshold=False, aspect=False):
        for ii, frame in enumerate(frames[0:16]):
            frame, thrs1, confidences1 = call_infer_network(infer_network, frame, args, width, height, 
            request_id=ii+request_id, conf=conf, threshold=threshold, aspect=args.aspect)
            frames[ii] = frame
        frames[16:], thrs2, confidences2 = call_sync_infer_network(infer_network, frames[16:], args, width, height, 
        request_id=request_id, conf=conf, threshold=threshold, aspect=args.aspect)

        return frames, np.append(thrs1, thrs2).tolist(), np.append(confidences1, confidences2).tolist()

    counter = 0
    ix = []
    cx = []

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        
        ### TODO: Read from the video capture ###
        ret, frame = cap.read()

        if not ret:
            break

        if counter == 0:
            counter += 1
            continue

        if counter % 2 == 0:
            counter += 1
            continue

        # if ((counter % 8) == 0) or ((counter % 4) == 0) or ((counter % 2) == 0) or \
        #     ((counter % 5) == 0) or ((counter % 6) == 0) or ((counter % 10) == 0) or ((counter % 14) == 0) or ((counter % 16) == 0):
        #     counter += 1
        #     continue

        frames.append(frame)

        if (counter) % args.batch_size == 0:

            start_time = time.time()
            ### TODO: Pre-process the image as needed ###
            p_frames = []

            for frame in frames:
                p_frame = preprocessing(frame, infer_network.get_input_shape(), args.grayscale)
                p_frames.append(p_frame)
            
            end_time = time.time()
            logging.info("""Frame preprocessing time: {t}""".format(t=(end_time - start_time)))

            start_time = time.time()
            ### TODO: Start asynchronous inference for specified request ###
            # infer_network.sync_inference(p_frames)
            if args.mode == "async":
                for ii, p_frame in enumerate(p_frames):
                    infer_network.async_inference(p_frame, request_id=ii)
                end_time = time.time()
            elif args.mode == "sync" or args.mode == "sync_batch":
                infer_network.sync_inference(p_frames)
                end_time = time.time()
            elif args.mode == "sync_async":
                with ThreadPoolExecutor(max_workers=2) as executor:
                    for ii in executor.map(sync_async, [infer_network]*2, [p_frames]*2, 
                    ['sync', 'async'], [5]*2):
                        pass
                end_time = time.time()
            logging.info("""Frame inference time: {t}""".format(t=(end_time - start_time)))

            ### TODO: Wait for the result ###
            start_time = time.time()
            if args.mode == "async":
                for ii, frame in enumerate(frames):
                    frame, thrs, confidences = call_infer_network(infer_network, 
                    frame, args, width, height, ii, conf=args.is_conf, 
                    threshold=args.is_threshold, aspect=args.aspect)
                    frames[ii] = frame
            elif args.mode == "sync":
                frames, thrs, confidences = call_sync_infer_network(infer_network, frames, 
                args, width, height, 0, conf=args.is_conf, 
                threshold=args.is_threshold, aspect=args.aspect)
            elif args.mode == "sync_batch":
                frames, thrs, confidences = call_sync_batch_infer_network(infer_network, frames, 
                args, width, height, 0, conf=args.is_conf, 
                threshold=args.is_threshold, aspect=args.aspect)
            elif args.mode == "sync_async":
                frames, thrs, confidences = call_sync_async_infer_network(infer_network, frames, 
                args, width, height, 5, conf=args.is_conf, 
                threshold=args.is_threshold, aspect=args.aspect)

            end_time = time.time()
            # logging.info("""Thresholds count: {t}""".format(t=(np.mean(thrs))))
            logging.info("""Frame extract time: {t}""".format(t=(end_time - start_time)))

            start_time = time.time()
            
            # for frame in frames:
            #     out.write(frame)

            end_time = time.time()

            logging.info("""Frame paint time: {t}""".format(t=(end_time - start_time)))

            for frame in frames:
                sys.stdout.buffer.write(frame.astype(np.uint8))
                sys.stdout.flush()

            people_count = 1 if len(confidences) > 0 else 0
            person_duration = len(confidences) / len(frames) * 0.042
            
            client.publish("people", 
            json.dumps({ "people": people_count }))

            client.publish("person/duration", 
            json.dumps({ "person/duration": person_duration }))
            
            frames = []

        counter += 1
        
        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###

    # print("scores: ", ix, "scores mean: ", np.mean(ix))
    # print("Confidences: ", cx, np.mean(cx))
    
    cap.release()
    cv2.destroyAllWindows()
    
    return counter

def convert_perf_time(perf_text):

    import re
    
    result = re.split("\n", perf_text)

    order = ['preprocessing', 'inference', 'extract', 'paint']

    def extract(r):
        l = r.split(" ")
        return l[len(l)-1]

    result = list(map(lambda x: float(x), 
    list(filter(lambda x: x.strip() != "", list(map(extract, result))))))

    return [dict(zip(order, result[idx:idx+4])) for idx in range(0,len(result),4)]

def convert_perf_time_video_len(args, perf_stats, reference_video_len=None):
    per_frame_execution_time = 0.0

    for stat in perf_stats:
        per_frame_execution_time += stat['preprocessing']
        per_frame_execution_time += stat['inference']
        per_frame_execution_time += stat['extract']
        per_frame_execution_time += stat['paint']

    return per_frame_execution_time / reference_video_len

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    # logging.basicConfig(filename=args.output_log, filemode='w', level=logging.INFO)
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    counter = infer_on_stream(args, client)

    # perf_text = open(args.output_log, "r").read()

    # perf_stats = convert_perf_time(perf_text)

    # per_frame_execution_time = convert_perf_time_video_len(args, perf_stats, counter)

    # print(per_frame_execution_time)
    # print("FPS: ", 1 / per_frame_execution_time)

if __name__ == '__main__':
    main()
