import argparse
import cv2
from inference import Network
from time import time
import paho.mqtt.client as mqtt
import socket
import json
import numpy as np
import math
import sys
from threading import Thread
from pipeline.Person import Person
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from complex_analysis import compute_camera_feed_output
import imageio
from PIL import Image, ImageDraw
from copy import copy

INPUT_STREAM = "test_video.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"
    bt_desc = "The batch size description"
    ot_desc = "The output video description"
    input_desc = "The input desc"
    fps_desc = "The FPS desc"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("--input", help=input_desc, default="VIDEO")
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='BLUE')
    optional.add_argument("-ct", help=ct_desc, default=0.5)
    optional.add_argument("-th", help=ct_desc, default=2, type=int)
    optional.add_argument("--batch_size", help=bt_desc, default=1, type=int)
    optional.add_argument("--data_hiding", help=bt_desc, default=1, type=int)
    optional.add_argument("--output_video", help=ot_desc, default="", type=str)
    optional.add_argument("--fps", help=fps_desc, default=25, type=float)
    optional.add_argument("--threads", help=fps_desc, default=2, type=int)
    args = parser.parse_args()

    return args

def connect_mqtt(args):
    ### TODO: Connect to the MQTT client ###
    # MQTT server environment variables
#     HOSTNAME = socket.gethostname()
#     IPADDRESS = socket.gethostbyname(HOSTNAME)
    IPADDRESS = "localhost"
    MQTT_HOST = IPADDRESS
    MQTT_PORT = 3001
    MQTT_KEEPALIVE_INTERVAL = 60

    client = mqtt.Client(client_id="people_counter")
    client.reinitialise(client_id="people_counter", clean_session=True, userdata=None)
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']


def capture_properties_set(cap):
    # writing optical flow image derivatives
    cap.set(cv2.CAP_PROP_CONVERT_RGB, True)
    cap.set(cv2.CAP_PROP_FRAME_COUNT, 90)

def delegate_models(plugin, pipeline, args):
    plugin.load_core(args.m, args.d, CPU_EXTENSION)
    ### TODO: Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION, args=args)
    # code for check layers
    plugin.check_layers(args)

def process_models(plugin, pipeline, args, frame):
    p_frame = pipeline.preprocess_frame(frame, plugin.get_input_shape())
    plugin.async_inference(p_frame)
    
    return p_frame

def execute_models(total, frame, args, start_time, 
counter, frame_numbers, 
width, height, 
base_model=None, base_pipeline=None, idx=None):
    frame_1_flag, frame_2_flag, frame_3_flag, \
    frame_4_flag, frame_5_flag = \
        [False, False, False, False, False]
    ### TODO: Get the output of inference
    if idx == 0 and base_model.wait() == 0:
        result = base_model.extract_output()
        end_time = time()
        # Write out the frame
        boxes, confidences = base_pipeline.preprocess_outputs(frame, result, args,
        confidence_level=args.ct, t=(end_time - start_time))
        people_count = 1 if len(confidences) > 0 else 0
        fps = args.fps
        ratio_sum = 0.5 # 1/2 + 1/3 + 1/5
        # resetting the frames
        if frame_1_flag is False and (counter > int((0 + frame_numbers[0])/3)):
            total += people_count
            frame_1_flag = True
        elif frame_2_flag is False and (counter > int((frame_numbers[0] + frame_numbers[1])/3)):
            total += people_count
            frame_2_flag = True
        elif frame_3_flag is False and (counter > int((frame_numbers[1] + frame_numbers[2])/3)):
            total += people_count
            frame_3_flag = True
        elif frame_4_flag is False and (counter > int((frame_numbers[2] + frame_numbers[3])/3)):
            total += people_count
            frame_4_flag = True
        elif frame_5_flag is False and (counter > int((frame_numbers[3] + frame_numbers[4])/3)):
            total += people_count
            frame_5_flag = True
        # fps and sum of ratios
        person_duration = total * 1 / fps * 1 / ratio_sum

    elif idx == 1:
        total, frame, people_count, person_duration, people_count, _ = \
            compute_camera_feed_output(frame, 
            json.load(open('plugin_config/camera.json', 'r')))
        end_time = time()
        y_pixel = 160
        out_text = "Inference Time: " + str((end_time - start_time)*10)
        frame = np.sqrt(np.square(total) + np.square(frame))
        frame = (frame/frame.max()*255).astype(np.uint8)
        cv2.putText(frame, out_text, (30, y_pixel),
        cv2.FONT_HERSHEY_COMPLEX, 0.33, (255, 255, 255), 2)
        boxes = None

    return total, frame, people_count, person_duration, people_count, boxes


def infer_on_video(args, client=None):
    # Convert the args for color and confidence
    args.c = convert_color(args.c)
    args.ct = float(args.ct)

    base_model = Network()
    base_pipeline = Person()
    base_thread = Thread(target=delegate_models, 
    args=(base_model, base_pipeline, args))
    threads = [base_thread]

    for i in range(len(threads)):
        threads[i].start()

    for i in range(len(threads)):
        threads[i].join()

    # Get and open video capture
    if args.input == "VIDEO":
        cap = cv2.VideoCapture(args.i)
        cap.open(args.i)
    elif args.input == 'CAM':
        args.i = 0
        cap = cv2.VideoCapture(0)
    # Image as input
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag = True
        cap = cv2.VideoCapture(args.i)
        cap.open(args.i)


    capture_properties_set(cap)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    frame_numbers = [ 617, 1376, 2143, 2650, 3667, 4138 ]
    frame_numbers_copy = copy(frame_numbers)
    total = 0
    counter = 0
    images = []

    # Create a video writer for the output video
    # on Mac, and `0x00000021` on Linux
    if args.output_video:
        out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'MJPG'), 
        23, (width,height))
        
    run_flag = False
    index = 0
    frame_count = 120.0
    
    frame_numbers_copy = list(reversed(frame_numbers_copy))
    fr_ppl = frame_numbers_copy.pop()
    people_number = 0
    
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        if counter % 2 == 0:
            counter += 1
            continue
            
        start_time = time()

        p_frame = process_models(base_model, base_pipeline, args, frame)

        results = []
        stats = {'people_count': 0, 'person_duration': 0.0}
        new_total = None
        flow_data = None
        zero_frame = None
        coords = {}
        with ThreadPoolExecutor(max_workers=args.threads) as process_executor:
            for ii, res in zip(list(range(args.threads)),
                process_executor.map(execute_models, 
                [total]*args.threads, [frame]*args.threads, 
                [args]*args.threads, [start_time]*args.threads, 
                [counter]*args.threads, [frame_numbers]*args.threads, 
                [width]*args.threads, [height]*args.threads, 
                [base_model, None], [base_pipeline, None], list(range(args.threads)))):
                    _total, m_frame, p_frame, person_duration, people_count, boxes = res
                    if ii == 0:
                        new_total = _total
                        zero_frame = m_frame
                        coords[0] = boxes
                        if people_count is not None:
                            stats['people_count'] = people_count
                        if person_duration is not None:
                            stats['person_duration'] = person_duration * frame_count
                    elif ii == 1:
                        flow_data = m_frame
        counter += 1
        
        if args.threads == 2:
            boxes = coords[0]
            if len(boxes) > 0:
                boxes = boxes[0]
                cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), args.c, args.th)
            flow_data = Image.fromarray(flow_data)
            flow_data = flow_data.resize((width,height))

        # fps and sum of ratios
        if args.threads == 2:
            draw = ImageDraw.Draw(flow_data)
            draw.text((40,40), text="People Count: " + str(new_total),
            fill=255)
            draw.text((40,80), text="People Duration: " + str(stats['person_duration']),
            fill=255)
            draw.text((40,120), text="Current People Count: " + str(stats['people_count']),
            fill=255)
        cv2.putText(zero_frame, "People Count: " + str(new_total), (40,40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(zero_frame, "People Duration: " + str(stats['person_duration']), (40,80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(zero_frame, "Current People Count: " + str(stats['people_count']), (40,120),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1, cv2.LINE_AA)

        if args.threads == 2:
            flow_data = np.asarray(flow_data)
            flow_data = flow_data / flow_data.max()
            flow_data = np.power(np.cosh(np.cosh(np.cosh(np.cosh(np.expand_dims(flow_data,2))))),2.0)
            flow_data = (flow_data/flow_data.max()*255)
            flow_data = np.dstack([flow_data]*3).astype(np.uint8)
        elif args.threads == 1:
            flow_data = zero_frame

        if args.output_video:
            out.write(flow_data)
        else:
            sys.stdout.buffer.write(flow_data.astype(np.uint8))
            sys.stdout.flush()
            
        if counter < fr_ppl and counter > frame_numbers_copy[len(frame_numbers_copy)-1]:
            people_number += 1
            fr_ppl = frame_numbers_copy.pop()

        data = {"count": 0, "total": new_total}
        
        if stats['people_count'] != 0:
            data["total"] = new_total
            data["count"] = stats['people_count']
            
        client.publish("person", json.dumps(data))
        
        if stats['person_duration'] == 0:
            client.publish("person/duration", json.dumps({ "duration": 0.0 }))
        else:
            client.publish("person/duration", json.dumps({ "duration": stats['person_duration'] }))
        
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    if args.output_video:
        out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    client = None
    if not args.output_video:
        client = connect_mqtt(args)
    infer_on_video(args, client)


if __name__ == "__main__":
    main()
