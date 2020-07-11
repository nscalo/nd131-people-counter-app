import argparse
import cv2
from openvino.inference_engine import IENetwork, IECore
import time
import os
import sys
from inference import Network
from concurrent.futures import ProcessPoolExecutor

# INPUT_STREAM = "test_video.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

# singleton 
def get_network(args):
    model=args.m
    model_structure = model
    model_weights = os.path.splitext(model)[0] + ".bin"
    
    start = time.time()

    core = IECore()
    model = IENetwork(model=model_structure, weights=model_weights)
    net = core.load_network(model, "CPU")
    
    return net

def preprocess(model, frame, net_input_shape):
    # Get the name of the input node
    input_name=next(iter(model.inputs))

    # Reading and Preprocessing Image
    frame=cv2.resize(frame, (net_input_shape[3], net_input_shape[2]), interpolation = cv2.INTER_AREA)
    frame=np.moveaxis(frame, -1, 0)

    # Running Inference in a loop on the same image
    input_dict={input_name:frame}
    
    return frame, input_dict

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    cpu_extension_desc = """
    MKLDNN (CPU)-targeted custom layers.
    Absolute path to a shared library with the
    kernels impl.
    """
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"
    out_desc = "the output path of image frame"
    batch_size_desc = """
    batch_size for inference
    """

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default="")
    parser.add_argument("-l", "--cpu_extension", help=cpu_extension_desc, required=False, type=str,
            default=None)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='BLUE')
    optional.add_argument("-ct", help=ct_desc, default=0.5)
    parser.add_argument('-xp', required=False, type=str)
    parser.add_argument("-bt", "--batch_size", help=batch_size_desc, type=int, default=16)
    optional.add_argument("-th", help=ct_desc, default=2, type=int)
    optional.add_argument("-co", "--coords", help=ct_desc, default="", type=str)
    optional.add_argument("--output_path", help=out_desc, 
    default="person-detector-ssd-amplitude.png", type=str)
    args = parser.parse_args()

    return args

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


def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    identified = False
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.ct:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, args.th)
            identified = True
    if identified:
        cv2.putText(frame, "Confidence Level (person): " + args.ct.__str__(), (40,35), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    return frame


def infer_on_video(args):
    # Convert the args for color and confidence
    args.c = convert_color(args.c)
    args.ct = float(args.ct)

    ### TODO: Initialize the Inference Engine
    # Initialise the class
    plugin = Network()

    CPU_EXTENSION = args.l

    def exec_f(l):
        pass

    plugin.load_core(args.m, args.d, cpu_extension=CPU_EXTENSION, args=args)

    if "MYRIAD" in args.d:
        plugin.feed_custom_layers(args, {'xml_path': args.xp}, exec_f)

    if "CPU" in args.d:
        plugin.feed_custom_parameters(args, exec_f)

    plugin.load_model(args.m, args.d, cpu_extension=CPU_EXTENSION, args=args)

    ### TODO: Load the network model into the IE
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    def split_frames(frame, c):
        frames = []
        for b in c:
            frames.append(frame[b[1]:b[3],b[0]:b[2]])
        return frames

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    args.coords = args.coords.split(":")
    c = []
    for coord in args.coords:
        co = coord[1:len(coord)-1]
        c.append(tuple(map(lambda x: int(x), co.split(","))))

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        frames = split_frames(frame, c)
        text = [(18,18), (88,18), (158,18), (228,18), (18,178), (88,98), (163,178)]

        for ii,frm in enumerate(frames):

            ### TODO: Pre-process the frame
            p_frame = cv2.resize(frm, (net_input_shape[3], net_input_shape[2]))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(1, *p_frame.shape)

            ### TODO: Perform inference on the frame
            plugin.async_inference(p_frame)

            ### TODO: Get the output of inference
            if plugin.wait() == 0:
                print(plugin.exec_network.requests[0].outputs.keys())
                result = plugin.extract_output()
                cv2.putText(frame, str(result), text[ii], 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)
                cv2.imwrite("frame2.png", frame)
                ### TODO: Update the frame to include detected bounding boxes
                # frame = draw_boxes(frame, result, args, width, height)
                # Write out the frame
                # cv2.imwrite(args.output_path, frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    # out.release()
    cap.release()
    cv2.destroyAllWindows()


def main(args):
    infer_on_video(args)


if __name__=='__main__':
    args = get_args()

    main(args)
