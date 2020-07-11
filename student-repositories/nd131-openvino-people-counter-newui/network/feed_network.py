import argparse
### TODO: Load the necessary libraries
import os
from inference import Network
from openvino.inference_engine.ie_api import IENetLayer
import numpy as np
import cv2

CPU_EXTENSION = "/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/nd131-openvino-people-counter-newui/custom_layers/arcface/cl_pnorm/user_ie_extensions/cpu/build/libpnorm_cpu_extension.so"
VPU_EXTENSION = "/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libmyriadPlugin.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"

    # -- Create the arguments
    parser.add_argument("-m", help=m_desc)
    parser.add_argument('-l', required=False, type=str)
    parser.add_argument('-xp', required=False, type=str)
    parser.add_argument('-d', required=False, type=str)
    parser.add_argument('--img', required=False, type=str)
    parser.add_argument('--batch_size', required=False, type=int, default=1)
    parser.add_argument('--factor', required=False, type=float, default=1e-2)
    args = parser.parse_args()

    return args

def preprocessing(network, img_path, args):
    input_shape = network.get_input_shape()
    img = cv2.imread(img_path)
    img = cv2.resize(img, (input_shape[3],input_shape[2]))
    img = img.transpose((2,0,1))
    img = img.reshape(1, *img.shape)

    img = img.astype(np.float64) - img.min()*args.factor
    img = img.astype(np.uint8)

    return img

def load_to_IE(args, model_xml, img_path):
    ### TODO: Load the Inference Engine API
    # plugin = IECore()

    network = Network()

    CPU_EXTENSION = args.l

    def exec_f(l):
        pass

    network.load_core(model_xml, args.d, cpu_extension=CPU_EXTENSION, args=args)

    if "MYRIAD" in args.d:
        network.feed_custom_layers(args, {'xml_path': args.xp}, exec_f)

    if "CPU" in args.d:
        network.feed_custom_parameters(args, exec_f)

    network.load_model(model_xml, args.d, cpu_extension=CPU_EXTENSION, args=args)

    img = preprocessing(network, img_path, args)
    
    network.sync_inference(img)

    print(network.extract_output().shape)

    network.check_layers(args)

    return


def main():
    args = get_args()
    load_to_IE(args, args.m, args.img)


if __name__ == "__main__":
    main()
