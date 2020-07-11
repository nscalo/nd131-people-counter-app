import torch
from torch import nn
import tensorflow as tf
import argparse
from keras import backend as K
import tensorflow as tf
from google.protobuf import text_format
import tfgraphviz as tfg
from dotenv import load_dotenv
import numpy as np

load_dotenv('.env')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb_filename", default=None, 
        help='conversion protocol buffer filename', required=True)
    parser.add_argument("--pb_filename_text", default=None, 
        help='conversion protocol buffer filename', required=True)
    parser.add_argument("--write_graph", default=0, 
        help='write graph', required=False)
    parser.add_argument("--visualize_graph", default=0, 
        help='write graph', required=False)
    parser.add_argument("--folder", default="models", 
        help='folder name', required=True)
    parser.add_argument('--arch', '-a', metavar='ARCH', default=None, required=True)
    parser.add_argument('--num-features', default=5, type=int,
        help='dimention of embedded features', required=True)

    return parser.parse_args()

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()

  import os
  file_ext = os.path.splitext(model_file)[1]

  with open(model_file, "rb") as f:
    if file_ext == '.pbtxt':
      text_format.Merge(f.read(), graph_def)
    else:
      graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def, name='')
    tf.io.write_graph(graph_def, '/tmp/', 'optimized_graph.pb',as_text=False)

def create_model(args):
    import arcface
    return arcface.obtain_arcface_model(args)

def freeze_session(session, keep_var_names=None, keep_output_names=None, 
output_names=None, clear_devices=None):

    from tensorflow.python.framework.graph_util import convert_variables_to_constants, \
        extract_sub_graph

    graph = session.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        mod_graph_def = tf.GraphDef()
        nodes = []
        remove_nodes = ['arc_face_1/arcface_acos', 'arc_face_1/arcface_cos']
        ops = []
        if clear_devices:
            for node in input_graph_def.node:
                # if node.name not in remove_nodes:
                nodes.append(node)
                node.device = ""
                ops.append(node.op)
            mod_graph_def.node.extend(nodes)
            # Delete references to deleted nodes
            for node in mod_graph_def.node:
                inp_names = []
                for inp in node.input:
                    # if (remove_nodes[0] in inp) or (remove_nodes[1] in inp):
                    #     pass
                    # else:
                    inp_names.append(inp)

                del node.input[:]
                node.input.extend(inp_names)
        print("Removed nodes: ", ', '.join(remove_nodes))
        frozen_graph = convert_variables_to_constants(session, mod_graph_def, 
        output_names, freeze_var_names)

        return frozen_graph

if __name__ == "__main__":

    args = parse_args()
    model = create_model(args)

    frozen_graph = freeze_session(K.get_session(),
        output_names=[out.op.name for out in model.outputs], clear_devices=True)
    if args.write_graph:
        tf.train.write_graph(frozen_graph, "models/mnist_vgg8_arcface_5d", name=args.pb_filename, as_text=False)
        tf.train.write_graph(frozen_graph, "models/mnist_vgg8_arcface_5d", name=args.pb_filename_text, as_text=True)
    elif args.visualize_graph:
        print("Visualizing graph..")
        tf.keras.utils.plot_model(
            model,
            to_file="model.png",
            show_shapes=False,
            show_layer_names=True,
            rankdir="TB",
        )

        