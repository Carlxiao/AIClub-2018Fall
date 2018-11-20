from google.protobuf import text_format
import tensorflow as tf
import argparse
import os

"""
Print the name, input shapes and output shapes of (selected) ops in the graph in a SavedModel.
See: https://www.tensorflow.org/guide/saved_model#build_and_load_a_savedmodel
"""

parser = argparse.ArgumentParser()
parser.add_argument('--file', help='path of the SavedModel directory',
                    default='model/resnet_v2_fp32_savedmodel_NCHW_jpg/1538687370')
args = parser.parse_args()

graph = tf.Graph()
graph_def = tf.GraphDef()
sess = tf.InteractiveSession(graph=graph)
tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], args.file)

for op in graph.get_operations():
    name = op.name
    # name = op.name[7:]
    if name.endswith(('_w', '_b', '_dim')): continue
    # if not name.startswith(('model')): continue
    print('Op name:', name)
    print('Input shapes:', [str(t.shape) for t in op.inputs])
    print('Output shapes:', [str(t.shape) for t in op.outputs])
