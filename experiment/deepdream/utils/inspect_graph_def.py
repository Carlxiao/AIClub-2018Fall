from google.protobuf import text_format
import tensorflow as tf
import argparse
import os

"""
Print the name, input shapes and output shapes of (selected) ops in the graph stored in .pb file.
See: https://www.tensorflow.org/guide/extend/model_files#graphdef
"""

parser = argparse.ArgumentParser()
parser.add_argument('--file', help='path of the .pb file',
                    default='model/inception5h/tensorflow_inception_graph.pb')
args = parser.parse_args()

graph = tf.Graph()
graph_def = tf.GraphDef()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(args.file, 'rb') as f:
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def)

for op in graph.get_operations():
    if op.name.endswith(('_w', '_b', '_dim')): continue
    print('Op name:', op.name[7:])
    print('Input shapes:', [str(t.shape) for t in op.inputs])
    print('Output shapes:', [str(t.shape) for t in op.outputs])
