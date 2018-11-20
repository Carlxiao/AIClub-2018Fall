import numpy as np
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import os

"""
DeepDream (with guided picture)
Rewrite from: https://github.com/llSourcell/deep_dream_challenge
"""

def restore_graph(model_path):
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    input_tensor = tf.placeholder(np.float32, name='input')
    imagenet_mean = 117.0
    input_normalized = tf.expand_dims(input_tensor - imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input':input_normalized})
    
    return graph, input_tensor, sess

def showarray(a):
    a = np.uint8(np.clip(a, 0, 1)*255)
    plt.imshow(a)
    plt.show()

def render_deepdream(graph, input_tensor, sess,
                        target_tensor, image_base, image_dream,
                        n_iter=20, rate=1.5, n_octave=4, octave_scale=1.4):

    def resize(img, size):
        img = tf.expand_dims(img, 0)
        op = tf.image.resize_bilinear(img, size)[0,:,:,:]
        return sess.run(op)

    def calc_grad_tiled(img, grad_tensor, tile_size=512):
        '''Compute the value of tensor grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over 
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        # sx, sy = np.random.randint(sz, size=2)
        sx, sy = np.random.randint(10, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for x in range(0, max(h-sz//2, sz), sz):
            for y in range(0, max(w-sz//2, sz), sz):
                tile = img_shift[x:x+sz, y:y+sz]
                g = sess.run(grad_tensor, {input_tensor: tile})
                grad[x:x+sz, y:y+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    dream_feature = sess.run(target_tensor, {input_tensor: image_dream})
    ch = dream_feature.shape[-1]
    dream_feature = tf.constant(np.reshape(dream_feature, [-1, ch]), tf.float32)
    base_feature = tf.reshape(target_tensor, [-1, ch])
    prod = tf.matmul(base_feature, tf.transpose(dream_feature))
    target_feature = tf.gather(dream_feature, tf.argmax(prod, 1), axis=0)

    loss = tf.reduce_mean(tf.square(base_feature - target_feature))
    grad = tf.gradients(loss, input_tensor)[0]

    # split the image into a number of octaves

    octaves = []
    image = image_base
    image_scale = image_base.shape[0] / image_dream.shape[0]
    image = resize(image, np.int32(np.float32(image.shape[:2]) / image_scale))
    for _ in range(n_octave - 1):
        hw = image.shape[:2]
        lo = resize(image, np.int32(np.float32(hw) / octave_scale))
        hi = image - resize(lo, hw)
        image = lo
        octaves.append(hi)
    
    # generate details octave by octave
    for octave in range(n_octave):
        if octave > 0:
            hi = octaves[-octave]
            image = resize(image, hi.shape[:2]) + hi
        for i in range(n_iter):
            # g = calc_grad_tiled(image, grad)
            g = calc_grad_tiled(image, grad, tile_size=512)
            # g = sess.run(grad, {input_tensor:image})
            image -= g * (rate / (np.abs(g).mean() + 1e-7))
        
    image = resize(image, np.int32(np.float32(image.shape[:2]) * image_scale))
    showarray(image/255.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', help='path of the base picture')
    parser.add_argument('--dream', help='path of the dream picture')
    parser.add_argument('--model', help='path of pretrained model',
                        default='model/inception5h/tensorflow_inception_graph.pb')
    args = parser.parse_args()
    
    graph, input_tensor, sess = restore_graph(args.model)
 
    img_base = np.float32(PIL.Image.open(args.base))
    img_dream = np.float32(PIL.Image.open(args.dream))
     
    op_name = 'mixed4c'
    tensor = graph.get_tensor_by_name("import/{}:0".format(op_name))
    # tensor = tensor[:, :, :, 7]
    render_deepdream(graph, input_tensor, sess, tensor, img_base, img_dream)
  
if __name__ == '__main__':
    main()
