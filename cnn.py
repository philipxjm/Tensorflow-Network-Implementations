import argparse
import sys

import tensorflow as tf
import functools

from tools.ops import *

from mnist import mnist_data

def doublewrap(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

class Cnn_Model:
    def __init__(self,
            image,
            label,
            dropout=1.0,
            conv_size=5,
            conv_stride=1,
            ksize=2,
            pool_stride=2,
            padding="SAME"):
        self.image = image
        self.label = label
        self.dropout = dropout

        self.conv_size = conv_size
        self.conv_stride = conv_stride
        self.ksize = ksize
        self.pool_stride = pool_stride
        self.padding = padding

        self.prediction
        self.optimize
        self.accuracy

    @define_scope
    def prediction(self):
        #input image
        # input_image = self.image
        input_image = tf.reshape(self.image, [-1, 28, 28, 1])

        #conv1
        h_conv1 = relu(conv2d(input_image, 32, name='conv_1'))

        #pool1
        h_pool1 = max_pool(h_conv1, name='pool_1')

        #conv2
        h_conv2 = relu(conv2d(h_pool1, 64, name='conv_2'))

        #pool2
        h_pool2 = max_pool(h_conv2, name='pool_2')

        #fc1
        h_fc1 = relu(fully_connected(h_pool2, 1024, name='fc1'))

        #dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout)

        #fc2
        result = fully_connected(h_fc1_drop, 10, name='fc2')

        return result

    @define_scope
    def optimize(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
            logits=self.prediction))
        return tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    @define_scope
    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def weight_variable(self, shape):
        randomized_initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(randomized_initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

def main():
    # Import data
    mnist = mnist_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

    # Construct graph
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    dropout = tf.placeholder(tf.float32)
    model = Cnn_Model(image, label, dropout)

    # Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            images, labels = mnist.train.next_batch(10)
            images_eval, labels_eval = mnist.test.next_batch(10)
            if i % 100 == 0:
                accuracy = sess.run(model.accuracy, {image: images_eval,
                    label: labels_eval, dropout: 1.0})
                print('step %d, accuracy %g' % (i, accuracy))
            sess.run(model.optimize, {image: images, label: labels, dropout: 0.5})

    print("finished")

if __name__ == '__main__':
    main()
