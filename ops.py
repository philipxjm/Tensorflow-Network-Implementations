import tensorflow as tf
import math
import numpy as np
import functools as ft

try:
	image_summary = tf.image_summary
	scalar_summary = tf.scalar_summary
	histogram_summary = tf.histogram_summary
	merge_summary = tf.merge_summary
	SummaryWriter = tf.train.SummaryWriter
except:
	image_summary = tf.summary.image
	scalar_summary = tf.summary.scalar
	histogram_summary = tf.summary.histogram
	merge_summary = tf.summary.merge
	SummaryWriter = tf.summary.FileWriter

def conv2d(input, output_dim,
					 conv_h=5, conv_w=5, conv_s=1,
					 padding="SAME", name="conv2d", stddev=0.02):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [conv_h, conv_w, input.get_shape().as_list()[-1], output_dim],
			initializer=tf.truncated_normal_initializer(stddev=stddev))
		b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
		c = tf.nn.conv2d(input, w, strides=[1, conv_s, conv_s, 1], padding=padding)

		return c + b

def max_pool(input, ksize=2, pool_stride=2, padding="SAME", name="max_pool"):
	with tf.variable_scope(name):
		return tf.nn.max_pool(
				input,
				ksize=[1,ksize,ksize,1],
				strides=[1,pool_stride,pool_stride,1],
				padding=padding)

def fully_connected(input, output_dim, name="fc", stddev=0.02):
	with tf.variable_scope(name):
		unfolded_dim = ft.reduce(lambda x, y: x*y, input.get_shape().as_list()[1:])
		w = tf.get_variable('w',
			[unfolded_dim, output_dim],
			initializer=tf.truncated_normal_initializer(stddev=stddev))
		b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
		input_flat = tf.reshape(input, [-1, unfolded_dim])

		return tf.matmul(input_flat, w) + b

def weight_variable(shape, name="weight"):
	with tf.variable_scope(name):
		randomized_initial = tf.truncated_normal(shape, stddev = 0.1)
		return tf.get_variable('w', [k_h, k_w, input_.get_shape().as_list()[-1], output_dim],
			initializer=tf.truncated_normal_initializer(stddev=stddev))

def bias_variable(shape, name="bias"):
	with tf.variable_scope(name):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def relu(x, name="relu"):
	return tf.nn.relu(x)
