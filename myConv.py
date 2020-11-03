# coding: utf-8
import tensorflow as tf


class MyConv(tf.keras.layers.Layer):
	def __init__(self, chn=32, conv_kernel=(3, 3), pool_kernel=(2, 2), is_pool=True):
		super(MyConv, self).__init__()
		self.is_pool = is_pool

		self.conv = tf.keras.layers.Conv2D(chn, conv_kernel)
		self.batchnorm = tf.keras.layers.BatchNormalization()
		self.relu = tf.keras.layers.ReLU()
		self.pool = tf.keras.layers.MaxPool2D(pool_kernel)

	def call(self, x):
		x = self.conv(x)
		x = self.batchnorm(x)
		x = self.relu(x)

		if self.is_pool:
			x = self.pool(x)

		return x
