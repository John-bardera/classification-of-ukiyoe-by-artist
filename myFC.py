# coding: utf-8

import tensorflow as tf


class MyFC(tf.keras.layers.Layer):
	def __init__(self, hidden_chn=64, out_chn=10):
		super(MyFC, self).__init__()
		self.flatten = tf.keras.layers.Flatten()
		self.fc1 = tf.keras.layers.Dense(hidden_chn, activation='relu')
		self.fc2 = tf.keras.layers.Dense(hidden_chn, activation='relu')
		"""
		self.fc2 = tf.keras.layers.Dense(hidden_chn * 2, activation='relu')
		self.fc3 = tf.keras.layers.Dense(hidden_chn, activation='relu')
		self.fc4 = tf.keras.layers.Dense(hidden_chn / 2, activation='relu')
		"""
		self.fcOut = tf.keras.layers.Dense(out_chn, activation='softmax')

	def call(self, x):
		x = self.flatten(x)
		x = self.fc1(x)
		x = self.fc2(x)
		"""
		x = self.fc3(x)
		x = self.fc4(x)
		"""
		x = self.fcOut(x)
		return x, self.fcOut.weights
