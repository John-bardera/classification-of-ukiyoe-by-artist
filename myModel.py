# coding: utf-8

import tensorflow as tf
from myConv import MyConv
from myFC import MyFC


class MyModel(tf.keras.Model):
	def __init__(self):
		super(MyModel, self).__init__()
		self.conv1 = MyConv(chn=64, conv_kernel=(3, 3), pool_kernel=(2, 2), is_pool=True)
		self.conv2 = MyConv(chn=128, conv_kernel=(3, 3), pool_kernel=(2, 2), is_pool=True)
		self.conv3 = MyConv(chn=128, conv_kernel=(3, 3), pool_kernel=(2, 2), is_pool=False)
		"""
		self.conv3 = MyConv(chn=256, conv_kernel=(3, 3), pool_kernel=(2, 2), is_pool=True)
		self.conv4 = MyConv(chn=512, conv_kernel=(3, 3), pool_kernel=(2, 2), is_pool=False)
		self.fc = MyFC(hidden_chn=512, out_chn=10)
		"""
		self.fc = MyFC(hidden_chn=128, out_chn=10)

	def call(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		# x = self.conv4(x)
		return self.fc(x)
