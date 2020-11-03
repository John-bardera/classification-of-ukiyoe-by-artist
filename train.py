# coding: utf-8

import time
import numpy as np
import tensorflow as tf
from myModel import MyModel
from dataLoader import DataLoader


if __name__ == '__main__':
	# 4 mnist
	# H, W, C = 28, 28, 1
	# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	# 4 ukiyoe
	DATA_PATH = './data/'
	H, W, C = 224, 224, 3
	RH, RW = 224, 224
	x_train, y_train, x_test, y_test = DataLoader(0.2).load(DATA_PATH)
	# x_train = tf.image.resize_with_pad(x_train, RH, RW)
	# x_test = tf.image.resize_with_pad(x_test, RH, RW)
	if C == 1:
		x_train = np.sum(x_train, axis=-1) / 3
		x_test = np.sum(x_test, axis=-1) / 3

	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255
	x_train = x_train.reshape(x_train.shape[0], H, W, C)
	x_test = x_test.reshape(x_test.shape[0], H, W, C)

	model = MyModel()

	loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
	optimizer = tf.keras.optimizers.Adam()

	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

	test_loss = tf.keras.metrics.Mean(name='train_loss')
	test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')


	@tf.function
	def train_step(x, t):
		with tf.GradientTape() as tape:
			pred, weights = model(x, training=True)
			loss = loss_object(t, pred) + tf.norm(weights[0], axis=0)

		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		train_loss(loss)
		train_accuracy(t, pred)


	@tf.function
	def test_step(x, t):
		test_pred, test_weights = model(x)
		t_loss = loss_object(t, test_pred) + tf.norm(test_weights[0], axis=0)

		test_loss(t_loss)
		test_accuracy(t, test_pred)


	def train(model, x_train, y_train, epochs, batch_size):
		ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size)
		for epoch in range(epochs):
			for batch_num, (images, labels) in enumerate(ds_train):
				timef = time.time()
				if H != RH:
					images = resize(images)
				train_step(images, labels)
				timet = time.time()
				if batch_num % 10 == 0:
					tf.print(
						f'Epoch-Batch: {epoch + 1}-{batch_num + 1}, '
						f'Time: {"{:.2f}".format(timet - timef)}s, '
						f'Loss: {train_loss.result()}, '
						f'Accuracy: {train_accuracy.result() * 100}'
					)
			test_step(x_test, y_test)
			tf.print(
				f'---Test Result---\n'
				f'Epoch: {epoch + 1}, '
				f'Loss: {test_loss.result()}, '
				f'Accuracy: {test_accuracy.result() * 100}'
			)
		train_loss.reset_states()
		train_accuracy.reset_states()

	def resize(images):
		return tf.image.resize(images, [RH, RW])

	train(model, x_train, y_train, epochs=10, batch_size=32)
