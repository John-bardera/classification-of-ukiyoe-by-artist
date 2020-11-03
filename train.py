# coding: utf-8

import tensorflow as tf
from myModel import MyModel


if __name__ == '__main__':
	H, W, C = 28, 28, 1
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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

		ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
		for epoch in range(epochs):
			for images, labels in ds_train:
				train_step(images, labels)
				tf.print(f'Epoch: {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result() * 100}')
		train_loss.reset_states()
		train_accuracy.reset_states()

	train(model, x_train, y_train, epochs=2, batch_size=32)
