# coding: utf-8

import os
import numpy as np


class DataLoader(object):
	def __init__(self, validation_size: float):
		self._basename_list = [
			'ukiyoe-train-imgs.npz',
			'ukiyoe-train-labels.npz'
		]
		self.validation_size = validation_size

	def load(self, data_path: str, random_seed: int=13) -> np.ndarray:
		filenames_list = self._make_filenames(data_path)
		data_list = [np.load(filename)['arr_0'] for filename in filenames_list]

		all_images, all_labels = data_list

		# shuffle data
		np.random.seed(random_seed)
		perm_idx = np.random.permutation(len(all_images))
		all_images = all_images[perm_idx]
		all_labels = all_labels[perm_idx]

		# split train and validation
		if self.validation_size < 1:
			validation_num = int(len(all_labels) * self.validation_size)
		else:
			validation_num = int(self.validation_size)

		validation_images = all_images[:validation_num]
		validation_labels = all_labels[:validation_num]

		train_images = all_images[validation_num:]
		train_labels = all_labels[validation_num:]

		return train_images, train_labels, validation_images, validation_labels

	def _make_filenames(self, datapath: str) -> list:
		filenames_list = [os.path.join(datapath, basename) for basename in self._basename_list]
		return filenames_list
