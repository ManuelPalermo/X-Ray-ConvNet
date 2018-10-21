import numpy as np
import keras
import cv2
#from skimage import io
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, images_paths, labels, batch_size=64, image_dimensions = (128,128,3), shuffle=True, augment=False):
		self.labels       = labels
		self.images_paths = images_paths
		self.dim          = image_dimensions
		self.batch_size   = batch_size
		self.shuffle      = shuffle
		self.augment      = augment
		self.on_epoch_start()


	def __len__(self):
		'Denotes the number of batches per epoch'
		return self.images_paths.shape[0] // self.batch_size

	def on_epoch_start(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.images_paths))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __getitem__(self, index):
		'Generate one batch of data'
		# Selects indices of data for next batch
		indexes = self.indexes[index*self.batch_size : (index + 1)*self.batch_size]
		batch_image  = [self.images_paths[k] for k in indexes]
		batch_labels = [self.labels[k] for k in indexes]

		# Generate data
		images, labels = self._data_generation(batch_image, batch_labels)

		# Augment data
		if self.augment == True:
			aug_images = ImageDataGenerator(
						featurewise_center=False,  # set input mean to 0 over the dataset
						samplewise_center=False,  # set each sample mean to 0
						featurewise_std_normalization=False,  # divide inputs by std of the dataset
						samplewise_std_normalization=False,  # divide each input by its std
						zca_whitening=False,  # apply ZCA whitening
						rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
						zoom_range=0.05,  # randomly zoom image
						shear_range=0.15,  # randomly shear image
						width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
						height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
						horizontal_flip=False,  # randomly flip images horizontally
						vertical_flip=False,  # randomly flip images vertically
						fill_mode='constant')  # how to fill newly created pixels (which appear for example after a rotation)
			aug_images.fit(images)
			images, labels = aug_images.flow(images, labels, batch_size=self.batch_size).next()
		#####prints examples of input images which will be fed to the NN
		#for (img, lbl) in zip(images,labels):
		#	plt.imshow(img[:,:,0], cmap='gray')
		#	plt.title(f'Label: {lbl[0]}')
		#	plt.show()
		#input()
		#####
		return images, labels

	def _data_generation(self, batch_images, batch_labels):
		'Generates data containing batch_size samples'
		# Initialization
		images = np.empty((self.batch_size, *self.dim))
		labels = np.empty((self.batch_size, 1), dtype=np.int8)
		# Generate data
		for i, (img_path, img_label) in enumerate(zip(batch_images, batch_labels)):
			images[i] = cv2.imread(img_path).reshape(self.dim)
			labels[i] = img_label
		return images, labels
