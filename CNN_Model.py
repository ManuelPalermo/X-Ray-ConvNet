from keras.models import Model

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation, concatenate


class BinaryModel:
	def __init__(self, image_dimensions=(128,128,1)):
		self.input_dim = image_dimensions
		self.model     = self.create_model()


	def create_model(self):
		'Creates the keras model for binary output'
		# if smaller image_dimensions are used, reduce the number of pooling layers
		model = Sequential()

		model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='elu', input_shape=self.input_dim))
		model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same'))
		model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same'))
		model.add(BatchNormalization())
		model.add(Dropout(0.25))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='elu'))
		model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='elu'))
		model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='elu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.25))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='elu'))
		model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='elu'))
		model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='elu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.25))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='elu'))
		model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='elu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.25))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='Same', activation='elu'))
		model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='Same', activation='elu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.25))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='Same', activation='elu'))
		model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='Same', activation='elu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.25))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='Same', activation='elu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.25))
		model.add(MaxPool2D(pool_size=(2, 2)))

		#dense block with activation function
		model.add(Flatten())
		model.add(Dense(1024, activation="elu"))
		model.add(Dropout(0.5))
		model.add(Dense(1, activation="sigmoid"))
		return model


	def get_model(self):
		'Returns the created model'
		return self.model