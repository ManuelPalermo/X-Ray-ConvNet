from keras.models import Model

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation, concatenate
from keras.applications import mobilenetv2, densenet, Xception, NASNetMobile


class BinaryModel:
	def __init__(self, model2load='custom', percent2retrain=1/4, image_dimensions=(128,128,3), n_classes=1):
		self.input_dim  = image_dimensions
		self.n_classes  = n_classes
		self.model      = self.select_model(model2load, percent2retrain)


	def select_model(self, model2load, percent2retrain):
		'Selects the desired model to be loaded'
		if 0>percent2retrain>1:
			raise Exception('Invalid train percentage chosen! Value must be between 0-1')
		elif model2load == 'custom':
			return self.custom_model()
		elif model2load == 'mobile':
			return self.mobile_net(percent2retrain)
		elif model2load == 'nasmobile':
			return self.nas_mobile_net(percent2retrain)
		elif model2load == 'dense121':
			return self.dense_net121(percent2retrain)
		elif model2load == 'dense169':
			return self.dense_net169(percent2retrain)
		elif model2load == 'xception':
			return self.xception(percent2retrain)
		else:
			raise Exception ('No valid net has been chosen! Choose one of: (mobile, nasmobile, dense121, dense169, xception)')




	def mobile_net(self, percent2retrain):
		'Returns a mobilenet architecture NN'
		mobile_net_model = mobilenetv2.MobileNetV2(input_shape=self.input_dim,
							                        weights='imagenet',
							                        include_top=False)
		#freeze base layers
		if percent2retrain<1:
			for layer in mobile_net_model.layers[:-int(len(mobile_net_model.layers)*percent2retrain)]: layer.trainable = False

		# add classification top layer
		model = Sequential()
		model.add(mobile_net_model)
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(self.n_classes, activation='sigmoid'))
		return model


	def nas_mobile_net(self, percent2retrain):
		'Returns a mobilenet architecture NN'
		nas_mobile_model = NASNetMobile(input_shape=self.input_dim,
                                        weights='imagenet',
                                        include_top=False)
		# freeze base layers
		if percent2retrain < 1:
			for layer in nas_mobile_model.layers[:-int(len(nas_mobile_model.layers)*percent2retrain)]: layer.trainable = False

		# add classification top layer
		model = Sequential()
		model.add(nas_mobile_model)
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(self.n_classes, activation='sigmoid'))
		return model


	def dense_net121(self, percent2retrain):
		'Returns a Densenet121 architecture NN'
		dense_model = densenet.DenseNet121(input_shape=self.input_dim,
                                           weights='imagenet',
                                           include_top=False)
		# freeze base layers
		if percent2retrain < 1:
			for layer in dense_model.layers[:-int(len(dense_model.layers)*percent2retrain)]: layer.trainable = False

		# add classification top layer
		model = Sequential()
		model.add(dense_model)
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(self.n_classes, activation='sigmoid'))
		return model


	def dense_net169(self, percent2retrain):
		'Returns a Densenet169 architecture NN'
		dense_model = densenet.DenseNet169(input_shape=self.input_dim,
		                                   weights='imagenet',
		                                   include_top=False)
		# freeze base layers
		if percent2retrain < 1:
			for layer in dense_model.layers[:-int(len(dense_model.layers)*percent2retrain)]: layer.trainable = False

		# add classification top layer
		model = Sequential()
		model.add(dense_model)
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(self.n_classes, activation='sigmoid'))
		return model


	def xception(self, percent2retrain):
		'Returns a Xception architecture NN'
		xception_model= Xception(input_shape=self.input_dim,
                               weights='imagenet',
                               include_top=False)
		
		# freeze base layers
		if percent2retrain < 1:
			for layer in xception_model.layers[:-int(len(xception_model.layers)*percent2retrain)]: layer.trainable = False

		# add classification top layer
		model = Sequential()
		model.add(xception_model)
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(self.n_classes, activation='sigmoid'))
		return model


	def custom_model(self):
		'Creates the keras model for binary output'
		# if smaller image_dimensions are used, reduce the number of pooling layers
		model = Sequential()

		model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=self.input_dim))
		model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same'))
		model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same'))
		model.add(BatchNormalization())
		model.add(Dropout(0.1))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
		model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.1))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
		model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.1))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
		model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.1))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
		model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.1))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='Same', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.1))
		model.add(MaxPool2D(pool_size=(2, 2)))

		model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='Same', activation='relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.1))
		model.add(MaxPool2D(pool_size=(2, 2)))

		#dense block with activation function
		model.add(Flatten())
		model.add(Dense(128, activation="relu"))
		model.add(Dropout(0.5))
		model.add(Dense(self.n_classes, activation="sigmoid"))
		return model


	def get_model(self):
		'Returns the created model'
		return self.model
