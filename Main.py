import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LambdaCallback


from CNN_Model import BinaryModel
from DataLoader import DataLoader



############################# Parameters ############################
test_trained_model     = False
load_previous_weights  = False

samples_to_train  = 20000     #max: 78468
samples_to_val    = 500     #max: 11219
samples_to_test   = 10000    #max: 22433
epochs = 100
batch_size = 32
image_shape = (128, 128, 1) # changes require reprocessing images
model_learn_rate = 0.00001

#decrease resource usage:
idle_time_on_batch = 0.1
idle_time_on_epoch = 30
#####################################################################




print('##### Loading Data #####')
################################################ Load Data ################################################
data_loader = DataLoader(batch_size=batch_size,
                        img_shape=image_shape,
                        ntrain=samples_to_train,
                        nval=samples_to_val,
                        ntest=samples_to_test,
                        undersample=True,
                        augment_data=True,
                        shuffle=True,
                        plot_distribuition=True)

train_data = data_loader.load_train_generator()
val_data   = data_loader.load_validation_generator()
test_data  = data_loader.load_test_generator()


################################################ Create NN model ################################################
if not test_trained_model:
	print('##### Building NN Model #####')
	model = BinaryModel(image_dimensions=image_shape).get_model()

	if load_previous_weights == True:
		print('Loading Model Weights')
		model.load_weights("model_weights.hdf5")

	optimizer = Adam(lr=model_learn_rate,
	                 beta_1=0.9,
	                 beta_2=0.999,
	                 epsilon=1e-08,
	                 decay=0.0,
	                 amsgrad=False)

	model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['acc'])

	learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
	                                            patience=5,
	                                            verbose=1,
	                                            factor=0.5,
	                                            min_lr=0.000000001)

	early_stop = EarlyStopping(monitor="val_loss",
	                           mode="min",
	                           patience=20)

	checkpoint = ModelCheckpoint('model_weights.hdf5',
	                             monitor='val_loss',
	                             verbose=1,
	                             save_best_only=True,
	                             mode='min',
	                             save_weights_only=True)

	# sleep after each batch and epoch (prevent laptop from melting) (sleeps for x sec)(remove for faster training)
	idle = LambdaCallback(on_epoch_end=lambda batch, logs: time.sleep(idle_time_on_epoch), on_batch_end=lambda batch,logs: time.sleep(idle_time_on_batch))

	# save model to json file
	with open("model.json", "w") as json_model:
		json_model.write(model.to_json())


	print('##### Training Model #####')
	########################################## Train Model ###############################################
	model.summary()
	history = model.fit_generator(generator=train_data,
	                              validation_data=val_data,
	                              epochs=epochs,
	                              steps_per_epoch=len(train_data),
	                              verbose=2,
	                              callbacks=[learning_rate_reduction, early_stop, checkpoint, idle],
	                              # use_multiprocessing=True,
	                              # workers=2
	                              )

	############################# Check Loss and Accuracy graphics over training ########################
	fig, ax = plt.subplots(2, 1, figsize=(6, 6))
	ax[0].plot(history.history['loss'], label="TrainLoss")
	ax[0].plot(history.history['val_loss'], label="ValLoss")
	ax[0].legend(loc='best', shadow=True)

	ax[1].plot(history.history['acc'], label="TrainAcc")
	ax[1].plot(history.history['val_acc'], label="ValAcc")
	ax[1].legend(loc='best', shadow=True)
	plt.show()



else: # if use_trained_model:
	print('##### Loading NN Model #####')
	from keras.models import model_from_json

	with open('model.json', 'r') as json_model:
		model = model_from_json(json_model.read())

	print('Loading Model Weights')
	model.load_weights("model_weights.hdf5")

	optimizer = Adam(lr=model_learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
	model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['acc'])




print('##### Evaluating Model on Test Data #####')
################################# Evaluate model on Test Data ############################
test_score = model.evaluate_generator(test_data, verbose=2)
print('\nModel Accuracy: ', test_score[1])

print('\nParameters used:',
	'\ntrain_samples:   ',samples_to_train,
	'\nepochs:          ',epochs,
	'\nbatch_size:      ',batch_size,
	'\ninit_learn_rate: ',model_learn_rate)


print('##### Plotting Confusion Matrix #####')
predict_out = model.predict_generator(test_data, verbose=2)
test_predict = (predict_out > 0.5).astype(np.int8)

conf_matrix = confusion_matrix(y_true=data_loader.test_data[1], y_pred=test_predict)

sns.heatmap(conf_matrix, annot=True, cmap='Blues', cbar=False, square=True, xticklabels=['Normal','Abnormal'], yticklabels=['Normal','Abnormal'])
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
