import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import exposure, transform


def preprocess_data(n_to_process=-1, img_shape=(128,128)):
	'''
	Prepossesses all the images:
		- apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to increase contrast
		- resize to the desired dimensions

	'''
	for f in ('images_001', 'images_002','images_001', 'images_003','images_004', 'images_005','images_006',
	          'images_007','images_008', 'images_009', 'images_010', 'images_011', 'images_0012'):
		os.makedirs(f'../database_preprocessed/{f}', exist_ok=True)

	train_data = pd.read_csv('../dataset/train_1.txt', header=None, index_col=None)[0].str.split(' ', 1)
	val_data   = pd.read_csv('../dataset/val_1.txt', header=None, index_col=None)[0].str.split(' ', 1)
	test_data  = pd.read_csv('../dataset/test_1.txt', header=None, index_col=None)[0].str.split(' ', 1)

	# number of samples to process
	train_data = train_data if (n_to_process==-1 or n_to_process>len(train_data)) else train_data[:n_to_process]
	val_data   = val_data if (n_to_process ==-1 or n_to_process>len(val_data)) else val_data[:n_to_process]
	test_data  = test_data if (n_to_process ==-1 or n_to_process>len(test_data)) else test_data[:n_to_process]


	train_paths = train_data.apply(lambda x: '../database/' + x[0]).as_matrix()
	val_paths   = val_data.apply(lambda x: '../database/' + x[0]).as_matrix()
	test_paths  = test_data.apply(lambda x: '../database/' + x[0]).as_matrix()
	all_paths   = np.hstack((train_paths, val_paths, test_paths))

	i=0
	for img_path in all_paths:
		i += 1
		if  i % max(1, int(len(all_paths)/1000))==0: print(i, '/', len(all_paths))
		new_path = img_path.replace('database', 'database_preprocessed')
		img = plt.imread(img_path)
		img = exposure.equalize_adapthist(img, clip_limit=0.05)
		img = transform.resize(img, img_shape, anti_aliasing=True)
		plt.imsave(fname=new_path, arr=img, cmap='gray')

preprocess_data(n_to_process=-1, img_shape=(128,128))
