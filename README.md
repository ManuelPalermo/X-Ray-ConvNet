# X-Ray Convolutional Neural Network
A Keras simplified implementation based on [ChesXNet](https://github.com/zoogzog/chexnet) for pathology detection in frontal chest X-ray images. Classification of Xray as Normal or Abnormal.
  [Original Paper](https://stanfordmlgroup.github.io/projects/chexnet/)


# Dataset
The ChestX-ray14 dataset comprises 112,120 frontal-view chest X-ray with 14 disease labels, which where simplified into 0(Normal xray) and 1(Abnormal xray) wether no pathology was found or any pathology was found.

# Preprocessing
Preprocessing was applied before training, and images saved to ./database_preprocessed/
The preprocessing consists in:
  * Applying Contrast Limited Adaptive Histogram Equalization (CLAHE) to increase contrast
  * Resize images from 1024x1024p to 128x128p
![Xray after applying contrast](https://i.imgur.com/Z9aIY77.png)


# Usage
  *Download the ChestX-ray14 database from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737)
  *Unpack archives in separate directories (e.g. images_001.tar.gz into images_001)
  *Run PreprocessData.py to create new database with processed data(increased contrast, image downsize(128x128))
  *Run python Main.py with desired Parameters
