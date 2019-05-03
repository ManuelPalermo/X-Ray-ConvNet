# X-Ray Convolutional Neural Network
A Keras simplified implementation based on [ChesXNet](https://github.com/zoogzog/chexnet) for pathology detection in frontal chest X-ray images which was in turn based on the original paper presented [here](https://stanfordmlgroup.github.io/projects/chexnet/). 
Learning was the main motivation for this work.



# Dataset
The ChestX-ray dataset comprises 112,120 frontal-view chest X-ray with 14 disease labels, which are later simplified into 0(Normal xray) and 1(Abnormal xray - Pathology found).



# Preprocessing
Preprocessing was applied before training, to reduce in-train cpu time, and images were saved to a new databse (./database_preprocessed/...).
<br><br>
The preprocessing consisted on:
  * Applying Contrast Limited Adaptive Histogram Equalization (CLAHE) to correct contrast(might introduce some error).
  * Resize images from 1024x1024p to 128x128p (Bigger dimensions would be preferable as most often patologies appear on a small area on the image, which might get lost or distorted upon resize. The dimensions were limited by the GPU memory).

<!---
[//]: # (![Xray after applying contrast](https://i.imgur.com/Z9aIY77.png))
-->




# Usage
  * Clone repository.
  * Download the ChestX-ray14 database from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737).
  * Unpack archives in separate directories (e.g. images_001.tar.gz into images_001).
  * Run PreprocessData.py to create new database with processed data.
  * Run Main.py with desired Parameters.
