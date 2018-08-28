# X-Ray Convolutional Neural Network
A Keras simplified implementation based on [ChesXNet](https://github.com/zoogzog/chexnet) for pathology detection in frontal chest X-ray images which is in turn based based on the original paper presented [here](https://stanfordmlgroup.github.io/projects/chexnet/). 



# Dataset
The ChestX-ray dataset comprises 112,120 frontal-view chest X-ray with 14 disease labels, which are later simplified into 0(Normal xray) and 1(Abnormal xray - Pathology found).



# Preprocessing
Preprocessing was applied before training, and images saved to new databse (./database_preprocessed/...)<br>
The preprocessing consists in:
  * Applying Contrast Limited Adaptive Histogram Equalization (CLAHE) to increase contrast
  * Resize images from 1024x1024p to 128x128p
  
  
![Xray after applying contrast](https://i.imgur.com/Z9aIY77.png)



# Usage
  * Clone repository
  * Download the ChestX-ray14 database from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737)
  * Unpack archives in separate directories (e.g. images_001.tar.gz into images_001)
  * Run PreprocessData.py to create new database with processed data(increased contrast, image downsize(128x128))
  * Run python Main.py with desired Parameters
  
  

# Results
The model was able to achieve ~69% accuracy on test data, after 5h of training on a GTX 950M with only 15k training images.
Better results could certainly be achieved over a longer training period, with more data.
