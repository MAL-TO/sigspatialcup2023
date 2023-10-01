# Sigspatial Cup 2023 - Auto-identification of Supraglacial Lakes on the Greenland ice sheet from Satellite Imagery

## General description
This repository includes the code that automates the detection of surface lakes on the Greenland ice sheet using satellite imagery. The goal is to identify and tag these lakes as polygons, in order to monitor the behavior of these lakes across multiple summer melt seasons.

## How to run the project
- Install the necessary libraries using the requirements.txt file.
- Run the file runner.py (e.g. with `python runner.py`), all neceesary intermediate files and folders will be created in the current working directory

## Solution description
### Dataset
We have employed the set of four large multi-part satellite images given by challenge organizers.
In order to increase the number of images available to train the network, we have cut each training image, already split into regions, into 512x512 images through a sliding window.
To overcome the jagged boundary problem we have bounded the initial regional image with the smallest external rectangle containing the region itself.

### Model
We used a DeepLabV3+ segmentation model with a ResNet18 backbone (in order to decrease the chance of overfitting given the low amount of data available: most of the images contain almost no information since they are basically white)
We used the Unified Focal Loss since it has been proven to be state of the art, especially in presence of class imbalance.

### Data Augmentation
Apart from the overlapping sliding window, we have employed random flips and rotations during the training phase.

### Post-processing
Polygons with area greater than 100 000 m^2 have not been considered.

### Predictive method
Since model takes as input 512x512 images, test images have been split using a sliding window as per the previously explained procedure for training images.
Then predictions masks related to the same region and time period have been merged together to obtain the final output.


### Credits
[Federico Borra](https://github.com/RicoBorra), 
[Claudio Savelli](https://github.com/ClaudioSavelli), 
[Giulia Monchietto](https://github.com/juliette23)




