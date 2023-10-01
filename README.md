# Sigspatial Cup 2023 - Auto-identification of Supraglacial Lakes on the Greenland ice sheet from Satellite Imagery

## General description
This repository includes the code that automates the detection of surface lakes on the Greenland ice sheet using satellite imagery. The goal is to identify and tag these lakes as polygons, in order to monitor the behavior of these lakes across multiple summer melt seasons.

## How to run the project

## Solution description
### Dataset
We have employed the set of four large multi-part satellite images given by challenge organizers.
In order to increase the number of images available to train the network, we have cut each training image, already splitted into regions, into 512x512 images through a sliding window.
Images obtained are not overlapping in order to decrease correlation between consecutive images.
To overcome the jagged boundary problem we have bounded the initial regional image with the smallest external rectangle containing the region itself.

### Model

### Data Augmentation

### Post-processing
Polygons with area greater than 100 000 m^2 have not been considered.

### Predictive method
Since model takes as input 512x512 images, test images have been splitted using a sliding window as the previously explained procedure for training images.
Then predictions masks related to the same region and time period have been merged together to obtain teh final output.

### Credits
[Federico Borra](https://github.com/RicoBorra), 
[Claudio Savelli](https://github.com/ClaudioSavelli), 
[Giulia Monchietto](https://github.com/juliette23)




