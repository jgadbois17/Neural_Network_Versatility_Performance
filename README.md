# Neural Network Versatility and Performance 

This repository contains code, data, and results for my thesis culminating activity to satisfy requirements for my graduate program. 

* **Title:** [The Versatility and Performance of Neural Networks for Digital Signal Data](https://www.etdadmin.com/student/mylist?siteId=675&submissionId=783560)
* **Name:** Joseph (Joey) Gadbois 
* **School:** California State University, Long Beach 
* **Program:** Master's of Science, Applied Statistics 

***Note:** All code is done in Python 3.7.*

## Topic Information

The topic considers the versatility of neural networks and various architectures and their performance in comparison to traditional statistical models for digital signal data. The two specific types of data used are image data and one-dimensional sequential data and the experimental projects include image classification, image segmentation, time series forecasting, and anomaly detection. The models implemented for each task are provided below. 

* **Image Classification:** Linear Discriminant Analysis, Convolutional Neural Network 
* **Image Segmentation:** K-Means Clustering, U-Net Fully Convolutional Network 
* **Time Series Forecasting:** ARIMA, Recurrent Neural Network, Temporal Convolutional Network 
* **Anomaly Detection:** Normal Hidden Markov Model, Masked Autoencoder for Density Estimation 


## Repository Outline 

* **dspML:** folder containing all of the functions and classes used for experimentation 
  * ***datasets:*** folder containing all datasets used for the different topics 
  * ***data:*** contains functions to load the datasets from the base directory 
  * ***preprocessing:*** contains preprocessing functions for both image and sequential data 
  * ***plot:*** contains functions and classes for plotting data and results 
  * ***models:*** folder containing models implemented organized by image and sequence tasks including functions that go along with specific models 
  * ***evaluation:*** constains evaluation functions for the different tasks 
  * ***utils:*** contains a few random functions that I couldn't decide what script to include them in 

* **experiments:** folder containing one folder for each sub-topic included in the thesis where each topic folder contains python scripts of each experiment for each model 
* **notebooks:** folder containing one Jupyter notebook for each sub-topic to bring the whole process together for all models implemented 
* **exploratory:** folder containing random exploratory python scripts that were not included in the experiments 


