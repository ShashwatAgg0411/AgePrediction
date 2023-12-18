
# Age Prediction using OpenCV and Deep Neural Networks

The project enables users to do age prediction on both static images and live videos. The project uses pre trained Deep Neural Network models for detecting face out of a frame and then doing the age predcitions.


## Pre-requisites

Ensure that you have the following dependencies installed:

* Python 3.x
* OpenCV
* TensorFlow (for pre-trained deep learning models)
* Numpy

You can install the required Python packages using the following command:

```bash
pip install opencv-python tensorflow numpy
```




## Project Structure

* `README.md`: Project documentation
* `deploy.prototxt`: The prototxt file specifies the architecture of the pre-trained deep learning model for face detection.
* `res10_300x300_ssd_iter_140000.caffemodel`: The caffe-model contains pre-trained weights for the face detection model.
* `age_deploy.prototxt`: The prototxt file specifies the architecture of the pre-trained deep learning model for face detection.
* `age_net.caffemodel`: The caffe-model contains pre-trained weights for the face detection model.
* `age_prediction_image.py`: The script for age prediction from static images.
* `age_prediction_video.py`: The script for age prediction from video streams.
## Features

The model is capabe to predict age on:-

- an image on device
- any Video on device
- a live webcam feed 
- any media accessible through URL


## Usage
### 1. Instructions for prediction on images

1.1. Clone the repository
```bash
git clone https://github.com/ShashwatAgg0411/AgePrediction.git
cd AgePrediction
```
1.2. Download the pre trained deep learning model architecture and model weights for both face detection and age prediction

1.3. Open the age_prediction_image.py script and make the following changes

* Comment out or remove the existing line that reads the input path from the command line (line 23):
```bash
# InputPath="shashwat1.jpg"
```
* Uncomment the line to specify the input path directly in the code (line 22):
```bash
InputPath="/path/to/your/input/image"
```
* Replace "/path/to/your/input/image" with the actual path to the image on device or an image URL for which you want to predict the age.
1.4.  Run the age_prediction_image script

### 2. Instructions for prediction on videos

2.1. Clone the repository
```bash
git clone https://github.com/ShashwatAgg0411/AgePrediction.git
cd AgePrediction
```
2.2. Download the pre trained deep learning model architecture and model weights for both face detection and age prediction

2.3. Open the age_prediction_video.py script and make the following changes

* Comment out or remove the existing line that reads the input path from the command line (line 145):
```bash
# InputPath=0
```
* Uncomment the line to specify the input path directly in the code (line 44):
```bash
InputPath="/path/to/your/input/file"
```
* Replace "/path/to/your/input/image" with 0 for webcam feed otherwise the actual path to the video on device or a video URL for which you want to predict the age.
2.4. Run the age_prediction_video script






