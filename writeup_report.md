#**Behavioral Cloning** 



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture.png "Model Visualization"
[image3]: ./examples/center_2017_02_13_15_30_52_605-recovery1.png "Recovery Image"
[image4]: ./examples/center_2017_02_13_15_33_08_937-recovery2.png "Recovery Image"
[image6]: ./examples/Normal-image.png "Normal Image"


###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 32 and 48 (model.py lines 174-176) 

Further convolution neural network with 3x3 filter sizes and depths 64 (model.py lines 178-179) 

Each convolutional layer includes RELU layers to introduce nonlinearity (code line 174 -179 ), and the data is normalized in the model using a Keras lambda layer (code lines 171). 

This is followed by fully connected layers with depths 100, 50, 10 and 1 (model.py lines 183 -195 ) 

####2. Attempts to reduce overfitting in the model


The model contains subsamplling during convolutional layers in order to reduce overfitting (model.py lines 120 -137). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 198 ).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with LeNet try it out and then improvise

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because of the nature of the image data. However, with just LeNet could not get satisfactory results, hence I opted for reproducing the Nvidia model. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I sub sampled the images and added variations to input data set by adding random brightness and shadows

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track near the bridges and curves.  to improve the driving behavior in these cases, I drove in reverse direction and collected data set and included recovery images

The biggest revelation was with respect to the nature of data set. I had four types of data set
##### 0. Driving forward with three laps
##### 1. Collect recovery data - starting from edges
##### 2. Driving in the reverse direction
##### 3. Intentionally brake the car and reduce the throttle, as I move towards the edges

I was able to observe the model was able to mimick driving behavior data. Specially, since throttle was not included in the simulation, that adverself affected the model. Intending to go towards the edges

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road with just reverse direction data

####2. Final Model Architecture

The final model architecture (model.py lines 174-179) consisted of a convolution neural network with the following layers and layer sizes

Here is a visualization of the architecture 

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]


Then I did not collect data on track two.

To augment the data sat, I also flipped images and angles thinking that this would improve regularization

After the collection process, I had 8000 data points. I then preprocessed this data by adding random brightness, flipping the images and adding random shadows

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by loss stauration. I used an adam optimizer so that manually training the learning rate wasn't necessary.


The video of the simulated drive can be seen here,
https://youtu.be/7Vwx5-i0eEU

