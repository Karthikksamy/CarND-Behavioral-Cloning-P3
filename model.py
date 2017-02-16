import os
import csv
import cv2
import numpy as np
import sklearn
import pandas as pd 



#root_path = './data/'
root_path = '../p3-BehaviouralCloning/p3-data/'

#dirs = ['0 original three laps/','1 recovery data/','2 reverse direction/','3 braking throttle/']
dirs = ['2 reverse direction/']



images = []
measurements = []

############################## Section 1: Collect all data 

samples = []
for directory in dirs:
	with open(root_path+directory+'driving_log.csv') as csvfile:
		print('root csv file path:',root_path+directory+'driving_log.csv')
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)

	#print('last stored line',lines[-1])
	print('\nlines length is ',len(samples))	
	print('\nlines shape is ',np.array(samples).shape)	

############################## Section 2: split the data to fit in memory

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


print("************* train_samples",np.array(train_samples).shape)

angle_correction = 0.15 # this is a tunable parameter, to augment the data 

def augment_brightness_camera_images(image):
    """ This function is add brightness randomly to the data. 
    Courtesy:  https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ma7zjswgn """

    # The HSV - Hue Saturation Value representation converts the image from RGB space to HSV space
    # where the Value(brightness) represents the  brightness that is randomly increased

    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def add_random_shadow(image):
    """ This function is to add Shadows randomly to the data. 
    Courtesy:  https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ma7zjswgn """

    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)] = 1
    #random_bright = .25+.7*np.random.uniform()
    if (np.random.randint(2)==1 ):
    	random_bright = .5
    	cond1 = shadow_mask==1
    	cond0 = shadow_mask==0
    	if np.random.randint(2)==1 :
    		image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
    	else:
    		image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def generator(samples, batch_size=32):
    """ 
        This function reads the input set of lines from the CSV file and takes all the three sides of the camera 

        The images from these 3 camera angles are processed as below,
            1. random brightness added and steering angle correction required for that
            2. random shadow added to the image
            3. image flipped, to add balance between the left and right turning data. The steering angle inverted to the opposite side
            4. random brightness added to the flipped image """

    num_samples = len(samples)
    #batch_size = num_samples
    print('num_samples',num_samples)
    while 1: # Loop forever so the generator never terminates

        for offset in range(0, num_samples, batch_size): #this loop will be run for each iteration
        	#print('iteration------------>',offset, batch_size)
        	batch_samples = samples[offset:offset+batch_size]

        	images = []
        	angles = []
        	for batch_sample in batch_samples:
        		for i in range(3): #include the center, right and left angles 
        			file_name = root_path+batch_sample[i].split('/')[-3]+'/IMG/'+batch_sample[i].split('/')[-1]
        			image = cv2.imread(file_name)
        			images.append(image) # 

        		angle = float(batch_sample[3]) #steering angle is the fourth element in the input file
        		angles.append(angle)
        		angles.append(angle+angle_correction) #for right angle correction
        		angles.append(angle-angle_correction) #for left angle correction

        	############## Section 3: Augmenting the data to add balance and regularization to the learning
        	augmented_images = []
        	augmented_angles = []

        	for image,angle in zip(images, angles) : 
        		augmented_images.append(image) 
        		augmented_angles.append(angle)

        		augmented_images.append(augment_brightness_camera_images(image) )  #brightness augmentation
        		augmented_angles.append(angle)

        		augmented_images.append(add_random_shadow(image)) #add random shadow
        		augmented_angles.append(angle)


        		flipped_image = cv2.flip(image,1) # Generated new data here
        		flipped_angle = float(angle) * -1.0 #numpy array converts automatically  to string
        		augmented_images.append(flipped_image)  #### Included the new data
        		augmented_angles.append(flipped_angle)  #### Included the new data to the training data set

        		augmented_images.append(augment_brightness_camera_images(flipped_image) ) #brightness augmentation
        		augmented_angles.append(flipped_angle)


        	X_train = np.array(augmented_images)
        	y_train = np.array(augmented_angles)


        	#print("image shape",np.array(images).shape)
        	#print("augmented image shape",np.array(augmented_images).shape)
        	#print("X_train shape",X_train[-1].shape)
        	yield sklearn.utils.shuffle(X_train, y_train) #pass the iterator for containing the shuffled input data


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


ch, row, col = 3, 160, 320  #  image format channels, row, colum



#### keras model
import keras 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Implementing the NVIDIA architecture below,
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row,col,ch), output_shape=(row,col,ch))) # lambda for normalization
model.add(Cropping2D(cropping=((70,25),(0,0)))) #how much to crop row(i.e top,bottom),col(i.e left,right)

model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu')) 
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))

model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))

# tried with Len Initially
#model.add(MaxPooling2D())
#model.add(Convolution2D(16,5,5,subsample=(2,2),activation='relu'))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Flatten(input_shape=(160,320,3)))
#model.add(Dense(120))
#model.add(Dense(84))
model.add(Dense(1))

#using mse loss function as it is continuous output regression problem
model.compile(optimizer='adam', loss='mse')

#multiplying by 15 as each position has three images and each of on those are augmented further
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3*5, validation_data=\
	validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')

	
