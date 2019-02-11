import cv2
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Activation
from keras.layers.convolutional import Cropping2D
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

samples = []
with open('data/driving_log.csv') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        samples.append(row)

#Training samples and validation samples
train_samples, validn_samples = train_test_split(samples, test_size=0.2)

#Generator for training and testing
def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []

            for sample in batch_samples:
                path = 'data/IMG/'
                center_img = cv2.cvtColor(cv2.imread(path + sample[0].split('/')[-1]), cv2.COLOR_BGR2RGB)
                left_img = cv2.cvtColor(cv2.imread(path + sample[1].split('/')[-1]), cv2.COLOR_BGR2RGB)
                right_img = cv2.cvtColor(cv2.imread(path + sample[2].split('/')[-1]), cv2.COLOR_BGR2RGB)

                angle = float(sample[3])
                correction = 0.4
                center_angle = angle
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                images.extend([center_img, left_img, right_img])
                angles.extend([center_angle, left_angle, right_angle])
                
                #Flipped data
#                 center_flipped_img = cv2.flip(center_img, 1)
#                 left_flipped_img = cv2.flip(left_img, 1)
#                 right_flipped_img = cv2.flip(right_img, 1)
#                 center_flipped_angle = center_angle * -1.0
#                 left_flipped_angle = left_angle * -1.0
#                 right_flipped_angle = right_angle * -1.0
                
#                 images.extend([center_flipped_img, left_flipped_img, right_flipped_img])
#                 angles.extend([center_flipped_angle, left_flipped_angle, right_flipped_angle])
                
                images.extend([cv2.flip(center_img, 1), cv2.flip(left_img, 1), cv2.flip(right_img, 1)])
                angles.extend([center_angle * -1.0, left_angle * -1.0, right_angle * -1.0])
                
            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)
            
train_generator = generator(train_samples, batch_size = 32)
validn_generator = generator(validn_samples, batch_size = 32)  

#Model similar to nVidia's implementation
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,20), (0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), activation='elu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='elu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='elu'))
model.add(Conv2D(64, (3,3), activation='elu'))
model.add(Conv2D(64, (3,3), activation='elu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
                    validation_data=validn_generator, validation_steps=len(validn_samples), epochs=5, verbose = 1)

model.save('model.h5')