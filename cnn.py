# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 15:28:52 2019

@author: Vishal
"""


#importing the libraries

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import Callback
from keras.layers import Dropout
import os
from keras.backend import backend

class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1
 
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'
 
script_dir = os.path.dirname(__file__)
training_set_path = os.path.join(script_dir, 'dataset/training_set')
test_set_path = os.path.join(script_dir, 'dataset/test_set')

#inintializing the ANN

classifier = Sequential()

# Convolutional layer
# Step 1 - Convolution
input_size = (128, 128)
classifier.add(Conv2D(32, (3, 3), input_shape=(*input_size, 3), activation='relu'))
 
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 is optimal
 
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1, activation='sigmoid'))
 
# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# we create two instances with the same arguments

from keras.preprocessing.image import ImageDataGenerator


batch_size = 32
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=input_size,
        batch_size= batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=input_size,
        batch_size=batch_size,
        class_mode='binary')

history = LossHistory()

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/batch_size,
        epochs=90,
        validation_data=validation_generator,
        validation_steps=2000/batch_size,
        max_q_size=100,
        callbacks=[history])

model_backup_path = os.path.join(script_dir, 'dataset/cat_or_dogs_model.h5')
classifier.save(model_backup_path)
print("Model saved to", model_backup_path)
 
# Save loss history to file
loss_history_path = os.path.join(script_dir, 'loss_history.log')
myFile = open(loss_history_path, 'w+')
myFile.write(history.losses)
myFile.close()
 
backend.clear_session()
print("The model class indices are:", training_set.class_indices)

"""
import numpy as np 
from keras.preprocessing import image

test_image = image.load_img('dataset/dog_3.jpg', target_size= (64, 64))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
training_set.class_indices
result = classifier.predict(test_image)
if result[0][0] == 1:
    print("Dog")
else:
    print("cat")"""
