from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Dense, Reshape, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling3D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import os
imagecount, testcount, categorycount = 0, 0, 0
train_directory = 'data/train'
for i in os.listdir(train_directory + "/"):
    imagecount += len(os.listdir(train_directory+ "/" + str(i)))
    categorycount += 1

test_directory = 'data/test'
for i in os.listdir(test_directory + "/"):
    testcount += len(os.listdir(test_directory+ "/" + str(i)))

def plot(hist):
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


'''  Building and Compiling the CNN model  '''
model = Sequential()

# First convolution layer and pooling
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))  #INPUT: 1
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
model.add(Flatten())

# Adding a fully connected layer
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=categorycount, activation='softmax'))

# Compiling the CNN
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


'''  Generating Image Data(training and testing sets) '''
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_directory,
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 color_mode='grayscale',  #GRAYSCALE
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_directory,
                                            target_size=(64, 64),
                                            batch_size=5,
                                            color_mode='grayscale', #GRAYSCALE
                                            class_mode='categorical')


history = model.fit_generator(
        training_set,
        steps_per_epoch=imagecount,
        epochs=15,
        validation_data=test_set,
        validation_steps=testcount)

plot(history)

# Saving the model
model_json = model.to_json()
with open("newmodel.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('newmodel.h5')
