from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution2D, Dropout, BatchNormalization
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
# from keras import optimizers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
from tensorflow.python.client import device_lib
import tensorflow as tf

print(device_lib.list_local_devices())
sess = tf.compat.v1.Session()
K.tensorflow_backend._get_available_gpus()
K.set_session(sess)

# Constructing the Convolutional Neural Network
classifier = Sequential()  # creating our object classifier, and setting it to the Sequential class

classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(1, 1), input_shape=(256, 256, 3)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())  # normalize our layer, allows model to converge faster in training; general speed improvements

classifier.add(Convolution2D(64, (3, 3), activation='relu', strides=(1, 1)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())

classifier.add(Convolution2D(128, (3, 3), activation='relu', strides=(1, 1)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())

classifier.add(Convolution2D(256, (3, 3), activation='relu', strides=(1, 1)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu', kernel_initializer='uniform'))
classifier.add(Dropout(0.4))  # to prevent overfitting
classifier.add(Dense(units=96, activation='relu', kernel_initializer='uniform'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=64, activation='relu', kernel_initializer='uniform'))
classifier.add(Dropout(0.6))
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# Compiling the CNN
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the CNN to our images

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.3,  # apply random transformations
        zoom_range=0.3,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(256, 256),
        batch_size=64,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(256, 256),
        batch_size=64,
        class_mode='binary')  # since we have a binary outcome

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32,  # number of images in our training set
        epochs=25,  # number of epochs we want to train in our CNN
        validation_data=test_set,
        validation_steps=2000/32)

classifier.save("dog_cat_classifier.h5")

