from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# Constructing the Convolutional Neural Network
classifier = Sequential()  # creating our object classifier, and setting it to the Sequential class

classifier.add(Convolution2D(128, (3, 3), activation='relu', strides=(1, 1), input_shape=(128, 128, 3)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(128, (3, 3), activation='relu', strides=(1, 1), input_shape=(128, 128, 3)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(128, (3, 3), activation='relu', strides=(1, 1), input_shape=(128, 128, 3)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the CNN to our images

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,  # apply random transformations
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')  # since we have a binary outcome

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32,  # number of images in our training set
        epochs=25,  # number of epochs we want to train in our CNN
        validation_data=test_set,
        validation_steps=2000/32)

classifier.save("dog_cat_class.h5")

