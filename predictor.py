import cnn
import numpy as np
from tensorflow.python.keras.preprocessing import image


test_image = image.load_img('insert_data_here', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.classifier.predict(test_image)
cnn.training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
