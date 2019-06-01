from tensorflow.python.keras.models import load_model
import numpy as np
from tensorflow.python.keras.preprocessing import image

model = load_model('dog_cat_class.h5')

def predict(filename):
    test_image = image.load_img(filename, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        print('dog')
    else:
        print('cat')

predict('download.jpg')