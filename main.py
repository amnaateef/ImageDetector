import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np


model = VGG16(weights='imagenet')

img_path = r'C:\images\download.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

predictions = model.predict(x)

decoded_predictions = decode_predictions(predictions, top=3)[0]

for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")
