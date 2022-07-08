import requests, shutil, PIL
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

resnet18 = tf.keras.models.load_model('models/Resnet18')

# Check its architecture
resnet18.summary()
# print("The current number of layers in the model")
# print(len(resnet18.layers))
# for layer in resnet18.layers:
#    print(layer)
#tf.keras.utils.plot_model(resnet18, 'my_first_model.png')


img = image.load_img('dog.jpg')
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
tf_output = resnet18.predict(x)
print(tf_output.shape)
