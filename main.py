import requests, shutil, PIL
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, \
    MaxPool2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np

# Parameters
c = 4
h = 36
w = 150

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

resnet14 = tf.keras.models.load_model('models/Resnet14')

# Check its architecture
resnet14.summary()
img = image.load_img('lane.jpg')
x = image.img_to_array(img)
x = tf.image.resize(x, [288, 800])
x = np.expand_dims(x, axis=0)
print(x.shape)
tf_output = resnet14.predict(x)
print(tf_output.shape)
tf_output = MaxPool2D((2, 2))(tf_output)
print(tf_output.shape)
tf_output = Conv2D(filters=8, kernel_size=1, padding="same", activation="linear", use_bias=False, name="dim_reduction")(tf_output)
print(tf_output.shape)
tf_output = Flatten()(tf_output)
print(tf_output.shape)
tf_output = Dropout(0.5)(tf_output)
Dense(units=2048, use_bias=True, activation='linear', name='fc')(x)
print(tf_output.shape)
tf_output = Dropout(0.5)(tf_output)
Dense(units=2048, use_bias=True, activation='linear', name='fc')(x)

# def model_test(input_shape, sub_model):
#  inputs = Input(input_shape)
#  eblock_1_1 = dense_convolve(inputs, n_filters=growth_rate)
#  eblock_1_2 = dense_convolve(eblock_1_1, n_filters=growth_rate);
#  dblock_1_1 = dense_convolve(eblock_1_2, n_filters=growth_rate);
#  dblock_1_2 = dense_convolve(dblock_1_1, n_filters=growth_rate);
#  final_convolution = Conv3D(2, (1, 1, 1), padding='same', activation='relu')(dblock_1_2)
#  intermedio = sub_model(final_convolution)
#  layer = LeakyReLU(alpha=0.3)(intermedio)
#  model = Model(inputs=inputs, outputs=layer)

#  return model
