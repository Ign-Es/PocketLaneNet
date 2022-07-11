import tensorflow as tf
from tensorflow.keras.models import Model


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

# Import saved ResNet18 model
resnet18 = tf.keras.models.load_model('../models/Resnet18')
# Check its architecture
print("Resnet18 Summary:\n")
resnet18.summary()
resnet14 = Model(resnet18.inputs, resnet18.get_layer("activation_11").output)
print("Resnet18 Summary:\n")
resnet14.summary()
resnet14.save('models/Resnet14')
