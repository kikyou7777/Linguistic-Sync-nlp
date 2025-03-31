import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Physical devices:", tf.config.list_physical_devices('CPU'))
print("Metal devices:", tf.config.list_physical_devices('GPU'))
