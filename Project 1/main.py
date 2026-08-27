import tensorflow as tf

from coco_builder import build_coco_dataset

build_coco_dataset()
# data = 

# model = tf.keras.Sequential([ # defining the nerual network
#    tf.keras.layers.Flatten(input_shape=(28, 28)),
#    tf.keras.layers.Dense(128, activation=tf.nn.relu), #  RELU = rectified linear unit.It doesn't matter what you put for 128 it just changes the number of 'nerouns'. More nerouns means slower run times
#    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])