import time
import numpy as np
import tensorflow as tf
from nets import inception
slim = tf.contrib.slim
from tensorflow.python.framework import meta_graph
import numpy as np
import json
import pprint

train_batch_size = 5
eval_batch_size = 2
height, width = 150, 150
num_classes = 1000

train_inputs = tf.random_uniform((train_batch_size, height, width, 3))
inception.inception_v3(train_inputs, num_classes)
eval_inputs = tf.random_uniform((eval_batch_size, height, width, 3))
logits, _ = inception.inception_v3(eval_inputs, num_classes,
                                   is_training=False, reuse=True)
predictions = tf.argmax(logits, 1)

with tf.Session() as sess:
  start_time = time.time()
  sess.run(tf.global_variables_initializer())
  sess.run(predictions)
  current_time = time.time()

 print(current_time-start_time)
