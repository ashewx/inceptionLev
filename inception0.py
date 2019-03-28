# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests the graph placer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.grappler import cluster as clusters
from tensorflow.python.grappler import graph_placer
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
import tensorflow as tf

import settings
FLAGS = settings.FLAGS

import os
import re
import copy
from datetime import datetime
import time
from datasets import DataSet
import datasets

import model
import train_operation
import slim.slim
import numpy as np

cluster = tf.train.ClusterSpec({"local": ["172.23.10.2:2222", "172.23.10.3:2223", "172.23.10.4:2224", "172.23.10.6:2225"]})
server1 = tf.train.Server(cluster, job_name="local", task_index=0)

class GraphPlacerTest():

  @staticmethod
  def _buildInception():
    g = tf.Graph()

    with g.as_default():
      # global step number
      global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
      dataset = DataSet()

      # get training set
      print("The number of training images is: %d" % (dataset.cnt_samples(FLAGS.traincsv)))
      images, labels = dataset.csv_inputs(FLAGS.traincsv, FLAGS.batch_size, distorted=True)

      images_debug = datasets.debug(images)

        # get test set
        #test_cnt = dataset.cnt_samples(FLAGS.testcsv)
      test_cnt = 100

      images_test, labels_test = dataset.test_inputs(FLAGS.testcsv, test_cnt)


      input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

      num_classes = FLAGS.num_classes
      restore_logits = not FLAGS.fine_tune

        # inference
        # logits is tuple (logits, aux_liary_logits, predictions)
        # logits: output of final layer, auxliary_logits: output of hidden layer, softmax: predictions
      logits = model.inference(images, num_classes, for_training=True, restore_logits=restore_logits)

        # loss
      model.loss(logits, labels, batch_size=FLAGS.batch_size)
      losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)

        # Calculate the total loss for the current tower.
      regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
        #total_loss = tf.add_n(losses, name='total_loss')

        # Compute the moving average of all individual losses and the total loss.
      loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
      loss_averages_op = loss_averages.apply(losses + [total_loss])

        # for l in losses + [total_loss]:
        #     # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        #     # session. This helps the clarity of presentation on TensorBoard.
        #     loss_name = re.sub('%s_[0-9]*/' % model.TOWER_NAME, '', l.op.name)
        #     # Name each loss as '(raw)' and name the moving average version of the loss
        #     # as the original loss name.
        #     tf.scalar_summary(loss_name + ' (raw)', l)
        #     tf.scalar_summary(loss_name, loss_averages.average(l))

        # loss to calcurate gradients
        #
      with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
      tf.summary.scalar("loss", total_loss)

        # Reuse variables for the next tower.
        #tf.get_variable_scope().reuse_variables()

        # Retain the summaries from the final tower.
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

        # Retain the Batch Normalization updates operations only from the
        # final tower. Ideally, we should grab the updates from all towers
        # but these stats accumulate extremely fast so we can ignore the
        # other stats from the other towers without significant detriment.
      batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)

        # add input summaries
        # summaries.extend(input_summaries)

        # train_operation and operation summaries
      train_op = train_operation.train(total_loss, global_step, summaries, batchnorm_updates)

    train_op = g.get_collection_ref(tf_ops.GraphKeys.TRAIN_OP)
    train_op.append(train_op)
    return g

  @staticmethod
  def _buildCluster(num_cpus=1, num_gpus=1):
    devices = []
    if num_gpus > 0:
      device_properties = device_properties_pb2.DeviceProperties(
          type='GPU',
          vendor='NVidia',
          model='Tesla K40m',
          frequency=745, #745 MHZ
          num_cores= 2888, # CUDA Cores
          environment={'architecture': '5.2',
                       'cuda': '10000',
                       'cudnn': '7031'},
          num_registers=65536,
          l1_cache_size=65536, #64KB
          l2_cache_size=1572864, #1.5 MB
          shared_memory_size_per_multiprocessor=49152, #49152 bytes
          memory_size=12884901888, # 12GB
          bandwidth=288000000) #288 GBps)
      devices.append(
        device_properties_pb2.NamedDevice(
            properties=device_properties, name='/job:local/task:0/device:GPU:0'))
      devices.append(
        device_properties_pb2.NamedDevice(
            properties=device_properties, name='/job:local/task:1/device:GPU:0'))
      devices.append(
        device_properties_pb2.NamedDevice(
            properties=device_properties, name='/job:local/task:2/device:GPU:0'))
      devices.append(
        device_properties_pb2.NamedDevice(
            properties=device_properties, name='/job:local/task:3/device:GPU:0'))

    device_properties = device_properties_pb2.DeviceProperties(
        type='CPU',
        frequency=2399,
        num_cores=32,
        l1_cache_size=32768,
        l2_cache_size=262144,
        l3_cache_size=20971520)
    devices.append(
      device_properties_pb2.NamedDevice(
          properties=device_properties, name='/job:local/task:0/device:CPU:0'))
    devices.append(
      device_properties_pb2.NamedDevice(
          properties=device_properties, name='/job:local/task:1/device:CPU:0'))
    devices.append(
      device_properties_pb2.NamedDevice(
          properties=device_properties, name='/job:local/task:2/device:CPU:0'))
    devices.append(
      device_properties_pb2.NamedDevice(
          properties=device_properties, name='/job:local/task:3/device:CPU:0'))

    return clusters.Cluster(devices=devices)

  def testBuild(self):
    graph = GraphPlacerTest._buildInception()
    mg = meta_graph.create_meta_graph_def(graph=graph)
    #gcluster = cluster.Cluster(devices=None) # Automatically generates local machine cluster
    gcluster = GraphPlacerTest._buildCluster()
    print(gcluster.ListDevices()) # Print clust info
    # Spend 15 seconds trying to optimize the placement of the model. This
    # should give us enough time to exercise the code, but not enough to find
    # a good placement, so we'll just check for legality.
    placed_mg = graph_placer.PlaceGraph(mg, allotted_time=108000, cluster=gcluster, verbose=True)
    placed_g = placed_mg.graph_def;
    meta_graph.export_scoped_meta_graph(filename="./g/g.meta", graph_def=placed_g)
    # node in placed_mg.graph_def.node:
     # print(node)


if __name__ == '__main__':
  placer = GraphPlacerTest()
  placer.testBuild()
