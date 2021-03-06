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
from tensorflow.python.grappler import cluster
from tensorflow.python.grappler import graph_placer
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from nets import inception
import tensorflow as tf

class GraphPlacerTest():

  @staticmethod
  def _buildInception():
    g = tf.Graph()
    with g.as_default():
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

    train_op = g.get_collection_ref(tf_ops.GraphKeys.TRAIN_OP)
    train_op.append(predictions)
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
          bandwidth=288000000) #288 GBps
      for i in range(num_gpus):
        devices.append(
            device_properties_pb2.NamedDevice(
                properties=device_properties, name='/GPU:' + str(i)))

    assert num_cpus > 0
    device_properties = device_properties_pb2.DeviceProperties(
        type='CPU',
        frequency=2399,
        num_cores=32,
        l1_cache_size=32768,
        l2_cache_size=262144,
        l3_cache_size=20971520)
    for i in range(num_cpus):
      devices.append(
          device_properties_pb2.NamedDevice(
              properties=device_properties, name='/CPU:' + str(i)))

    return cluster.Cluster(devices=devices)

  def testBuild(self):
    graph = GraphPlacerTest._buildInception()
    mg = meta_graph.create_meta_graph_def(graph=graph)
    #gcluster = cluster.Cluster(devices=None) # Automatically generates local machine cluster
    gcluster = GraphPlacerTest._buildCluster()
    print(gcluster.ListDevices()) # Print clust info
    # Spend 15 seconds trying to optimize the placement of the model. This
    # should give us enough time to exercise the code, but not enough to find
    # a good placement, so we'll just check for legality.
    placed_mg = graph_placer.PlaceGraph(mg, allotted_time=300, cluster=gcluster, verbose=True)
    placed_g = placed_mg.graph_def;
    meta_graph.export_scoped_meta_graph(filename="./g/g.meta", graph_def=placed_g)
    # node in placed_mg.graph_def.node:
     # print(node)


if __name__ == '__main__':
  placer = GraphPlacerTest()
  placer.testBuild()
