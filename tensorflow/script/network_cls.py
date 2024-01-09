import tensorflow as tf
from ocnn import *


# octree-based resnet55
def network_resnet(octree, flags, training=True, reuse=None):
  depth = flags.depth
  channels = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 8, 8]
  with tf.variable_scope("ocnn_resnet", reuse=reuse):
    data = octree_property(octree, property_name="feature", dtype=tf.float32, 
                           depth=depth, channel=flags.channel)
    data = tf.reshape(data, [1, flags.channel, -1, 1])

    with tf.variable_scope("conv1"):
      data = octree_conv_bn_relu(data, octree, depth, channels[depth], training)

    for d in range(depth, 2, -1):
      for i in range(0, flags.resblock_num):
        with tf.variable_scope('resblock_%d_%d' % (d, i)):
          data = octree_resblock(data, octree, d, channels[d], 1, training)
      with tf.variable_scope('max_pool_%d' % d):
        data, _ = octree_max_pool(data, octree, d)

    with tf.variable_scope("global_average"):
      data = octree_full_voxel(data, depth=2)
      data = tf.reduce_mean(data, 2)
    
    if flags.dropout[0]:
      data = tf.layers.dropout(data, rate=0.5, training=training)

    with tf.variable_scope("fc2"):
      logit = dense(data, flags.nout, use_bias=True)

  return logit


def network_resnet_grids(grids, flags, training=True, reuse=None):
  depth = flags.depth
  channels = [2048, 1024, 512, 256, 128, 64, 32, 16, 8] #[2048, 1024, 512, 256, 256, 128, 64, 32, 16]
  with tf.variable_scope("ocnn_resnet", reuse=reuse):
    with tf.variable_scope("conv1"):
      data = conv3d_bn_relu(grids, channels[depth], 3, 1, training)

    for d in range(depth, 2, -1):
      for i in range(0, flags.resblock_num):
        with tf.variable_scope('resblock_%d_%d' % (d, i)):
          data = grids_resblock(data, channels[d], 1, training)
      with tf.variable_scope('max_pool_%d' % d):
        data = max_pool3d(data)

    with tf.variable_scope("global_average"):
      # data = octree_full_voxel(data, depth=2)
      # data = tf.reduce_mean(data, 2)
      data = tf.reduce_mean(data, [2,3,4])

    if flags.dropout[0]:
      data = tf.layers.dropout(data, rate=0.5, training=training)

    with tf.variable_scope("fc2"):
      logit = dense(data, flags.nout, use_bias=True)

  return logit


# the ocnn in the paper
def network_ocnn(octree, flags, training=True, reuse=None):
  depth = flags.depth
  channels = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 8, 8] #[512, 256, 128, 64, 32, 16, 8, 4, 2] #[2048, 1024, 512, 256, 128, 64, 32, 32, 32]
  with tf.variable_scope("ocnn", reuse=reuse):
    data = octree_property(octree, property_name="feature", dtype=tf.float32,
                           depth=depth, channel=flags.channel)
    data = tf.reshape(data, [1, flags.channel, -1, 1])

    for d in range(depth, 2, -1):
      with tf.variable_scope('depth_%d' % d):
        data = octree_conv_bn_relu(data, octree, d, channels[d], training)
        data, _ = octree_max_pool(data, octree, d)

    with tf.variable_scope("full_voxel"):
      data = octree_full_voxel(data, depth=2)
      data = tf.layers.dropout(data, rate=0.5, training=training)

    with tf.variable_scope("fc1"):
      data = fc_bn_relu(data, channels[2], training=training)
      data = tf.layers.dropout(data, rate=0.5, training=training)

    with tf.variable_scope("fc2"):
      logit = dense(data, flags.nout, use_bias=True)

  return logit


# vanilla cnn, takes grids as input. has same structure with the ocnn in the paper
def network_cnn_grids(grids, flags, training=True, reuse=None):
  depth = flags.depth
  channels = [2048, 1024, 512, 256, 128, 64, 32, 16, 8] #[2048, 1024, 512, 256, 128, 64, 32, 32, 32]
  with tf.variable_scope("ocnn", reuse=reuse):
    for d in range(depth, 2, -1):
      with tf.variable_scope('depth_%d' % d):
        grids = conv3d_bn_relu(grids, channels[d], 3, 1, training)
        grids = max_pool3d(grids)

    grids = tf.layers.dropout(grids, rate=0.5, training=training)
    with tf.variable_scope("fc1"):
      data = fc_bn_relu(grids, channels[2], training=training)
      data = tf.layers.dropout(data, rate=0.5, training=training)

    with tf.variable_scope("fc2"):
      logit = dense(data, flags.nout, use_bias=True)

  return logit

def cls_network(octree, flags, training, reuse=False):
  if flags.name.lower() == 'ocnn':
    return network_ocnn(octree, flags, training, reuse)
  elif flags.name.lower() == 'resnet':
    return network_resnet(octree, flags, training, reuse)
  else:
    print('Error, no network: ' + flags.name)
