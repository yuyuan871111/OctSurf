import tensorflow as tf
from network_cls import network_ocnn, network_resnet, network_cnn_grids, network_resnet_grids
#from network_unet import network_unet
#from network_hrnet import HRNet
# from network_unet_scannet import network_unet34

def cls_network(octree, flags, training, reuse=False):
  if flags.name.lower() == 'ocnn':
    return network_ocnn(octree, flags, training, reuse)
  elif flags.name.lower() == 'resnet':
    return network_resnet(octree, flags, training, reuse)
  elif flags.name.lower() == 'cnn_grids':
    return network_cnn_grids(octree, flags, training, reuse)
  elif flags.name.lower() == 'resnet_grids':
    return network_resnet_grids(octree, flags, training, reuse)
  else:
    print('Error, no network: ' + flags.name)

    
