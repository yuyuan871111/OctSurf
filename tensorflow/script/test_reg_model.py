import tensorflow as tf

from config import parse_args
from tfsolver import TFSolver
from dataset import DatasetFactory
from network_factory import cls_network

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# configs
FLAGS = parse_args()

def get_output(dataset='test', training=False, reuse=False, task = 'class'):
  flags_data = FLAGS.DATA.train if dataset == 'train' else FLAGS.DATA.test
  y_type = 'int' if task == 'class' else 'float'
  octree, label = DatasetFactory(flags_data, y_type=y_type)()
  logit = cls_network(octree, FLAGS.MODEL, training, reuse)
  return [logit, label]

def check_input(dataset='test', training=False, reuse=False, task = 'class'):
  flags_data = FLAGS.DATA.train if dataset == 'train' else FLAGS.DATA.test
  y_type = 'int' if task == 'class' else 'float'
  octree, label = DatasetFactory(flags_data, y_type=y_type)()
  return octree, label

# run the experiments
if __name__ == '__main__':
  # solver = TFSolver(FLAGS.SOLVER, check_input)
  # solver.check_grids()
  #print(FLAGS.SOLVER)
  solver = TFSolver(FLAGS, get_output)
  test_size_dic = {'CASF': 285, 'general_2019': 1146, 'refined_2019': 394, 'decoy':1460, 'training_15241':15241, 'training_15235': 15235}
  solver.test_ave(test_size=test_size_dic[FLAGS.DATA.name])

  #solver.test_ave()
