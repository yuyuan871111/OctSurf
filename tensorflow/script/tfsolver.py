import os
from ocnn import *
from tqdm import tqdm
import tensorflow as tf
from learning_rate import LRFactory
from tensorflow.python.client import timeline

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class TFSolver:
  def __init__(self, flags, compute_graph=None, build_solver=build_solver):
    self.flags = flags.SOLVER
    self.graph = compute_graph
    self.build_solver = build_solver

  def build_train_graph(self):
    gpu_num = len(self.flags.gpu)
    train_params = {'dataset': 'train', 'training': True,  'reuse': False}
    test_params  = {'dataset': 'test',  'training': False, 'reuse': True}
    if gpu_num > 1:
      train_params['gpu_num'] = gpu_num
      test_params['gpu_num']  = gpu_num
      
    self.train_tensors, train_names = self.graph(**train_params)
    self.test_tensors, self.test_names = self.graph(**test_params)
    
    self.total_loss = self.train_tensors[train_names.index('total_loss')]
    solver_param = [self.total_loss, LRFactory(self.flags)]
    if gpu_num > 1:
      solver_param.append(gpu_num)
    self.train_op, lr = self.build_solver(*solver_param)  # qq: self.train_op is the optimizer.

    if gpu_num > 1: # average the tensors from different gpus for summaries
      with tf.device('/cpu:0'):
        self.train_tensors = average_tensors(self.train_tensors)
        self.test_tensors = average_tensors(self.test_tensors)        
    self.summaries(train_names + ['lr'], self.train_tensors + [lr,], self.test_names)

  def summaries(self, train_names, train_tensors, test_names):
    self.summ_train = summary_train(train_names, train_tensors)
    self.summ_test, self.summ_holder = summary_test(test_names)
    self.summ2txt(test_names, 'step', 'w')

  def summ2txt(self, values, step, flag='a'):
    test_summ = os.path.join(self.flags.logdir, 'test_summaries.csv')
    with open(test_summ, flag) as fid:      
      msg = '{}'.format(step)
      for v in values:
        msg += ', {}'.format(v)
      fid.write(msg + '\n')

  def build_test_graph(self):
    gpu_num = len(self.flags.gpu)
    test_params  = {'dataset': 'test',  'training': False, 'reuse': False}
    if gpu_num > 1: test_params['gpu_num'] = gpu_num
    self.test_tensors, self.test_names = self.graph(**test_params)
    if gpu_num > 1: # average the tensors from different gpus
      with tf.device('/cpu:0'):
        self.test_tensors = average_tensors(self.test_tensors)        

  def restore(self, sess, ckpt):
    print('Load checkpoint: ' + ckpt)
    self.tf_saver.restore(sess, ckpt)

  def initialize(self, sess):
    sess.run(tf.global_variables_initializer())

  def run_k_iterations(self, sess, k, tensors):
    num = len(tensors)
    avg_results = [0] * num
    for _ in range(k):
      iter_results = sess.run(tensors)
      for j in range(num):
        avg_results[j] += iter_results[j]
    
    for j in range(num):
      avg_results[j] /= k
    avg_results = self.result_callback(avg_results)
    return avg_results

  def result_callback(self, avg_results):
    return avg_results # calc some metrics, such as IoU, based on the graph output

  def qq_set_update_after_k_round(self):
    self.opt = tf.train.AdamOptimizer(learning_rate=0.1)
    tvs = tf.trainable_variables()
    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
    self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
    gvs = self.opt.compute_gradients(self.total_loss, tvs)
    self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
    self.train_step = self.opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

  def train(self):
    # build the computation graph
    self.build_train_graph()

    # qq: add
    # self.qq_set_update_after_k_round()

    # checkpoint
    start_iter = 1
    self.tf_saver = tf.train.Saver(max_to_keep=self.flags.ckpt_num)
    ckpt_path = os.path.join(self.flags.logdir, 'model')
    if self.flags.ckpt:        # restore from the provided checkpoint
      ckpt = self.flags.ckpt  
    else:                      # restore from the breaking point
      ckpt = tf.train.latest_checkpoint(ckpt_path)    
      if ckpt: start_iter = int(ckpt[ckpt.find("iter")+5:-5]) + 1

    # session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
      summary_writer = tf.summary.FileWriter(self.flags.logdir, sess.graph)

      print('Initialize ...')
      self.initialize(sess)
      if ckpt: self.restore(sess, ckpt)

      print('Start training ...')
      for i in tqdm(range(start_iter, self.flags.max_iter + 1), ncols=80):
        # training
        # qq: revise the training, to update gradients after multiple iterations
        # first 2 lines are original code.
        summary, _ = sess.run([self.summ_train, self.train_op])
        summary_writer.add_summary(summary, i)
        #if i == 0:
        #  sess.run(self.zero_ops)
        #if i % 10 !=0 or i ==0:
        #  sess.run(self.accum_ops)
        #else:
        #  sess.run(self.accum_ops)
        #  sess.run(self.train_step)
        #  sess.run(self.zero_ops)
        # qq: end revise

        # testing
        if i % self.flags.test_every_iter == 0:
          # run testing average
          avg_test = self.run_k_iterations(sess, self.flags.test_iter, self.test_tensors)

          # run testing summary
          summary = sess.run(self.summ_test, 
                             feed_dict=dict(zip(self.summ_holder, avg_test)))
          summary_writer.add_summary(summary, i)
          self.summ2txt(avg_test, i)

          # save session
          ckpt_name = os.path.join(ckpt_path, 'iter_%06d.ckpt' % i)
          self.tf_saver.save(sess, ckpt_name, write_meta_graph = False)
          
      print('Training done!')

  def timeline(self):
    # build the computation graph
    self.build_train_graph()

    # session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    timeline_skip, timeline_iter = 100, 2
    with tf.Session(config=config) as sess:
      summary_writer = tf.summary.FileWriter(self.flags.logdir, sess.graph)
      print('Initialize ...')
      self.initialize(sess)

      print('Start profiling ...')
      for i in tqdm(range(0, timeline_skip + timeline_iter), ncols=80):
        if i < timeline_skip:
          summary, _ = sess.run([self.summ_train, self.train_op])
        else:
          summary, _ = sess.run([self.summ_train, self.train_op], 
                                options=options, run_metadata=run_metadata)
          if (i == timeline_skip + timeline_iter - 1):
            # summary_writer.add_run_metadata(run_metadata, 'step_%d'%i, i)
            # write timeline to a json file
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(os.path.join(self.flags.logdir, 'timeline.json'), 'w') as f:
              f.write(chrome_trace)
        summary_writer.add_summary(summary, i)
      print('Profiling done!')

  def param_stats(self):
    # build the computation graph
    self.build_train_graph()

    # get variables
    train_vars = tf.trainable_variables()

    # print
    total_num = 0
    for idx, v in enumerate(train_vars):
      shape = v.get_shape()
      shape_str = '; '.join([str(s) for s in shape])
      shape_num = shape.num_elements()
      print("{:3}, {:15}, [{}], {}".format(idx, v.name, shape_str, shape_num))
      total_num += shape_num
    print('Total trainable parameters: {}'.format(total_num))

  def test(self):
    # build graph
    self.build_test_graph()
    #self.qq_set_update_after_k_round()

    # checkpoint
    assert(self.flags.ckpt)   # the self.flags.ckpt should be provided
    tf_saver = tf.train.Saver(max_to_keep=10)

    # start
    num_tensors = len(self.test_tensors)
    avg_test = [0] * num_tensors
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      summary_writer = tf.summary.FileWriter(self.flags.logdir, sess.graph)
      self.summ2txt(self.test_names, 'batch')

      # restore and initialize
      self.initialize(sess)
      print('Restore from checkpoint: %s' % self.flags.ckpt)
      tf_saver.restore(sess, self.flags.ckpt)

      print('Start testing ...')
      for i in range(0, self.flags.test_iter):
        iter_test_result = sess.run(self.test_tensors)
        iter_test_result = self.result_callback(iter_test_result)
        # run testing average
        for j in range(num_tensors):
          avg_test[j] += iter_test_result[j]
        # print the results
        reports = 'batch: %04d; ' % i
        for j in range(num_tensors):
          reports += '%s: %0.4f; ' % (self.test_names[j], iter_test_result[j])
        print(reports)
        self.summ2txt(iter_test_result, i)

    # Final testing results
    for j in range(num_tensors):
      avg_test[j] /= self.flags.test_iter
    avg_test = self.result_callback(avg_test)
    # print the results
    print('Testing done!\n')
    reports = 'ALL: %04d; ' % self.flags.test_iter
    for j in range(num_tensors):
      reports += '%s: %0.4f; ' % (self.test_names[j], avg_test[j])
    print(reports)
    self.summ2txt(avg_test, 'ALL')

  def test_ave(self, test_size = 285):
    # build graph
    import numpy as np
    outputs = self.graph('test', training=False, reuse=False, task = self.flags['task'])

    # checkpoint
    assert(self.flags.ckpt)   # the self.flags.ckpt should be provided
    tf_saver = tf.train.Saver(max_to_keep=10)

    # start
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

      # restore and initialize
      self.initialize(sess)
      tf_saver.restore(sess, self.flags.ckpt)

      print('Start testing ...')
      test_logits = []   # qq add to record all logits
      test_labels = []
      for i in range(0, self.flags.test_iter):
        iter_test_result = sess.run(outputs)
        test_logits.append(iter_test_result[0])
        test_labels.append(iter_test_result[1])
       # print(iter_test_result[0], iter_test_result[1])

    all_preds = np.array(test_logits).reshape(test_size, -1)
    all_labels = np.array(test_labels).reshape(test_size, -1)

    all_labels = all_labels.reshape(-1, test_size)
    all_preds = all_preds.reshape(-1, test_size)

    all_labels_mean = all_labels.mean(axis=0)
    all_preds_mean = all_preds.mean(axis=0)

    #all_labels = all_labels.reshape(test_size,-1)
    #all_preds = all_preds.reshape(test_size, -1)
    #all_labels_mean = all_labels.mean(axis=1)
    #all_preds_mean = all_preds.mean(axis=1)
    # if abs(all_labels.std(axis=0).sum()) < 1e-4:
    #   print(all_labels.std(axis=0))
    #   print(all_labels)

    print(all_labels_mean)
    #print(all_preds_mean)
    import pandas as pd
    df = pd.DataFrame({'label': all_labels_mean, 'pred': all_preds_mean})
    df.to_csv('pred_label.csv')

    def report_reg_metrics(all_labels, all_preds):
      from scipy.stats import pearsonr, spearmanr, kendalltau
      from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
      rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
      mae = mean_absolute_error(all_labels, all_preds)
      pearson_r = pearsonr(all_labels, all_preds)[0]
      pearson_r2 = pearson_r **2
      r2 = r2_score(all_labels, all_preds)
      spearman_r = spearmanr(all_labels, all_preds)[0]
      kendall_tau = kendalltau(all_labels, all_preds)[0]
      print('rmse: {}, mae: {}, pearsonr: {}, pearsonr2: {}, r2: {}, spearman: {}, kendall: {}'.format(rmse, mae, pearson_r, pearson_r2, r2, spearman_r, kendall_tau))
      pass

    def report_scatter(all_labels, all_probs):
      import matplotlib.pyplot as plt
      import seaborn as sns
      plt.figure()
      # sns.scatterplot(all_labels, all_probs)
      sns.regplot(all_labels, all_probs)
      plt.show()

    def report_cluster_corr(all_labels, all_probs):
      import pandas as pd
      clusters = pd.read_excel(r'./predicted clusters.xlsx', engine='openpyxl')
      clusters = clusters[:285]
      clusters.at[15, 'PDB code'] = "1E66"
      clusters.at[171, 'PDB code'] = "3E92"
      clusters.at[172, 'PDB code'] = "3E93"

      with open(r'./points_list_test_reg.txt', "r") as fid:
        pred_list = []
        for line in fid.readlines():
          pred_list.append(line.strip('\n'))

      #pred_values = []
      #for i in clusters["PDB code"]:  # loops through each protein for the respective PDB codes
      #  for j,item in enumerate(pred_list):  # loops through each line of the prediction value text file
      #    item = item.upper()  # changes each line to uppercase because the txt file PDB codes are lowercase and we need them in uppercase
      #    if item[18:22] == i:  # j[18:22] is the PDB code for the pred value. This matches the PDB codes of the prediction and true values of proteins
            # x = item[44:]  # j[44:] is the prediction value
            # x = float(x)  # turns predicion value from string to float
      #      x = all_probs[j]
      #      pred_values.append(x)
      #clusters["pred"] = pred_values  # adds a column in the cluster dataframe for predicted values
      #print(clusters)

      #corr = clusters.groupby('Cluster ID')[['Binding constant','pred']].corr().iloc[0::2,-1]
      import matplotlib.pyplot as plt
      import seaborn as sns
      #plt.figure()
      #sns.distplot(corr, kde=False)
      #plt.xlabel('Correlation')
      #plt.ylabel('Count')
      #plt.savefig('./cluster.png')

      #mean_df = clusters.groupby('Cluster ID').mean()
      #plt.figure()
      #sns.regplot(mean_df['Binding constant'], mean_df['pred'])
      #plt.xlabel('Cluster Mean Label')
      #plt.ylabel('Cluster Mean Pred')
      #plt.savefig('./cluster_mean.png')
      #print('Inter cluster corr: {}'.format(np.corrcoef(mean_df['Binding constant'], mean_df['pred'])[0,1]))

      print("Double Verify")
      cluster_list = []
      id_list = []
      clusters = clusters.set_index('PDB code')
      for j, item in enumerate(pred_list):
        item = item.upper()
        id = item[18:22]
        cluster_list.append(clusters.loc[id, 'Cluster ID'])
        id_list.append(id)
        print(id, all_labels[j], all_probs[j], clusters.loc[id, 'Binding constant'])        

      new_df = pd.DataFrame({"pred": all_probs, "label": all_labels, "cluster": cluster_list, "id": id_list})
      corr = new_df.groupby('cluster')[['label', 'pred']].corr().iloc[0::2, -1]
      plt.figure()
      sns.distplot(corr, kde=False)
      plt.xlabel('Correlation')
      plt.ylabel('Count')
      plt.savefig('./cluster.png')
      print('Corr: {}'.format(list(np.array(corr))))
      #print(new_df)
      new_df.to_csv('result.csv')

      mean_df = new_df.groupby('cluster').mean()
      plt.figure()
      sns.regplot(mean_df['label'], mean_df['pred'])
      plt.xlabel('Cluster Mean Label')
      plt.ylabel('Cluster Mean Pred')
      plt.savefig('./cluster_mean.png')
      print('Inter cluster corr: {}'.format(np.corrcoef(mean_df['label'], mean_df['pred'])[0,1]))

      print("<0: ", (corr<0).sum())
      print(">0.8: ", (corr>=0.8).sum())
      print(">0.9: ", (corr>=0.9).sum())
      print('min; ', corr.min())
    
    report_reg_metrics(all_labels_mean, all_preds_mean)
    report_scatter(all_labels_mean, all_preds_mean)
    report_cluster_corr(all_labels_mean, all_preds_mean)

  def check_grids(self, test_size = 285):
    # build graph
    import numpy as np
    outputs = self.graph('test', training=False, reuse=False, task = self.flags['task'])

    # checkpoint
    # tf_saver = tf.train.Saver(max_to_keep=10)

    # start
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

      # restore and initialize
      self.initialize(sess)
      # tf_saver.restore(sess, self.flags.ckpt)

      print('Start testing ...')
      for i in range(0, self.flags.test_iter):
        iter_test_result = sess.run(outputs)
        grids = iter_test_result[0]
        labels = iter_test_result[1]
        print(grids.shape)


  def run(self):
    eval('self.{}()'.format(self.flags.run))

