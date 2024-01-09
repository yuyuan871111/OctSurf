import re
import math
import numpy as np
import random
import os
import pandas as pd



def filter(folders, feature_need_sdf = []):
    results = []
    for folder in folders:
        if len(folder) != 4:
            continue
        if folder in feature_need_sdf:
            continue
        if os.path.isfile(folder):
            print(folder)
            continue
        results.append(folder)
    return results

def check_duplicate(folder1, folder2):
    results = []
    for folder in folder1:
        if folder in folder2:
            # print('folder1 subfolder {} also in folder2'.format(folder))
            continue
        results.append(folder)
    for folder in folder2:
        if folder in folder1:
            # print('folder2 subfolder {} also in folder1'.format(folder))
            pass
    return results

# use to check if all the compelx show in PL_data include pdb and mol2 file. Answer is NO.
# some data has the affinity data, but do not have pdb and mol2.
def check_dic_id(dic, general_folder, refine_folder, core_folder):
    for key in dic:
        if key not in general_folder and key not in refine_folder and key not in core_folder:
            print(key)


def read_affinity(file_name, mode):
    """
    read affinity data from pdbbind index file, refers to log Kd/Ki
    Benefit is: in general set, there are some <,>,~ in Kd/Ki, directly use log Kd/Ki might be better.
    """
    record_dic = {}
    with open(file_name, 'r') as data_fid:
        for line in data_fid:
            if '#' in line or '===' in line or len(line) == 0:
                continue
            line = re.sub('\s+', ' ', line).strip()
            contents = line.split(' ')
            id = contents[0]
            affinity = float(contents[3])
            if mode == 'reg':
                record_dic[id] = affinity
            else:
                if affinity > 6 + math.log10(5):
                    record_dic[id] = 1
                elif affinity < 6 - math.log10(5):
                    record_dic[id] = 0
                else:
                    print('skip {}, affinity is {}.'.format(id, affinity))
    return record_dic

# train/test/val split, with all pdbbind data.
def write_octree_list(general_dic, core_folders, refined_folders, depth = 8, mode = 'reg', view_num = 24):
    # test only include core set
    test_file_name = root_folder + 'octree_list_test_{}_{}.txt'.format(depth, mode)
    test_affinity = []
    with open(test_file_name, 'w') as f:
        for id in core_folders:
            if id not in general_dic:
                continue
            for v in range(0, view_num):
                path = 'coreset/{0}/octree_folder/{1}_points_{2}_2_{3:03d}.octree'.format(id, id, depth, v)
                line = '{} {}\n'.format(path, general_dic[id])
                f.write(line)
            test_affinity.append(general_dic[id])

    # validation is part of refined set
    random.seed(2020)
    val_id = random.sample(refined_folders, k=600)
    val_file_name = root_folder + 'octree_list_val_{}_{}.txt'.format(depth, mode)
    val_affinity = []
    with open(val_file_name, 'w') as f:
        for id in refined_folders:
            if id not in val_id:
                continue
            if id not in general_dic:
                print('id {} not in general_dic'.format(id))
                continue
            for v in range(0, view_num):
                path = 'refined-set/{0}/octree_folder/{1}_points_{2}_2_{3:03d}.octree'.format(id, id, depth, v)
                line = '{} {}\n'.format(path, general_dic[id])
                f.write(line)
            val_affinity.append(general_dic[id])

    train_file_name = root_folder + 'octree_list_train_{}_{}.txt'.format(depth, mode)
    train_affinity = []
    with open(train_file_name, 'w') as f:
        train_total = general_folders + refined_folders
        for id in train_total:
            if id in val_id:
                continue
            if id not in general_dic:
                continue
            for v in range(0, view_num):
                if id in refined_folders:
                    path = 'refined-set/{0}/octree_folder/{1}_points_{2}_2_{3:03d}.octree'.format(id, id, depth, v)
                else:
                    path = 'v2018-other-PL/{0}/octree_folder/{1}_points_{2}_2_{3:03d}.octree'.format(id, id, depth, v)
                line = '{} {}\n'.format(path, general_dic[id])
                f.write(line)
            train_affinity.append(general_dic[id])

    train_affinity = np.array(train_affinity)
    val_affinity = np.array(val_affinity)
    test_affinity = np.array(test_affinity)
    print('train_affinity size: {}, mean: {}, std: {}'.format(train_affinity.shape[0], np.mean(train_affinity), np.std(train_affinity)))
    print('val_affinity size: {}, mean: {}, std: {}'.format(val_affinity.shape[0], np.mean(val_affinity), np.std(val_affinity)))
    print('test_affinity size: {}, mean: {}, std: {}'.format(test_affinity.shape[0], np.mean(test_affinity), np.std(test_affinity)))

def write_points_list(general_dic, core_folders, refined_folders, mode = 'reg', density = None):
    # test only include core set
    test_file_name = root_folder + 'points_list_test_{}.txt'.format(mode)
    test_affinity = []
    with open(test_file_name, 'w') as f:
        for id in core_folders:
            if id not in general_dic:
                continue
            if density is None:
                path = 'coreset/{0}/{1}_points.points'.format(id, id)
            else:
                path = 'coreset/{0}/{1}_points_{2}.points'.format(id, id, density)
            # print(path)
            line = '{} {}\n'.format(path, general_dic[id])
            f.write(line)
            test_affinity.append(general_dic[id])

    # validation is part of refined set
    random.seed(2020)
    # val_id = random.choices(refined_folders, k=1000)
    val_id = random.sample(refined_folders, k=600)
    val_file_name = root_folder + 'points_list_val_{}.txt'.format(mode)
    val_affinity = []
    with open(val_file_name, 'w') as f:
        for id in refined_folders:
            if id not in val_id or id not in general_dic:
                continue
            if density is None:
                path = 'refined-set/{0}/{1}_points.points'.format(id, id)
            else:
                path = 'refined-set/{0}/{1}_points_{2}.points'.format(id, id, density)

            line = '{} {}\n'.format(path, general_dic[id])
            f.write(line)
            val_affinity.append(general_dic[id])



    train_file_name = root_folder + 'points_list_train_{}.txt'.format(mode)
    train_affinity = []
    with open(train_file_name, 'w') as f:
        train_total = general_folders + refined_folders
        for id in train_total:
            if id in val_id:
                continue
            if id not in general_dic:
                continue
            if id in refined_folders:
                if density is None:
                    path = 'refined-set/{0}/{1}_points.points'.format(id, id)
                else:
                    path = 'refined-set/{0}/{1}_points_{2}.points'.format(id, id, density)
            else:
                if density is None:
                    path = 'v2018-other-PL/{0}/{1}_points.points'.format(id, id)
                else:
                    path = 'v2018-other-PL/{0}/{1}_points_{2}.points'.format(id, id, density)

            # print(path)
            line = '{} {}\n'.format(path, general_dic[id])
            f.write(line)
            train_affinity.append(general_dic[id])

    train_affinity = np.array(train_affinity)
    val_affinity = np.array(val_affinity)
    test_affinity = np.array(test_affinity)
    print('train_affinity size: {}, mean: {}, std: {}'.format(train_affinity.shape[0], np.mean(train_affinity), np.std(train_affinity)))
    print('val_affinity size: {}, mean: {}, std: {}'.format(val_affinity.shape[0], np.mean(val_affinity), np.std(val_affinity)))
    print('test_affinity size: {}, mean: {}, std: {}'.format(test_affinity.shape[0], np.mean(test_affinity), np.std(test_affinity)))

if __name__ == "__main__":
    root_folder = '../data_folder/'

    mode = 'reg'

    general_file = root_folder + 'v2018-other-PL/index/INDEX_general_PL_data.2018'

    general_folders = os.listdir(root_folder + 'v2018-other-PL')
    refined_folders = os.listdir(root_folder + 'refined-set')
    core_folders = os.listdir(root_folder + 'coreset')

    general_folders = filter(general_folders)
    refined_folders = filter(refined_folders)
    core_folders = filter(core_folders)

    print(len(general_folders), len(refined_folders), len(core_folders))
    print('Check general and core')
    general_folders = check_duplicate(general_folders, core_folders)
    print('Check general and refine')
    general_folders = check_duplicate(general_folders, refined_folders)
    print('Check refine and core')
    refined_folders = check_duplicate(refined_folders, core_folders)
    print(len(general_folders), len(refined_folders), len(core_folders))

    core_id_list = core_folders
    general_dic = read_affinity(general_file, mode = mode)
    print(len(general_dic))

    # check_dic_id(general_dic, general_folders, refined_folders, core_folders)

    # write_octree_list(general_dic, core_folders, refined_folders, depth = 5, mode  = mode, view_num = 24)
    write_points_list(general_dic, core_folders, refined_folders, mode=mode, density = None)
