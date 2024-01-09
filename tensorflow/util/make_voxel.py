import sys
import numpy as np
sys.path.insert(0, '/home/qil15006/2020Summer/OCNN_Jun26_2020/octree/build/python')
import pyoctree
import numpy as np
from math import ceil, log2
import os

def read_points_file(filename):
    # read in .points file
    pts = pyoctree.Points()
    pts.read_points(filename)

    # parse .points file
    pts_num = pts.pts_num()
    coords = pts.points()
    normals = pts.normals()
    features = pts.features()

    # transform into numpy arrays
    coords = np.array(coords).reshape((pts_num, -1))
    normals = np.array(normals).reshape((pts_num, -1))
    features = np.array(features).reshape((pts_num, -1))
    features = np.concatenate((normals, features), axis = 1)

    return coords, features


def make_grid(coords, features, grid_resolution=1.0, radius=32.0):

    N, channel = features.shape
    grid_resolution = float(grid_resolution)
    radius = float(radius)

    box_size = ceil(2 * radius / grid_resolution)
    grid_coords = (coords + radius) / grid_resolution
    grid_coords = grid_coords.round().astype(int)

    in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
    grid = np.zeros((box_size, box_size, box_size, channel),
                    dtype=np.float32)
    count = np.zeros((box_size, box_size, box_size, channel),
                    dtype=np.float32)
    for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
        grid[x, y, z] += f
        count[x, y, z] += 1

    count[count==0] = 1
    grid = grid/count

    grid = np.swapaxes(grid, 0,3)
    return grid.tostring()

if __name__ == '__main__':
    coords, features = read_points_file('/media/data/data_share/pdbbind/refined_set_v2018/1a1e/1a1e_points_3.points')
    string = make_grid(coords, features, grid_resolution=1.0, radius=32.0)
    with open('grid_file.binary', 'br+') as f:
        f.write(bytearray(string))


# def make_grid_for_files(points_files, root_folder, save_folder, grid_resolution=1.0, radius=32.0):
#     with open(points_files, 'r') as f:
#         lines = f.readlines()
#
#         save_name_list = []
#         label_list = []
#         for line in lines:
#             file, label = line.split(' ')
#             id = file.split('/')[-2]
#             file_path = os.join(root_folder, file)
#
#             coords, features = read_points_file(file_path)
#             grid = make_grid(coords, features, grid_resolution, radius)
#             depth = int(log2(2*radius/grid_resolution))
#             save_name = '{}_{}.voxel'.format(id, depth)
#
#             grid.dump(save_folder + save_name)
#
#             save_name_list.append(save_name)
#             label_list.append(label)
#
#     with open('', 'w') as f:
#         for i in range(len(save_name_list)):
#             f.write('{} {}'.format(save_name_list[i], label_list[i]))



