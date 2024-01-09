import sys
import numpy as np
sys.path.insert(0, '../../octree/build/python')
import pyoctree
import numpy as np
import os




pro_lig_flag_pos = 18
kLeaf_flag = -1

# Can modify below
visualize = False  # True or False; If True, will generate vtk files, and can be visualized in Paraview
target_range = 32  # The radius of bounding box, default 32 generate a 64 * 64 * 64 Angstrom bounding box.
line_depth = 5     # Only plot octree division segment lines < line_depth.



def convert_key_to_xyz(key, depth):
    xyz = [0,0,0]
    for i in range(depth):
        for j in range(3):
            mask = 1 << (3 * i + 2 - j)
            xyz[j] |= (key & mask) >> (2 * i + 2 - j)
    return xyz

def octree_info_to_cubic_dic(depth, node_number, keys, children, features):
    cubic_dic = {}
    for node_idx_d in range(node_number):
        child = children[node_idx_d]
        if child == kLeaf_flag:
            continue
        key = keys[node_idx_d]
        xyz = convert_key_to_xyz(key, depth)
        [x,y,z] = xyz
        key_for_cubic_dic = '{}_{}_{}_{}'.format(depth, x,y,z)
        if features[node_idx_d][pro_lig_flag_pos] == 1:
            cubic_dic[key_for_cubic_dic] = 'Protein'
        elif features[node_idx_d][pro_lig_flag_pos] == 0:
            cubic_dic[key_for_cubic_dic] = 'Ligand'
        else:
            cubic_dic[key_for_cubic_dic] = 'Mixture'
    return cubic_dic








def vtk_from_cubic_dic(cubic_dic, total_depth, complex_id, differ=False,
                       shift = [0,0,0], save_folder = './results'):
    vtk_line_dic = {}
    for i in range(total_depth + 1):
        current_depth = total_depth - i
        base = target_range*2 / 2 ** current_depth
        vtk_dic = {}
        color = get_color_for_dep(current_depth + 1, differ)
        for key, value in cubic_dic.items():
            [depth, x, y, z] = [int(item) for item in key.split('_')]
            coord_x = x * base - target_range
            coord_y = y * base - target_range
            coord_z = z * base - target_range
            min_coord = [coord_x, coord_y, coord_z]
            max_coord = [coord_x + base, coord_y + base, coord_z + base]
            cubic = Cubic([min_coord, max_coord], key)
            cubic.set_color_new(value)
            vtk_dic = cubic.load_to_vtk_opt(vtk_dic)
            if current_depth < line_depth: # current_depth < depth for all.
                vtk_line_dic = cubic.write_oct_line_opt(vtk_line_dic, color)

        if not os.path.isdir('{}/{}/'.format(save_folder, complex_id)):
            os.makedirs('{}/{}/'.format(save_folder, complex_id))
        file_name = '{}/{}/{}_Octree_{}_{}'.format(save_folder, complex_id, complex_id, target_range, current_depth)
        vtk_writer(vtk_dic=vtk_dic, file_name = file_name, shift = shift)

        if current_depth > 0:
            upper_layer_dic = get_upper_layer_dic(cubic_dic)
            cubic_dic = upper_layer_dic

    color = get_color_for_dep(0, differ)
    vtk_line_dic = cubic.write_edge_line_opt(vtk_line_dic, color)

    pts, lines, colors = fill_vtk_lines(vtk_line_dic, shift)

    linesPolyData = vtk.vtkPolyData()
    linesPolyData.SetPoints(pts)
    linesPolyData.SetLines(lines)
    linesPolyData.GetCellData().SetScalars(colors)

    ShellVolmapper = vtk.vtkDataSetMapper()
    ShellVolmapper.SetScalarModeToUseCellData()
    ShellVolmapper.UseLookupTableScalarRangeOn()
    ShellVolmapper.SetInputData(linesPolyData)
    ShellVolactor = vtk.vtkActor()
    ShellVolactor.SetMapper(ShellVolmapper)

    modelwriter = vtk.vtkDataSetWriter()
    modelwriter.SetFileName('{}/{}/{}_segline_{}_{}.vtk'.format(save_folder, complex_id, complex_id, target_range, line_depth))
    modelwriter.SetInputData(ShellVolmapper.GetInput())
    modelwriter.Write()
    print('Save {}/{}/{}_segline_{}_{}.vtk'.format(save_folder, complex_id, complex_id, target_range, line_depth))


if __name__ == "__main__":
    # parse octree
    id = '1a1e'

    oct = pyoctree.Octree()
    print('Parse the 1a1e_points_10_2_000.octree just generated.')
    oct.read_octree("../data_example/pdbbind/{}/octree_folder/{}_points_10_2_000.octree".format(id, id))

    num_channel = oct.num_channel()
    print("Number of channel: {}".format(num_channel))

    depth = oct.depth()
    features = oct.features(depth)
    node_number = oct.num_nodes(depth)
    children = oct.children(depth)
    keys = oct.keys(depth)

    print("Depth: {}".format(depth))
    print("Number of nodes at depth {}:  {}".format(depth, node_number))

    features = np.array(features).reshape((num_channel, node_number))
    features = np.swapaxes(features, 0, 1)

    # for visualization
    if visualize:
        import vtk
        from utils import *
        print("Start Generate vtk files for Visualization in Paraview Software")
        cubic_dic = octree_info_to_cubic_dic(depth, node_number, keys, children, features)
        vtk_from_cubic_dic(cubic_dic, depth, id, differ = False, save_folder = './')

