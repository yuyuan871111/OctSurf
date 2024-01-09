# utils for visualization


import numpy as np
import vtk
import time

def read_xyz_file(file_name):
    with open('{}'.format(file_name), 'r') as data_fid:
        points, x, y, z = [], [], [], []
        for row in data_fid.readlines():
            row = row.replace('\n', '')
            numbers = row.split(' ')
            try:
                point_coord = [float(item) for item in numbers[0:3]]
                point_x = point_coord[0]
                point_y = point_coord[1]
                point_z = point_coord[2]
                points.append(point_coord)
                x.append(point_x)
                y.append(point_y)
                z.append(point_z)
            except ValueError:
                print('ValueError: {}, row {}'.format(file_name, row))
            except IndexError:
                print('IndexError: {}, row {}'.format(file_name, row))
            else:
                pass
        return np.array(x), np.array(y), np.array(z)

def read_feature_file(file_name):
    with open('{}'.format(file_name), 'r') as data_fid:
        features = []
        for row in data_fid.readlines():
            row = row.replace('\n', '')
            row_features = row.split(' ')
            try:
                row_features = [float(item) for item in row_features]
                features.append(row_features)
            except ValueError:
                print('ValueError: {}, row {}'.format(file_name, row))
            except IndexError:
                print('IndexError: {}, row {}'.format(file_name, row))
            else:
                pass
        return features

class Point(object):
    def __init__(self, coord):
        self.coord = coord
        self.string = '{}_{}_{}'.format(coord[0], coord[1], coord[2])
        self.order = None

    def get_coord(self):
        return self.coord

class Cubic(object):
    """
    A class that describe a cubic in 3D space.
    Can detect if this cubic contain points, then spilt into 8 child cubics and plot.
    """
    def __init__(self, coord, id):
        self.id = id
        min_coord = coord[0]
        max_coord = coord[1]
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.min_x = min_coord[0]
        self.min_y = min_coord[1]
        self.min_z = min_coord[2]
        self.max_x = max_coord[0]
        self.max_y = max_coord[1]
        self.max_z = max_coord[2]

        self.dot1 = [self.min_x, self.min_y, self.min_z]
        self.dot2 = [self.min_x, self.min_y, self.max_z]
        self.dot3 = [self.min_x, self.max_y, self.min_z]
        self.dot4 = [self.min_x, self.max_y, self.max_z]
        self.dot5 = [self.max_x, self.min_y, self.min_z]
        self.dot6 = [self.max_x, self.min_y, self.max_z]
        self.dot7 = [self.max_x, self.max_y, self.min_z]
        self.dot8 = [self.max_x, self.max_y, self.max_z]

        self.resolution = self.max_z - self.min_z
        self.mid_point = [(self.min_x + self.max_x)/2, (self.min_y + self.max_y)/2, (self.min_z + self.max_z)/2]

    def set_color_new(self, value):
        if value == 'Protein':
            self.color = [.4, .4, 1, 1]
            self.color_scale = 11
        elif value == 'Ligand':
            self.color = [.4, 1, .4, 1]
            self.color_scale = 12
        elif value == 'Mixture':
            self.color = [.4, 1, 1, 1]
            self.color_scale = 13
        elif value == 'Empty':
            self.color = [1, 1, 1, 1]
            self.color_scale = 14
        else:
            print('error in assign color')

    def member_test(self, points):
        if points[0] >= self.min_x and points[0] <= self.max_x and \
                points[1] >= self.min_y and points[1] <= self.max_y and \
                points[2] >= self.min_z and points[2] <= self.max_z:
            return True
        else:
            return False

    def load_to_vtk(self, vtk_dic):
        if 'points' not in vtk_dic:
            vtk_dic['points'] = []
            vtk_dic['num_points'] = 0
            vtk_dic['cells'] = []
            vtk_dic['num_cells'] = 0
            vtk_dic['cell_color_scale'] = []

            vtk_dic['Ligand'] = 0
            vtk_dic['Protein'] = 0
            vtk_dic['Mixture'] = 0
            vtk_dic['Empty'] = 0

        vtk_dic['points'] += [self.dot1, self.dot2, self.dot3, self.dot4, self.dot5, self.dot6, self.dot7, self.dot8]
        index_start = vtk_dic['num_points']

        # use all 6 surfaces as cells.
        new_cells = [[4, 0 + index_start, 1 + index_start, 3 + index_start, 2 + index_start],
                         [4, 4 + index_start, 5 + index_start, 7 + index_start, 6 + index_start],
                         [4, 0 + index_start, 4 + index_start, 6 + index_start, 2 + index_start],
                         [4, 2 + index_start, 6 + index_start, 7 + index_start, 3 + index_start],
                         [4, 3 + index_start, 7 + index_start, 5 + index_start, 1 + index_start],
                         [4, 0 + index_start, 4 + index_start, 5 + index_start, 1 + index_start]]
        vtk_dic['cells'] += new_cells
        vtk_dic['num_cells'] += 6
        for i in range(6):
            vtk_dic['cell_color_scale'].append(self.color_scale)

        if self.color_scale == 11:
            vtk_dic['Protein'] += 1
        elif self.color_scale == 12:
            vtk_dic['Ligand'] += 1
        elif self.color_scale == 13:
            vtk_dic['Mixture'] += 1
        elif self.color_scale == 14:
            vtk_dic['Empty'] += 1

        vtk_dic['num_points'] += 8
        if len(vtk_dic['points']) != vtk_dic['num_points']:
            print('points error')
            print(len(vtk_dic['points']), vtk_dic['num_points'])
        if len(vtk_dic['cells']) != vtk_dic['num_cells']:
            print('cell error')
        return vtk_dic

        # print('number of ligand cells: {}'.format(vtk_dic['ligand']))
        # print('number of pocket cells: {}'.format(vtk_dic['pocket']))
        # print('number of mix cells: {}'.format(vtk_dic['mix']))
        # print('number of empty cells: {}'.format(vtk_dic['empty']))
        # print('number of total cells: {}'.format(vtk_dic['ligand'] + vtk_dic['pocket'] + vtk_dic['mix'] + vtk_dic['empty']))

    def load_to_vtk_opt(self, vtk_dic_opt):
        if 'points' not in vtk_dic_opt:
            vtk_dic_opt['points'] = {}
            vtk_dic_opt['num_points'] = 0
            vtk_dic_opt['cells'] = []
            vtk_dic_opt['num_cells'] = 0
            vtk_dic_opt['cell_color_scale'] = []

            vtk_dic_opt['Ligand'] = 0
            vtk_dic_opt['Protein'] = 0
            vtk_dic_opt['Mixture'] = 0
            vtk_dic_opt['Empty'] = 0

        # current_cubic_points = [self.dot1, self.dot2, self.dot3, self.dot4, self.dot5, self.dot6, self.dot7, self.dot8]
        current_cubic_cells = [[self.dot1, self.dot2, self.dot4, self.dot3],
                               [self.dot5, self.dot6, self.dot8, self.dot7],
                               [self.dot1, self.dot5, self.dot7, self.dot3],
                               [self.dot3, self.dot7, self.dot8, self.dot4],
                               [self.dot4, self.dot8, self.dot6, self.dot2],
                               [self.dot1, self.dot5, self.dot6, self.dot2]]

        for cell in current_cubic_cells:
            cell_order = []
            for point in cell:
                point = Point(point)
                if point.string in vtk_dic_opt['points']:
                    point_order = vtk_dic_opt['points'][point.string]
                    cell_order.append(point_order)
                else:
                    point_order = len(vtk_dic_opt['points'])
                    vtk_dic_opt['points'][point.string] = point_order
                    cell_order.append(point_order)
            vtk_dic_opt['cells'].append(cell_order)
            vtk_dic_opt['cell_color_scale'].append(self.color_scale)

        vtk_dic_opt['num_cells'] += 6
        if self.color_scale == 11:
            vtk_dic_opt['Protein'] += 1
        elif self.color_scale == 12:
            vtk_dic_opt['Ligand'] += 1
        elif self.color_scale == 13:
            vtk_dic_opt['Mixture'] += 1
        elif self.color_scale == 14:
            vtk_dic_opt['Empty'] += 1
        else:
            print('Cubic {} have no color scale settled.'.format(self.id))

        if len(vtk_dic_opt['cells']) != vtk_dic_opt['num_cells']:
            print('cell error')
        return vtk_dic_opt

    def write_oct_line_opt(self, vtk_line_dic, color):
        """
        :param pts_dic: {key, value}, where key is Point.string, value is order of the point.
        :param lines_list:
        :param colors_list:
        :param color:
        :return:
        """
        if len(vtk_line_dic) == 0:
            vtk_line_dic['pts_dic'] = {}
            vtk_line_dic['lines_dic'] = {}
            vtk_line_dic['colors_list'] =  []
        pts_dic = vtk_line_dic['pts_dic']
        lines_dic = vtk_line_dic['lines_dic']
        colors_list = vtk_line_dic['colors_list']

        pts_len = len(pts_dic)
        pts_inc = 0

        p0 = [self.min_x, (self.min_y + self.max_y) / 2, (self.min_z + self.max_z) / 2]
        p1 = [self.max_x, (self.min_y + self.max_y) / 2, (self.min_z + self.max_z) / 2]
        p2 = [(self.min_x + self.max_x) / 2, self.min_y, (self.min_z + self.max_z) / 2]
        p3 = [(self.min_x + self.max_x) / 2, self.max_y, (self.min_z + self.max_z) / 2]
        p4 = [(self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2, self.min_z]
        p5 = [(self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2, self.max_z]

        p6 = [self.min_x, (self.min_y + self.max_y) / 2, self.min_z]
        p7 = [self.min_x, (self.min_y + self.max_y) / 2, self.max_z]
        p8 = [self.min_x, self.min_y, (self.min_z + self.max_z) / 2]
        p9 = [self.min_x, self.max_y, (self.min_z + self.max_z) / 2]

        p10 = [(self.min_x + self.max_x) / 2, self.min_y, self.min_z]
        p11 = [(self.min_x + self.max_x) / 2, self.min_y, self.max_z]
        p12 = [(self.min_x + self.max_x) / 2, self.max_y, self.min_z]
        p13 = [(self.min_x + self.max_x) / 2, self.max_y, self.max_z]

        p14 = [self.max_x, (self.min_y + self.max_y) / 2, self.min_z]
        p15 = [self.max_x, (self.min_y + self.max_y) / 2, self.max_z]
        p16 = [self.max_x, self.min_y, (self.min_z + self.max_z) / 2]
        p17 = [self.max_x, self.max_y, (self.min_z + self.max_z) / 2]

        line_list = [[p0, p1], [p2,p3], [p4, p5], [p6, p7], [p8, p9], [p14, p15], [p16, p17],
                         [p8, p16], [p10, p11], [p9, p17], [p12, p13], [p11, p13], [p7, p15],
                         [p6, p14], [p10, p12]]

        for line in line_list:
            start_point = Point(line[0])
            end_point = Point(line[1])
            if start_point.string in pts_dic and end_point.string in pts_dic:
                start_point_order = pts_dic[start_point.string]
                end_point_order = pts_dic[end_point.string]
            elif start_point.string not in pts_dic and end_point.string in pts_dic:
                start_point_order = pts_len + pts_inc
                pts_dic[start_point.string] = start_point_order
                pts_inc += 1
                end_point_order = pts_dic[end_point.string]
            elif start_point.string in pts_dic and end_point.string not in pts_dic:
                end_point_order = pts_len + pts_inc
                pts_dic[end_point.string] = end_point_order
                pts_inc += 1
                start_point_order = pts_dic[start_point.string]
            else:
                start_point_order = pts_len + pts_inc
                pts_dic[start_point.string] = start_point_order
                pts_inc += 1
                end_point_order = pts_len + pts_inc
                pts_dic[end_point.string] = end_point_order
                pts_inc += 1
            line_string = '{}_{}'.format(start_point_order, end_point_order)
            if line_string not in lines_dic:
                lines_dic[line_string] = len(lines_dic)
                colors_list.append(color)

        vtk_line_dic['pts_dic'] = pts_dic
        vtk_line_dic['lines_dic'], vtk_line_dic['colors_list'] = lines_dic, colors_list

        return vtk_line_dic

    def write_edge_line_opt(self, vtk_line_dic, color):
        if 'pts_dic' not in vtk_line_dic:
            vtk_line_dic['pts_dic'] = {}
            vtk_line_dic['lines_dic'] = {}
            vtk_line_dic['colors_list'] = []
        pts_dic = vtk_line_dic['pts_dic']
        lines_dic, colors_list = vtk_line_dic['lines_dic'], vtk_line_dic['colors_list']
        pts_len = len(pts_dic)
        pts_inc = 0

        p0 = self.dot1 #= [self.min_x, self.min_y, self.min_z]
        p1 = self.dot2 #= [self.min_x, self.min_y, self.max_z]
        p2 = self.dot3 #= [self.min_x, self.max_y, self.min_z]
        p3 = self.dot4 #= [self.min_x, self.max_y, self.max_z]
        p4 = self.dot5 #= [self.max_x, self.min_y, self.min_z]
        p5 = self.dot6 #= [self.max_x, self.min_y, self.max_z]
        p6 = self.dot7 #= [self.max_x, self.max_y, self.min_z]
        p7 = self.dot8 #= [self.max_x, self.max_y, self.max_z]

        line_list = [[p0,p1], [p1,p3], [p2,p3], [p0,p2], [p4,p5], [p5,p7], [p6,p7], [p4,p6],
                     [p0,p4], [p2,p6], [p3,p7], [p1,p5]]

        for line in line_list:
            start_point = Point(line[0])
            end_point = Point(line[1])
            if start_point.string in pts_dic and end_point.string in pts_dic:
                start_point_order = pts_dic[start_point.string]
                end_point_order = pts_dic[end_point.string]
            elif start_point.string not in pts_dic and end_point.string in pts_dic:
                start_point_order = pts_len + pts_inc
                pts_dic[start_point.string] = start_point_order
                pts_inc += 1
                end_point_order = pts_dic[end_point.string]
            elif start_point.string in pts_dic and end_point.string not in pts_dic:
                end_point_order = pts_len + pts_inc
                pts_dic[end_point.string] = end_point_order
                pts_inc += 1
                start_point_order = pts_dic[start_point.string]
            else:
                start_point_order = pts_len + pts_inc
                pts_dic[start_point.string] = start_point_order
                pts_inc += 1
                end_point_order = pts_len + pts_inc
                pts_dic[end_point.string] = end_point_order
                pts_inc += 1
            line_string = '{}_{}'.format(start_point_order, end_point_order)
            if line_string not in lines_dic:
                lines_dic[line_string] = len(lines_dic)
                colors_list.append(color)
        vtk_line_dic['pts_dic'] = pts_dic
        vtk_line_dic['lines_dic'], vtk_line_dic['colors_list'] = lines_dic, colors_list
        return vtk_line_dic

class Adaptive_Cubics(object):
    def __init__(self, cubic_dic, complex_id, focus_center = [0,0,0], center = [0,0,0],
                 adaptive_line = True, differ = False,
                 count = False):
        self.cubic_dic = cubic_dic
        self.complex_id = complex_id
        self.differ = differ
        self.adaptive_line = adaptive_line
        self.focus_center = focus_center
        self.center = center
        self.count = count

    def get_adaptive_cubic(self, core_area, side_area1, side_area2, side_area3,
                           target_range, line_depth, total_depth, save_folder, light = False):
        """
        :param core_area: the range core area that need the finest resolution
        :param side_area1: the range side area that need less resolution
        :param side_area2: more range side area that need much less resolution
        :return:
        """
        self.vtk_dic = {}
        self.fine_cubic_id_set = set()
        self.parent_cubic_id_set = set()

        self.fine_cubic_count = {}
        # self.non_fine_cubic_id_set = set()

        for value in ['Protein', 'Ligand', 'Mixture', 'Empty']:
            self.fine_cubic_count[value] = 0

        side1_layer_dic = self.construct_adaptive_dic_V1(self.cubic_dic, core_area,
                                                target_range, self.focus_center, light)
        side2_layer_dic = self.construct_adaptive_dic_V1(side1_layer_dic, side_area1,
                                                 target_range, self.focus_center, light)
        side3_layer_dic = self.construct_adaptive_dic_V1(side2_layer_dic, side_area2,
                                                 target_range, self.focus_center, light)
        side4_layer_dic = self.construct_adaptive_dic_V1(side3_layer_dic, side_area3,
                                                 target_range, self.focus_center, light)
        side5_layer_dic = self.construct_adaptive_dic_V1(side4_layer_dic, target_range,
                                                 target_range, self.focus_center, light)

        print('Fine Cubic Dic Length: {}'.format(len(self.fine_cubic_id_set)))
        # add for adaptive count:
        # count_cubic_id_set = set()
        # parent_cubic_id_set = set()

        if self.adaptive_line:
            segline_cubic_id_set = set()
            vtk_line_dic = {}
            for cubic_id in self.fine_cubic_id_set:
                # if not light:
                    # segline_cubic_id_set.add(cubic.id)  # why need this?
                    # pass
                # self.get_complement_adaptive_id(cubic_id, count_cubic_id_set, target_range)
                parent_cubic_id = get_parent_cubic_id(cubic_id, target_range)
                if parent_cubic_id in self.fine_cubic_id_set:
                    print('Parent and Fine conflict: {}'.format(parent_cubic_id))

                while int(parent_cubic_id.split('_')[0]) >= 0:
                    # self.get_complement_adaptive_id(parent_cubic_id, parent_cubic_id_set, target_range)
                    # parent_cubic_set.add(parent_cubic)
                    # self.get_complement_adaptive(parent_cubic, count_cubic_set, target_range)
                    if int(parent_cubic_id.split('_')[0]) < line_depth:
                        segline_cubic_id_set.add(parent_cubic_id)
                    parent_cubic_id = get_parent_cubic_id(parent_cubic_id, target_range)

            if not light:
                # for parent_cubic_id in parent_cubic_id_set:
                #     if int(parent_cubic_id.split('_')[0]) < line_depth:
                #         segline_cubic_id_set.add(parent_cubic_id)
                print('parent_cubic_id_set length: {}'.format(len(self.parent_cubic_id_set)))
                print('segline_cubic_id_set: {}'.format(len(segline_cubic_id_set)))
                for cubic_id in segline_cubic_id_set:
                    depth = int(cubic_id.split('_')[0])
                    color = get_color_for_dep(depth + 1, self.differ)
                    cubic = cubic_id_to_cubic(cubic_id, target_range)
                    vtk_line_dic = cubic.write_oct_line_opt(vtk_line_dic, color)

                max_cubic = Cubic(
                    [[-target_range, -target_range, -target_range], [target_range, target_range, target_range]],
                    '0_0_0_0')
                color = get_color_for_dep(0, self.differ)
                vtk_line_dic = max_cubic.write_edge_line_opt(vtk_line_dic, color)

                pts, lines, colors = fill_vtk_lines(vtk_line_dic, self.center)
                write_seg_vtk(pts, lines, colors, self.complex_id, target_range, line_depth, save_folder)

            #
            # for cubic_id in self.parent_cubic_id_set:
            #     if cubic_id not in segline_cubic_id_set:
            #         print('parent cubic {} not in segline_cubic_id_set'.format(cubic_id))

            print('Final Node: {}'.format(len(self.fine_cubic_id_set)))
            print('Mid Node: {}'.format(len(segline_cubic_id_set)))

        if not light:
            file_name = '{}/{}/{}_Octree_Adaptive_{}_{}'.format(save_folder, self.complex_id, self.complex_id, target_range, total_depth)
            vtk_writer(vtk_dic=self.vtk_dic, file_name=file_name, shift=self.center)

    def get_adaptive_count(self,core_area, side_area1, side_area2, side_area3,
                           target_range, line_depth, total_depth, save_folder, light = False):
        depth_cubic_dic = {}
        for depth in range(0, total_depth + 1):
            depth_cubic_dic[depth] = set()
        for cubic_id, value in self.cubic_dic:
            [depth, x, y, z] = [int(item) for item in cubic_id.split('_')]
            while depth >= 0:
                depth_cubic_dic[depth].add(cubic_id)
                depth = depth - 1
                x >> 1
                y >> 1
                z >> 1
                cubic_id = '{}_{}_{}_{}'.format(depth, x, y, z)







    def get_complement_adaptive_id(self, cubic_id, count_cubic_id_set, target_range):
        [depth, x, y, z] = [int(item) for item in cubic_id.split('_')]
        if depth <= 0:
            count_cubic_id_set.add(cubic_id)
            return
        x = x >> 1 << 1
        y = y >> 1 << 1
        z = z >> 1 << 1

        for i in range(0, 2):
            for j in range(0, 2):
                for k in range(0, 2):
                    string = '{}_{}_{}_{}'.format(depth, x + i, y + j, z + k)

                    # base = target_range * 2 / 2 ** depth
                    # coord_x = x * base - target_range + i
                    # coord_y = y * base - target_range + j
                    # coord_z = z * base - target_range + k
                    # min_coord = [coord_x, coord_y, coord_z]
                    # max_coord = [coord_x + base, coord_y + base, coord_z + base]
                    # comp_cubic = Cubic([min_coord, max_coord], string)
                    # count_cubic_set.add(comp_cubic)
                    count_cubic_id_set.add(string)

    def construct_adaptive_dic_V1(self, cubic_dic, area, target_range,
                                  focus_center=[0, 0, 0], light=False):
        '''
        :param cubic_dic: dictionary, key is cubic.id, value is Protein, Mixture, Ligand.
        :param area: within this area, use current depth in cubic_dic, otherwise need to compute upper layer
        :param vtk_dic: dictionary that used to plot cubic.
        :param seg_dic: dictionary that used to plot segment line of adaptive cibics.
        :return:
        '''

        side_dic = {}
        for key, value in cubic_dic.items():
            [depth, x, y, z] = [int(item) for item in key.split('_')]

            # if cubic's parent in the core range, the all child of its parent should be in fine_cubic_set.
            p_x = x >> 1
            p_y = y >> 1
            p_z = z >> 1
            p_depth = depth - 1

            p_base = target_range * 2 / 2 ** p_depth
            p_coord_x = p_x * p_base - target_range
            p_coord_y = p_y * p_base - target_range
            p_coord_z = p_z * p_base - target_range
            [focus_x, focus_y, focus_z] = focus_center

            # test if this cubic is in core area, if so, dump into adaptive cubic_dic
            core_flag_z = min(abs(p_coord_z - focus_z), abs(p_coord_z + p_base - focus_z)) <= area
            core_flag_y = min(abs(p_coord_y - focus_y), abs(p_coord_y + p_base - focus_y)) <= area
            core_flag_x = min(abs(p_coord_x - focus_x), abs(p_coord_x + p_base - focus_x)) <= area
            core_flag = core_flag_x and core_flag_y and core_flag_z

            p_id = '{}_{}_{}_{}'.format(p_depth, p_x, p_y, p_z)

            if not core_flag:
                if p_id not in side_dic:
                    side_dic[p_id] = value
                elif value != side_dic[p_id]:
                    side_dic[p_id] = 'Mixture'
                else:
                    pass

            else:
                self.parent_cubic_id_set.add(p_id)
                # self.non_fine_cubic_id_set.add(p_id)
                b_x = p_x << 1
                b_y = p_y << 1
                b_z = p_z << 1
                for i in range(0, 2):
                    for j in range(0, 2):
                        for k in range(0, 2):
                            n_x = b_x + i
                            n_y = b_y + j
                            n_z = b_z + k
                            n_id = '{}_{}_{}_{}'.format(depth, n_x, n_y, n_z)

                            if n_id in self.parent_cubic_id_set:
                                continue
                            if n_id in self.fine_cubic_id_set:
                                continue
                            self.fine_cubic_id_set.add(n_id)

                            if n_id in cubic_dic:
                                base = target_range * 2 / 2 ** depth
                                n_coord_x = n_x * base - target_range
                                n_coord_y = n_y * base - target_range
                                n_coord_z = n_z * base - target_range
                                min_coord = [n_coord_x, n_coord_y, n_coord_z]
                                max_coord = [n_coord_x + base, n_coord_y + base, n_coord_z + base]
                                n_cubic = Cubic([min_coord, max_coord], n_id)
                                n_value = cubic_dic[n_cubic.id]
                                n_cubic.set_color_new(value)
                                if not light:
                                    self.vtk_dic = n_cubic.load_to_vtk_opt(self.vtk_dic)
                                self.fine_cubic_count[n_value] += 1
                            else:
                                self.fine_cubic_count['Empty'] += 1
                                #cubic.set_color_new('Empty')
                                pass
        return side_dic

def cubic_id_to_cubic(id, target_range):
    '''
    based on id, target range, the cubic is identicle.
    :param id:
    :param target_range:
    :return:
    '''
    [depth, x, y, z] = [int(item) for item in id.split('_')]
    # (depth, x, y, z) = id
    base = target_range * 2 / 2 ** depth
    coord_x = x * base - target_range
    coord_y = y * base - target_range
    coord_z = z * base - target_range
    min_coord = [coord_x, coord_y, coord_z]
    max_coord = [coord_x + base, coord_y + base, coord_z + base]
    return Cubic([min_coord, max_coord], id)

def point_string_to_coord(string):
    coord = string.split('_')
    return [float(item) for item in coord]

def write_seg_vtk(pts, lines, colors, complex_id, target_range, line_depth, save_folder):
    linesPolyData = vtk.vtkPolyData()
    linesPolyData.SetPoints(pts)
    linesPolyData.SetLines(lines)
    linesPolyData.GetCellData().SetScalars(colors)

    # clean = vtk.vtkCleanPolyData()
    # clean.SetInputConnection(linesPolyData)
    # clean.Update()
    # pd = clean.GetOutput()

    # print('clean segline_Adaptive data.')
    # then = time.time()
    # # pd = linesPolyData.getPolyData()
    # # clean = vtk.vtkCleanPolyData()
    # # clean.SetInput(linesPolyData)
    # # clean.Update()
    # # pd = clean.GetOutput()
    #
    # # clean = vtk.vtkCleanPolyData()
    # # clean.SetInputConnection(linesPolyData)
    # # clean.Update()
    # # pd = clean.GetOutput()
    #
    # now = time.time()
    # diff = int(now - then)
    # print('clean segline_Adaptive data done. spend {} seconds'.format(diff))

    ShellVolmapper = vtk.vtkDataSetMapper()
    ShellVolmapper.SetScalarModeToUseCellData()
    ShellVolmapper.UseLookupTableScalarRangeOn()
    # ShellVolmapper.SetLookupTable(lut)
    ShellVolmapper.SetInputData(linesPolyData)
    ShellVolactor = vtk.vtkActor()
    ShellVolactor.SetMapper(ShellVolmapper)

    # modelwriter = vtk.vtkUnstructuredGridWriter()
    modelwriter = vtk.vtkDataSetWriter()
    modelwriter.SetFileName('{}/{}/{}_segline_Adaptive_{}_{}.vtk'.format(save_folder, complex_id, complex_id,
                                                                                target_range, line_depth))
    # modelwriter.SetInputData(grid)
    modelwriter.SetInputData(ShellVolmapper.GetInput())
    modelwriter.Write()
    print('Save {}/{}/{}_segline_Adaptive_{}_{}.vtk'.format(save_folder, complex_id, complex_id, target_range, line_depth))

def get_parent_cubic(cubic, target_range):
    id = cubic.id
    [depth, x, y, z] = [int(item) for item in id.split('_')]
    x = x >> 1
    y = y >> 1
    z = z >> 1
    depth -= 1
    base = target_range * 2 / 2 ** depth
    coord_x = x * base - target_range
    coord_y = y * base - target_range
    coord_z = z * base - target_range
    min_coord = [coord_x, coord_y, coord_z]
    max_coord = [coord_x + base, coord_y + base, coord_z + base]
    string = '{}_{}_{}_{}'.format(depth, x, y, z)
    parent_cubic = Cubic([min_coord, max_coord], string)
    return parent_cubic

def get_parent_cubic_id(id, target_range):
    [depth, x, y, z] = [int(item) for item in id.split('_')]
    x = x >> 1
    y = y >> 1
    z = z >> 1
    depth -= 1
    # base = target_range * 2 / 2 ** depth
    # coord_x = x * base - target_range
    # coord_y = y * base - target_range
    # coord_z = z * base - target_range
    # min_coord = [coord_x, coord_y, coord_z]
    # max_coord = [coord_x + base, coord_y + base, coord_z + base]
    parent_id = '{}_{}_{}_{}'.format(depth, x, y, z)
    # parent_cubic = Cubic([min_coord, max_coord], string)
    return parent_id

def get_color_for_dep(depth, differ = True):
    # need to be different with the volume scaler (0,1,2,3).
    if differ == False:
        color = 0
        return color
    # if depth == 0:
    #     color = 0 #[0, 0, 0] # white
    # elif depth == 1:
    #     color = 1 # [255, 0, 0] # red
    # elif depth == 2:
    #     color = 2 # [255, 128, 0]
    # elif depth == 3:
    #     color = 3 #[255,255,0]
    # elif depth == 4:
    #     color = 4 # [0, 255, 0]
    # elif depth == 5:
    #     color = 5 #[0, 255, 255]
    # elif depth == 6:
    #     color = 6 #[0, 0, 255]
    # elif depth == 7:
    #     color = 7 # [255, 0 , 255]
    # elif depth == 8 :
    #     color = 8
    # else:
    #     color = 9
    return depth

# A better way is use class in vtk package, this solve the color issue to distinguish ligand, pocket.
def vtk_writer(vtk_dic, file_name, shift = [0,0,0]):
    points = vtk.vtkPoints()
    grid = vtk.vtkUnstructuredGrid()
    Color = vtk.vtkFloatArray()
    Scale = vtk.vtkFloatArray()

    sorted_dic = sorted(vtk_dic['points'].items(), key=lambda x: x[1])
    pts = vtk.vtkPoints()
    for item in sorted_dic:
        # pts.InsertNextPoint(point_string_to_coord(item[0]))
        coord = point_string_to_coord(item[0])
        coord = [coord[0] + shift[0], coord[1] + shift[1], coord[2] + shift[2]]
        pts.InsertNextPoint(coord)
    grid.SetPoints(pts)

    # for vtk_point in vtk_dic['points']:
    #     Coord = vtk_point
    #     test = points.InsertNextPoint(*Coord)
    #     #Color.InsertTuple1(test, 0)
    # grid.SetPoints(points)

    for i in range(len(vtk_dic['cells'])):
        vtk_node = vtk_dic['cells'][i]
        elem = vtk.vtkQuad()
        elem.GetPointIds().SetId(0, vtk_node[0])
        elem.GetPointIds().SetId(1, vtk_node[1])
        elem.GetPointIds().SetId(2, vtk_node[2])
        elem.GetPointIds().SetId(3, vtk_node[3])
        Quad4cell = grid.InsertNextCell(elem.GetCellType(), elem.GetPointIds())
        color = vtk_dic['cell_color_scale'][i]
        Color.InsertTuple1(Quad4cell, color)

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(5)
    lut.SetTableRange(0, 5)
    lut.Build()
    lut.SetTableValue(0, 0, 0, 0, 1)  # Black
    lut.SetTableValue(1, 1, 0, 0, 1)  # Red
    lut.SetTableValue(2, 0, 1, 0, 1)  # Green
    lut.SetTableValue(3, 0, 0, 1, 1)  # Blue
    lut.SetTableValue(4, 1, 1, 1, 1)  # White

    grid.GetCellData().SetScalars(Color)
    ShellVolmapper = vtk.vtkDataSetMapper()
    # ShellVolmapper = vtk.vtkUnstructuredGridVolumeRayCastMapper()
    ShellVolmapper.SetScalarModeToUseCellData()
    ShellVolmapper.UseLookupTableScalarRangeOn()
    ShellVolmapper.SetLookupTable(lut)
    ShellVolmapper.SetInputData(grid)
    ShellVolactor = vtk.vtkActor()
    ShellVolactor.SetMapper(ShellVolmapper)

    modelwriter = vtk.vtkUnstructuredGridWriter()
    modelwriter.SetFileName('{}.vtk'.format(file_name))
    #modelwriter.SetInputData(grid)
    modelwriter.SetInputData(ShellVolmapper.GetInput())
    modelwriter.Write()
    print('Save {}.vtk'.format(file_name))


def construct_adaptive_dic(cubic_dic, area, vtk_dic, fine_cubic_set, target_range,
                           focus_center = [0,0,0], light =False):
    '''
    :param cubic_dic: dictionary, key is cubic.id, value is Protein, Mixture, Ligand.
    :param area: within this area, use current depth in cubic_dic, otherwise need to compute upper layer
    :param vtk_dic: dictionary that used to plot cubic.
    :param seg_dic: dictionary that used to plot segment line of adaptive cibics.
    :return:
    '''
    side_dic = {}
    for key, value in cubic_dic.items():
        [depth, x, y, z] = [int(item) for item in key.split('_')]
        base = target_range * 2 / 2 ** depth
        coord_x = x * base - target_range
        coord_y = y * base - target_range
        coord_z = z * base - target_range
        min_coord = [coord_x, coord_y, coord_z]
        max_coord = [coord_x + base, coord_y + base, coord_z + base]
        [focus_x, focus_y, focus_z] = focus_center

        # test if this cubic is in core area, if so, dump into adaptive cubic_dic
        core_flag_z = min(abs(coord_z - focus_z), abs(coord_z + base - focus_z)) <= area
        core_flag_y = min(abs(coord_y - focus_y), abs(coord_y + base - focus_y)) <= area
        core_flag_x = min(abs(coord_x - focus_x), abs(coord_x + base - focus_x)) <= area
        core_flag = core_flag_x and core_flag_y and core_flag_z

        # find parent cubic or upper layer cubic.
        # parent cubic for adaptive segment line, upper layer cubic for cubic plot.
        x = x >> 1
        y = y >> 1
        z = z >> 1
        depth -= 1
        string = '{}_{}_{}_{}'.format(depth, x, y, z)

        if core_flag:
            cubic = Cubic([min_coord, max_coord], key)
            cubic.set_color_new(value)
            if not light:
                vtk_dic = cubic.load_to_vtk_opt(vtk_dic)
            fine_cubic_set.add(cubic)
        else:
            if string not in side_dic:
                side_dic[string] = value
            elif value != side_dic[string]:
                side_dic[string] = 'Mixture'
            else:
                pass
    return side_dic




def centerize(pocket_xyz, ligand_xyz, by_ligand = True, protein_xyz = None):
    if by_ligand:
        [x, y, z] = ligand_xyz
        center_x, center_y, center_z = x.mean(), y.mean(), z.mean()
        x, y, z = x - center_x, y - center_y, z - center_z
        ligand_xyz = [x, y, z]
        [x, y, z] = pocket_xyz
        x, y, z = x - center_x, y - center_y, z - center_z
        pocket_xyz = [x, y, z]
        ligand_center = [0, 0, 0]
        center = [center_x, center_y, center_z]

        if protein_xyz is not None:
            [x, y, z] = protein_xyz
            x, y, z = x - center_x, y - center_y, z - center_z
            protein_xyz = [x, y, z]

    else:
        [x_ligand, y_ligand, z_ligand] = ligand_xyz
        x_ligand_max, y_ligand_max, z_ligand_max = x_ligand.max(), y_ligand.max(), z_ligand.max()
        x_ligand_min, y_ligand_min, z_ligand_min = x_ligand.min(), y_ligand.min(), z_ligand.min()
        if protein_xyz is None:
            [x_pocket, y_pocket, z_pocket] = pocket_xyz
            x_protein_max, y_protein_max, z_protein_max = x_pocket.max(), y_pocket.max(), z_pocket.max()
            x_protein_min, y_protein_min, z_protein_min = x_pocket.min(), y_pocket.min(), z_pocket.min()
        else:
            [x_protein, y_protein, z_protein] = protein_xyz
            x_protein_max, y_protein_max, z_protein_max = x_protein.max(), y_protein.max(), z_protein.max()
            x_protein_min, y_protein_min, z_protein_min =  x_protein.min(), y_protein.min(), z_protein.min()
        x_max, y_max, z_max = max(x_ligand_max, x_protein_max), max(y_ligand_max, y_protein_max), max(z_ligand_max, z_protein_max)
        x_min, y_min, z_min = min(x_ligand_min, x_protein_min), min(y_ligand_min, y_protein_min), min(z_ligand_min, z_protein_min)
        center = [(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2]
        ligand_center = [x_ligand.mean() - center[0], y_ligand.mean() - center[1], z_ligand.mean()- center[2]]
        ligand_xyz = [ligand_xyz[0] - center[0], ligand_xyz[1] - center[1], ligand_xyz[2] - center[2]]
        pocket_xyz = [pocket_xyz[0] - center[0], pocket_xyz[1] - center[1], pocket_xyz[2] - center[2]]
        if protein_xyz is not None:
            [x, y, z] = protein_xyz
            x, y, z = x - center[0], y - center[1], z - center[2]
            protein_xyz = [x, y, z]

    return pocket_xyz, ligand_xyz, protein_xyz, center, ligand_center

def filter_by_center_range(x, y, z, target_range, feature=None): # Here x, y, z is list
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    min_x, min_y, min_z = -target_range, -target_range, -target_range
    max_x, max_y, max_z = target_range, target_range, target_range

    flags_x = np.logical_and(x > min_x, x < max_x)
    flags_y = np.logical_and(y > min_y, y < max_y)
    flags_z = np.logical_and(z > min_z, z < max_z)
    flag1 = np.logical_and(flags_x, flags_y)
    flag2 = np.logical_and(flag1, flags_z)
    x = x[(flag2)]
    y = y[(flag2)]
    z = z[(flag2)]

    if feature is not None:
        feature = np.array(feature)
        feature = feature[(flag2)]
        return x,y,z,feature
    return x, y, z

def get_vol_index(x, y, z, depth, target_range):
    min_x, min_y, min_z = -target_range, -target_range, -target_range
    x = x - min_x
    y = y - min_y
    z = z - min_z
    resolution = target_range*2 / 2 ** depth
    x = (x // resolution).astype(int)
    y = (y // resolution).astype(int)
    z = (z // resolution).astype(int)
    # vol_set = set()
    # for i in range(len(x)):
    #     vol_set.add((x[i], y[i], z[i]))
    # print(vol_set)
    # print(min(x), max(x))
    return x, y, z

def get_comp_dic(cubic_dic, comp_layer_dic = {}):
    for key, value in cubic_dic.items():
        [depth, x, y, z] = [int(item) for item in key.split('_')]
        x = x >> 1 << 1
        y = y >> 1 << 1
        z = z >> 1 << 1

        for i in range(0,2):
            for j in range(0,2):
                for k in range(0,2):
                    string = '{}_{}_{}_{}'.format(depth, x+i, y+j, z+k)
                    if string not in cubic_dic:
                        comp_layer_dic[string] = 'Empty'
                    else:
                        comp_layer_dic[string] = cubic_dic[string]
                    pass

    # for key, value in comp_layer_dic.items():
    #     if value == 'Ligand':
    #         print(key)
    return comp_layer_dic



def get_upper_layer_dic(cubic_dic):
    upper_layer_dic = {}
    for key, value in cubic_dic.items():
        [depth, x, y, z] = [int(item) for item in key.split('_')]
        x = x >> 1
        y = y >> 1
        z = z >> 1
        depth -= 1
        string = '{}_{}_{}_{}'.format(depth, x,y,z)
        if string not in upper_layer_dic:
            upper_layer_dic[string] = value
        elif value != upper_layer_dic[string]:
            upper_layer_dic[string] = 'Mixture'
        else:
            pass
    return upper_layer_dic


def create_cubic_dic(protein_xyz, ligand_xyz, depth):
    cubic_dic = {}
    [x, y, z] = protein_xyz
    print('Protein points number: {}'.format(len(x)))
    for i in range(len(x)):
        string = '{}_{}_{}_{}'.format(depth, x[i], y[i], z[i])
        if string not in cubic_dic:
            cubic_dic[string] = 'Protein'
        else:
            pass

    [x, y, z] = ligand_xyz
    print('Ligand points number: {}'.format(len(x)))
    for i in range(len(x)):
        string = '{}_{}_{}_{}'.format(depth, x[i], y[i], z[i])
        if string not in cubic_dic:
            cubic_dic[string] = 'Ligand'
        elif cubic_dic[string] == 'Protein':
            cubic_dic[string] = 'Mixture'
        else:
            pass
    return cubic_dic

def create_cubic_dic_with_feature(protein_xyz, ligand_xyz, depth, pocket_feature, ligand_feature):
    cubic_dic = {}
    [x, y, z] = protein_xyz
    for i in range(len(x)):
        string = '{}_{}_{}_{}'.format(depth, x[i], y[i], z[i])
        feature = pocket_feature[i]
        atom_type = feature[0]
        atom_index = 'P' + str(feature[1])
        atom_detail_type = 'P' + str(feature[0])
        if string not in cubic_dic:
            cubic_dic[string] = {}
            cubic_dic[string]['PLM'] = 'Protein'
            cubic_dic[string]['Atoms'] = [atom_type]
            cubic_dic[string]['Atom_index'] = [atom_index]
            cubic_dic[string]['Atom_Detail_Type'] = [atom_detail_type]
        else:
            cubic_dic[string]['Atom_Detail_Type'].append(atom_detail_type)
            if atom_type not in cubic_dic[string]['Atoms']:
                cubic_dic[string]['Atoms'].append(atom_type)
            if atom_index not in cubic_dic[string]['Atom_index']:
                cubic_dic[string]['Atom_index'].append(atom_index)

    [x, y, z] = ligand_xyz
    for i in range(len(x)):
        string = '{}_{}_{}_{}'.format(depth, x[i], y[i], z[i])
        feature = ligand_feature[i]
        atom_type = feature[0]
        atom_index = 'L' + str(feature[1])
        atom_detail_type = 'L' + str(feature[0])
        if string not in cubic_dic:
            cubic_dic[string] = {}
            cubic_dic[string]['PLM'] = 'Ligand'
            cubic_dic[string]['Atoms'] = [atom_type]
            cubic_dic[string]['Atom_index'] = [atom_index]
            cubic_dic[string]['Atom_Detail_Type'] = [atom_detail_type]
        elif cubic_dic[string]['PLM'] == 'Protein':
            cubic_dic[string]['PLM'] = 'Mixture'
            cubic_dic[string]['Atom_Detail_Type'].append(atom_detail_type)
        else:
            cubic_dic[string]['Atom_Detail_Type'].append(atom_detail_type)
        if atom_type not in cubic_dic[string]['Atoms']:
            cubic_dic[string]['Atoms'].append(atom_type)
        if atom_index not in cubic_dic[string]['Atom_index']:
            cubic_dic[string]['Atom_index'].append(atom_index)
    return cubic_dic

def line_string_to_idx(string):
    contents = string.split('_')
    contents = [int(item) for item in contents]
    return contents

def fill_vtk_lines(vtk_line_dic, shift = [0,0,0]):
    pts_dic = vtk_line_dic['pts_dic']
    lines_dic, colors_list = vtk_line_dic['lines_dic'], vtk_line_dic['colors_list']

    pts = vtk.vtkPoints()
    sorted_dic = sorted(pts_dic.items(), key=lambda x: x[1])
    for item in sorted_dic:
        coord = point_string_to_coord(item[0])
        coord = [coord[0] + shift[0], coord[1] + shift[1], coord[2] + shift[2]]
        pts.InsertNextPoint(coord)


    lines = vtk.vtkCellArray()
    colors = vtk.vtkFloatArray()
    sorted_line_dic = sorted(lines_dic.items(), key=lambda x: x[1])
    for item in sorted_line_dic:
        line_idx = line_string_to_idx(item[0])
        line0 = vtk.vtkLine()
        line0.GetPointIds().SetId(0, line_idx[0])
        line0.GetPointIds().SetId(1, line_idx[1])
        lines.InsertNextCell(line0)

    # for line in lines_list:
    #     points = line.split('_')
    #     points = [int(item) for item in points]
    #     line0 = vtk.vtkLine()
    #     line0.GetPointIds().SetId(0, points[0])
    #     line0.GetPointIds().SetId(1, points[1])
    #     lines.InsertNextCell(line0)
    for color in colors_list:
        colors.InsertNextTuple1(color)
    return pts, lines, colors
