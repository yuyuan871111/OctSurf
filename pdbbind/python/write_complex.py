import os

path = os.getcwd()
complex_id = path.split('/')[-1]


# file_list is used to get .points from pdb points.
id_list_file = path + '/file_list.txt'
with open(id_list_file, 'w') as fw:
    fw.write(complex_id)

# point_list is used to get .octrees from .points
id_list_file = path + '/point_list.txt'
with open(id_list_file, 'w') as fw:
    fw.write(path  + '/' + complex_id + '_points.points')