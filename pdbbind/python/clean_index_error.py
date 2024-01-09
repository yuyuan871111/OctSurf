# Some CONECT index in pdb file is 0, which cause cdk does not work.
# This file is used to revise the pdb file, by simply remove the CONECT record include 0 index.

def clean_index_for_file(folder):
    contents = folder.split('/')
    id = contents[-1]
    # ligand_file = folder + '/{}_ligand.mol2'.format(id)
    if id == 'index' or id == 'readme':
        return None
    file = folder + '/{}_pocket.pdb'.format(id)

    rewrite = False
    new_file_lines = []

    with open(file, 'r') as f:
        rows = f.readlines()
        for row in rows:
            if 'CONECT' in row:
                if ' 0' in row:
                    rewrite = True
                    continue
            new_file_lines.append(row)

    if rewrite:
        print('Need clean index error: ', id)
        with open(file, 'w') as f:
            for line in new_file_lines:
                f.write(line)
        return id
    else:
        return None

import os

if __name__ == "__main__":
    path = os.getcwd()
    clean_index_for_file(path)