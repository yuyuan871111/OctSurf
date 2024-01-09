# here 6 is a parameter to control the density of point clouds
# 6 is for depth 10, can try smaller numbers e.g. 3-5 for smaller depth
java -cp ../../../java/cdk-2.3-SNAPSHOT.jar:../../../java Surface_for_single 6 ico pdbbind

python ../../../python/write_complex.py
python ../../../python/atomic_feature.py
../../../../octree/build/pdb_to_points --complex_id file_list.txt
mkdir octree_folder
../../../../octree/build/octree --depth 10 --filenames point_list.txt --output_path ./octree_folder
cd octree_folder
ls -lh

