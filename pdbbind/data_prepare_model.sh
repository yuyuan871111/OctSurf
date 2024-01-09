# prepare points file
cd data_folder/v2018-other-PL
bash data_prepare_points.sh

cd ../refined-set
bash data_prepare_points.sh

cd ../coreset
bash data_prepare_points.sh

cd ..
mkdir tfrecords

# separate the data into train/test/val
cd ../../python
python generate_pdb_list.py

# generate the tfrecords for tf model.
cd ../../../tensorflow/util
python convert_tfrecords.py --file_dir ../../pdbbind/data_folder --list_file ../../pdbbind/data_folder/points_list_test_reg.txt --records_name ../../pdbbind/data_folder/tfrecords/test_reg_points_den3.tfrecords --label_type float
python convert_tfrecords.py --file_dir ../../pdbbind/data_folder --list_file ../../pdbbind/data_folder/points_list_val_reg.txt --records_name ../../pdbbind/data_folder/tfrecords/val_reg_points_den3.tfrecords --label_type float
python convert_tfrecords.py --file_dir ../../pdbbind/data_folder --list_file ../../pdbbind/data_folder/points_list_train_reg.txt --records_name ../../pdbbind/data_folder/tfrecords/train_reg_points_den3.tfrecords --label_type float
