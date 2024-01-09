for D in ./*/;
do
    echo "${D}";
    cd "${D}";
    java -cp ../../../java/cdk-2.3-SNAPSHOT.jar:../../../java Surface_for_single 3 ico pdbbind;
    python ../../../python/write_complex.py
    python ../../../python/atomic_feature.py
    ../../../../octree/build/pdb_to_points --complex_id file_list.txt
    cd ..;
done