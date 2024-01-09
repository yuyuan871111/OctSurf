# OctSurf: Efficient Hierarchical Voxel-based Molecular Surface Representation for the Protein-Ligand Affinity Prediction.

## Disclaimer

**This is the copied repository for OctSurf: Efficient Hierarchical Voxel-based Molecular Surface Representation for the Protein-Ligand Affinity Prediction.**
**I am a user of OctSurf and found the original repository is under Enterprise GitHub, which is not easy to share and use for general GitHub users. Thus, based on the MIT License, I copied the original repository to my repository and made it public.**

**The below readme is copied from the original repository.**

## Experiments

### Download PDBbind   
Download PDBbind general, refined, and core(CASF) from http://www.pdbbind.org.cn.   
Also fix some minor problems in data(replace several mol2 files by transforming sdf, and remove the CONECT with index 0 in pdb file). 
```angular2
cd pdbbind
bash data_download.sh
cd ..
```

### Set-up Enviroment
Install packages and compile the required tools, e.g. the java tool for generating surface points, the C++ code for octree, and the operation (convolution etc.) API that work with tensorflow. (cmake/3.10.2, gcc/5.4.0, cuda/10.1)
```angular2
# compile java
cd pdbbind/java
javac -cp cdk-2.3-SNAPSHOT.jar Surface_for_single.java
cd ../..

# set-up TF enviroment and compile cpp
conda create -n OctSurf_env tensorflow-gpu==1.14.0
conda activate OctSurf_env
conda install -c conda-forge openbabel==3.0.0
conda install -c conda-forge yacs tqdm
pip install sklearn
pip install matplotlib
pip install seaborn
# uncomment if want to visualize in Paraview
# pip install vtk==9.0.3


cd octree/external && git clone --recursive https://github.com/wang-ps/octree-ext.git
cd .. && mkdir build && cd build
cmake .. -DUSE_CUDA=ON && cmake --build . --config Release
export PATH=`pwd`:$PATH

cd ../../../tensorflow/libs
python build.py
cd ../../
```

### Octree Generation Example (Optional)
Provide one example data 1A1E from refined-set.    
Following steps can generate the points and build the OctSurf. (Default density for points is 6, and depth for OctSurf is 10. Can be specified in octree_for_single.sh file)    
We also provide python tool to parse the generated OctSurf, and generate vtk files that can be visualized in Paraview.    
```angular2
cd pdbbind/data_example/pdbbind
cp ../../bash_scripts/octree_for_single.sh .
cd 1a1e
bash ../octree_for_single.sh
cd octree_folder
ls -lh
cd ../../../../

# parse by python, and visualization(optional)
cd python
python octree_parse.py
cd ../
```
### CNN Modeling
#### prepare the data for modeling   
First it will generate the points file for each complex in general, refined, core set. (The density of points can be specificed in data_prepare_points.sh, low resolution OctSurf can use low density points to accelerate the process, here for depth=6 model, we use density 3.)   
Then the points and labels will be transformed into tfrecords file.   
```angular2
bash data_prepare_model.sh
cd ..
```
#### train model   
Specify the config files (the network architecture, the input/log path, iterations etc.)  
```angular2
cd tensorflow/script
python run_cls.py --config configs/train_resnet_depth6.yaml
```

#### test performance    
Specify the config files (the path for pretrained model/test dataset, network architecture, iterations etc)  
Test the pre-trained model on test dataset, and report the performance.  
```angular2
python test_reg_model.py --config configs/test_resnet_depth6.yaml
```

## Acknowledgments
Code is inspired by [O-CNN](https://wang-ps.github.io/O-CNN.html).

The code is released under the **MIT license**.  
Please contact us (Qinqing Liu qinqing.liu@uconn.edu, Minghu Song minghu.song@uconn.edu )
if you have any problems about our implementation.  

