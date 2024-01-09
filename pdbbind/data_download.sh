mkdir data_folder
cd data_folder

wget "http://www.pdbbind.org.cn/download/pdbbind_v2018_refined.tar.gz"
wget "http://www.pdbbind.org.cn/download/pdbbind_v2018_other_PL.tar.gz"
wget "http://www.pdbbind-cn.org/download/CASF-2016.tar.gz"

tar -xzvf pdbbind_v2018_refined.tar.gz
tar -xzvf pdbbind_v2018_other_PL.tar.gz
tar -xzvf CASF-2016.tar.gz
mv CASF-2016/coreset .

cp ../bash_scripts/* v2018-other-PL
cp ../bash_scripts/* refined-set
cp ../bash_scripts/* coreset

cd v2018-other-PL
bash correct_mol2.sh
bash correct_index0.sh

cd ../refined-set
bash correct_index0.sh

cd ../coreset
bash correct_index0.sh


