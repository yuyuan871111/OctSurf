#include <iostream>
#include <fstream>

#include "points.h"
#include "filenames.h"
#include "cmd_flags.h"

DEFINE_string(complex_id, kRequired, "", "The file include input complex_id");
DEFINE_string(output_path, kOptional, "./", "The output path");

using std::cout;
using std::vector;


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: custom_points");
    return 0;
  }
//  auto& flag_map = FlagRegistry::Registry();
//  string complex_id = flag_map.find("complex_id");


    Points point_cloud;
    vector<float> pts, normals, features, labels;

    // read ID_name from file:
    std::ifstream infile0(FLAGS_complex_id);
	string ID_name;
	std::getline(infile0, ID_name);
	infile0.close();

    std::cout << "the complex id is: " << ID_name << "." << std::endl;
    string ligand_file_name = "./" + ID_name + "_cdk_ligand.xyz";
    std::ifstream infile(ligand_file_name);

    int lig_num = 0;
    int i;
    float x, y, z;
    float center[3] = {0,0,0};
    std::string line;
    std::string ele1, ele2, ele3;
    std::string delimiter = " ";
    while (infile >> ele1 >> ele2 >> ele3)
    {
        x = std::stof(ele1);
        y = std::stof(ele2);
        z = std::stof(ele3);

        lig_num += 1;

        pts.push_back(x); pts.push_back(y); pts.push_back(z);

        center[0] += x;
        center[1] += y;
        center[2] += z;
    }

//    std::cout << center[0] << center[1] << center[2] << lig_num  << std::endl;
    std::cout << "ligand point number: " << lig_num << std::endl;

    center[0] = center[0] / lig_num;
    center[1] = center[1] / lig_num;
    center[2] = center[2] / lig_num;
    std::cout << "center is: " << center[0] << " " << center[1] << " " << center[2] << std::endl;

    infile.close();

    std::ofstream outputFile;
    outputFile.open("center.txt");
    outputFile << center[0] << " " << center[1] << " " << center[2] ;
    outputFile.close();


    string pocket_file_name = "./" + ID_name + "_cdk_pocket.xyz";
    std::ifstream infile2(pocket_file_name);

    int pro_num = 0;
    while (infile2 >> ele1 >> ele2 >> ele3)
    {
        //std::cout << ele1 << " " << ele2 << " " << ele3 << ". ";
        x = std::stof(ele1);
        y = std::stof(ele2);
        z = std::stof(ele3);

        pro_num++;

        //i = n - 1;
        pts.push_back(x); pts.push_back(y); pts.push_back(z);

    }
    std::cout << "last value in pts: " << pts[pts.size()-1] << std::endl;

    infile2.close();
    std::cout << "pocket point number: " << pro_num << std::endl;




    // QQ:  create dictionary{id: feature} according to index, the index start from 1, same as mol2 file and pdb file.
    string ligand_feature_file_name = "./" + ID_name + "_ligand_feature.txt";
    std::ifstream infile3(ligand_feature_file_name);
    std::map<int, vector<float>> ligand_feature_dic;
    std::string ele, ele0, ele4, ele5, ele6, ele7, ele8, ele9, ele10, ele11, ele12, ele13,
                ele14, ele15, ele16, ele17, ele18, ele19, ele20, ele21, ele22, ele23, ele24;

    int idx;
    while (infile3 >> ele0 >> ele1 >> ele2 >> ele3 >> ele4 >> ele5 >> ele6 >> ele7 >> ele8 >> ele9 >> ele10 >>
     ele11 >> ele12 >> ele13 >> ele14 >> ele15 >> ele16 >> ele17 >> ele18 >> ele19 >> ele20 >>
     ele21 >> ele22 >> ele23 >> ele24)
    {
//        std::cout << ele1 << " " << ele2 << " " << ele3 << " " << ele4 << " " << ele5 <<" " <<  ele6<< std::endl;
        idx = std::stoi(ele0);
//        std::cout << "index is: " << idx << '\n';

        vector<float> feature;
        feature.clear();
//        vdwrad = std::stof(ele18);
        x = std::stof(ele1);
        y = std::stof(ele2);
        z = std::stof(ele3);

//        std::cout << "x,y,z is " << x << " " << y << " "<< z << std::endl;
        feature.push_back(x); feature.push_back(y); feature.push_back(z);

        feature.push_back(std::stof(ele4));
        feature.push_back(std::stof(ele5));
        feature.push_back(std::stof(ele6));
        feature.push_back(std::stof(ele7));
        feature.push_back(std::stof(ele8));
        feature.push_back(std::stof(ele9));
        feature.push_back(std::stof(ele10));
        feature.push_back(std::stof(ele11));
        feature.push_back(std::stof(ele12));
        feature.push_back(std::stof(ele13));
        feature.push_back(std::stof(ele14));
        feature.push_back(std::stof(ele15));
        feature.push_back(std::stof(ele16));
        feature.push_back(std::stof(ele17));
        feature.push_back(std::stof(ele18));
        feature.push_back(std::stof(ele19));
        feature.push_back(std::stof(ele20));
        feature.push_back(std::stof(ele21));
        feature.push_back(std::stof(ele22));
        feature.push_back(std::stof(ele23));
        feature.push_back(std::stof(ele24));

        ligand_feature_dic[idx] = feature;

//        for(int i =0; i< 24; i++){std::cout << i << " " << feature[i] << " ";}
//        std::cout << std::endl;
//        std::cout << "feature size: " << feature.size() <<std::endl;
        feature.clear();

    }
    infile3.close();
    std::cout << "ligand_feature_dic.size() is " << ligand_feature_dic.size() << '\n';

    string pocket_feature_file_name = "./" + ID_name + "_pocket_feature.txt";
    std::ifstream infile4(pocket_feature_file_name);
    std::map<int, vector<float>> pocket_feature_dic;

    while (infile4 >> ele0 >> ele1 >> ele2 >> ele3 >> ele4 >> ele5 >> ele6 >> ele7 >> ele8 >> ele9 >> ele10 >>
     ele11 >> ele12 >> ele13 >> ele14 >> ele15 >> ele16 >> ele17 >> ele18 >> ele19 >> ele20 >>
     ele21 >> ele22 >> ele23 >> ele24)
    {
        //std::cout << ele1 << " " << ele2 << " " << ele3 << ". ";
        idx = std::stoi(ele0);
//        std::cout << "index is: " << idx << '\n';

        vector<float> feature;
        x = std::stof(ele1);
        y = std::stof(ele2);
        z = std::stof(ele3);

        feature.push_back(x); feature.push_back(y); feature.push_back(z);

        feature.push_back(std::stof(ele4));
        feature.push_back(std::stof(ele5));
        feature.push_back(std::stof(ele6));
        feature.push_back(std::stof(ele7));
        feature.push_back(std::stof(ele8));
        feature.push_back(std::stof(ele9));
        feature.push_back(std::stof(ele10));
        feature.push_back(std::stof(ele11));
        feature.push_back(std::stof(ele12));
        feature.push_back(std::stof(ele13));
        feature.push_back(std::stof(ele14));
        feature.push_back(std::stof(ele15));
        feature.push_back(std::stof(ele16));
        feature.push_back(std::stof(ele17));
        feature.push_back(std::stof(ele18));
        feature.push_back(std::stof(ele19));
        feature.push_back(std::stof(ele20));
        feature.push_back(std::stof(ele21));
        feature.push_back(std::stof(ele22));
        feature.push_back(std::stof(ele23));
        feature.push_back(std::stof(ele24));

        pocket_feature_dic[idx] = feature;
//        for(int i =0; i< 24; i++){std::cout << i << feature[i] << " ";}
//        std::cout << std::endl;
//        std::cout << "feature size: " << feature.size() <<std::endl;
        feature.clear();

    }
    infile4.close();
    std::cout << "pocket_feature_dic.size() is " << pocket_feature_dic.size() << '\n';







        // QQ: set normal and feature
    string ligand_pt_source_file_name = "./" + ID_name + "_cdk_ligand.txt";
    std::ifstream infile5(ligand_pt_source_file_name);
    int atom_idx; // QQ: atom_idx is the index in mol2, pdb file.
    float normal_x, normal_y, normal_z;
    idx = 0;
    float atom_x, atom_y, atom_z;
    float vdwrad;
    while (infile5 >> ele1 >> ele)
    {
//        std::cout << ele1 << " " << ele << std::endl;
        atom_idx = std::stoi(ele);
//        std::cout << atom_idx <<std::endl;
        auto feature = ligand_feature_dic[atom_idx];
        atom_x = feature[0];
        atom_y = feature[1];
        atom_z = feature[2];
        vdwrad = feature[17];  // this is van der waal radius.
//        std::cout << "pocket vdwrad " << vdwrad  << std::endl;

        normal_x = (pts[idx * 3] - atom_x)/vdwrad;
        normal_y = (pts[idx * 3 + 1] - atom_y)/vdwrad;
        normal_z = (pts[idx * 3 + 2] - atom_z)/vdwrad;

//        std::cout << normal_x << " " << normal_y << " "<< normal_z <<'\n';

        normals.push_back(normal_x);
        normals.push_back(normal_y);
        normals.push_back(normal_z);
//        normal.push_back(0);

//        std::cout << "feature size: " << feature.size() <<std::endl;
        for (int j=3; j <feature.size(); j++){
//            if (j != 17){features.push_back(feature[j]);}
            features.push_back(feature[j]);
        }
        idx += 1;
//        std::cout << idx << std::endl;
    }
    infile5.close();
    std::cout << "idx is: " << idx <<'\n';
    std::cout << "normals size: " << normals.size() <<'\n';
//    std::cout << "features size: " << features.size() <<'\n';

    string pocket_pt_source_file_name = "./" + ID_name + "_cdk_pocket.txt";
    std::ifstream infile6(pocket_pt_source_file_name);

    idx = 0;
    while (infile6 >> ele1 >> ele)
    {
        atom_idx = std::stoi(ele);
        auto feature = pocket_feature_dic[atom_idx];
        atom_x = feature[0];
        atom_y = feature[1];
        atom_z = feature[2];
        vdwrad = feature[17];

//        auto temp = (lig_num + idx) * 3;
//        std::cout << temp <<std::endl;
//        std::cout << "lig vdwrad " << vdwrad  << std::endl;
        normal_x = (pts[(lig_num + idx) * 3] - atom_x)/vdwrad;
        normal_y = (pts[(lig_num + idx) * 3 + 1] - atom_y)/vdwrad;
        normal_z = (pts[(lig_num + idx) * 3 + 2] - atom_z)/vdwrad;

//        std::cout << "normal is " << normal_x << " " << normal_y << " " << normal_z << std::endl;

        normals.push_back(normal_x);
        normals.push_back(normal_y);
        normals.push_back(normal_z);
//        normal.push_back(1);

        //feature.size is 24, add indicator for ligand/pocket, in total we have 25 features.
//        std::cout << "feature size" << feature.size() << std::endl;
        for (int j=3; j <feature.size(); j++){
//            if (j != 17){features.push_back(feature[j]);}
            features.push_back(feature[j]);
        }
        idx += 1;
    }
    infile6.close();


    // centralize
	int npt = pts.size()/3;
	std::cout << "total point number: " << npt << std::endl;
    for (int n = 0; n < npt; ++n)
    {
        for (int m = 0; m < 3; ++m)
        {
            pts[3 * n + m] -= center[m];
        }
    }


    std::cout << "idx is: " << idx <<'\n';
    std::cout << "normals size: " << normals.size() <<'\n';
    std::cout << "features size: " << features.size() <<'\n';

//    for(int i = 0; i < 9; i++){std::cout << "pts " << i << " "  << pts[i] <<std::endl;}
//    for(int i = 0; i < 21; i++){std::cout << "features " << i << " " << features[i] <<std::endl;}
//    for(int i = 21; i < 42; i++){std::cout << "features " << i << " " << features[i] <<std::endl;}
//    for(int i = 0; i < 9; i++){std::cout << "normals " << i << " "  << normals[i] <<std::endl;}

//    for(int i = 0; i < normals.size(); i++)
//    {
//        if (normals[i] > 1 || normals[i] < -1) {
//            int t = i/3 * 3;
//            std::cout << "normals " << t << " "  << normals[t] <<std::endl;
//            std::cout << "normals " << t+1 << " "  << normals[t+1] <<std::endl;
//            std::cout << "normals " << t+2 << " "  << normals[t+2] <<std::endl;
//        }
//    }

    point_cloud.set_points(pts, normals, features);
    point_cloud.write_points(FLAGS_output_path  + ID_name +"_points.points");
    std::cout << "write to " << FLAGS_output_path  + ID_name +"_points.points\n";


//
//      points.assign(pts, pts + 3*(pro_num+lig_num));
//  // Their normals are (1.0, 0.0, 0.0) and (0.57735, 0.57735, 0.57735)
//  float ns[] = { 1.0, 0.0, 0.0, 0.57735, 0.57735, 0.57735 };
//  normals.assign(ns, ns + 6);
//  // They may also have 4 channel colors (0.5, 0.5, 0.5, 1.0) and (1.0, 0, 0, 0, 0.8)
//  float rgba[] = { 0.5, 0.5, 0.5, 1.0, 1.0, 0, 0, 0, 0.8 };
//  features.assign(rgba, rgba + 8);
//  // Their labels are 0 and 1 respectively
//  float lb[] = { 0.0, 1.0 };
//  labels.assign(lb, lb + 2);



    return 0;
}