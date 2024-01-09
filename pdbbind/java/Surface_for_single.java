import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.nio.file.Path; 
import java.nio.file.Paths;


import javax.vecmath.Point3d;

import org.openscience.cdk.ChemFile;
import org.openscience.cdk.geometry.surface.AdaptiveNumericalSurface;
import org.openscience.cdk.geometry.surface.NumericalSurface;
import org.openscience.cdk.geometry.surface.Point_Type;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.io.Mol2Reader;
import org.openscience.cdk.io.PDBReader;
import org.openscience.cdk.io.iterator.IteratingSDFReader;
import org.openscience.cdk.silent.SilentChemObjectBuilder;
import org.openscience.cdk.io.MDLV2000Reader;
import org.openscience.cdk.tools.manipulator.ChemFileManipulator;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.vecmath.Point3d;

import org.openscience.cdk.ChemFile;
import org.openscience.cdk.geometry.surface.AdaptiveNumericalSurface;
import org.openscience.cdk.geometry.surface.NumericalSurface;
import org.openscience.cdk.geometry.surface.Point_Type;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.io.Mol2Reader;
import org.openscience.cdk.io.PDBReader;
import org.openscience.cdk.io.iterator.IteratingSDFReader;
import org.openscience.cdk.silent.SilentChemObjectBuilder;
import org.openscience.cdk.io.MDLV2000Reader;
import org.openscience.cdk.tools.manipulator.ChemFileManipulator;

public class Surface_for_single {
	//public javax.vecmath.Point3d[] getAllSurfacePoints();
	
		public static String get_id(String filename) {
			String[] parts = filename.split(".");
			String part1 = parts[0];
			String[] parts1 = part1.split("_");
			String id = parts1[0];
			return id;
		}
		
		public static void handle_single_mol2(String filename, String path, String id, int tess_level, String tess_type, String file_type, boolean append) {
			
			try {
				File file = new File(path + "/" + id + "/" + filename);
				Mol2Reader reader = new Mol2Reader(new FileInputStream(file));
				ChemFile crambin = reader.read(new ChemFile());   
				// ChemFile crambin = reader.read(new ChemFile());  generate warning: org.openscience.cdk.config.atomtypes.AtomTypeHandle WARN: Unrecognized hybridization in config file: tetrahedral
				// refer to https://github.com/johnmay/cdk/blob/master/base/core/src/main/java/org/openscience/cdk/config/atomtypes/AtomTypeHandler.java
				// Another warning: org.openscience.cdk.io.Mol2Reader WARN: Not reading molecule qualifiers
				// refer to https://github.com/cdk/cdk/blob/master/storage/io/src/main/java/org/openscience/cdk/io/Mol2Reader.java
				List<IAtomContainer> containers= ChemFileManipulator.getAllAtomContainers(crambin);
				String write_filename = path + "/" + id + "/" + id + "_cdk_"+ file_type + ".xyz";
				String atom_filename = path + "/" + id + "/" + id + "_cdk_"+ file_type + ".txt";
				//FileOutputStream outputStream = new FileOutputStream(write_filename);
				BufferedWriter writer = new BufferedWriter(new FileWriter(write_filename, append));
				BufferedWriter writer1 = new BufferedWriter(new FileWriter(atom_filename, append));
				System.out.println("container size: " + containers.size());
				for (int i = 0; i< containers.size(); i++) {
					IAtomContainer container = containers.get(i);
//					String description = container.toString();
					
//					AdaptiveNumericalSurface new_surface = new AdaptiveNumericalSurface(container, 0, tess_level, tess_type);
					NumericalSurface new_surface = new NumericalSurface(container, 0, tess_level, tess_type);
//					System.out.println(container);
//					System.out.println(description);
//					System.out.println(new_surface);
					try {
					//new_surface.calculateSurface();
					//Point3d[] points= new_surface.getAllSurfacePoints();
					ArrayList<org.openscience.cdk.geometry.surface.Point_Type> point_types = new_surface.getAllPointswithAtomType();

//					System.out.println(points);
					
					System.out.println(point_types.size());
					for (int j = 0; j < point_types.size(); j++) {
						org.openscience.cdk.geometry.surface.Point_Type point_type = point_types.get(j);
						Point3d coord = (point_type).getCoord();
						int atom = ((Point_Type) point_type).getAtom();
						int atom_index = ((Point_Type) point_type).getIndex();
						String str = coord.x + " " + coord.y + " " + coord.z + "\n";
//						if (j == 1){System.out.println(str);}
						writer.write(str);
//						writer1.write(atom + "\n");
						writer1.write(atom + " " + atom_index + "\n");
					}} catch (Exception ex) {System.out.println("remind QQ Error: " + ex);}
				}
				/*IChemObject pdb_mol = reader.read();*/
				reader.close();
				
				writer.close();
				writer1.close();
				System.out.println("Finished " + write_filename);
			}  catch (Exception ex) {System.out.println(ex);}
			//return container;
		}
		
		public static void handle_single_pdb(String filename, String path, String id, int tess_level, String tess_type, String file_type) {
			//String id = get_id(filename);
//			System.out.println(filename);
//			String[] parts = filename.split("_");
			//System.out.println(parts);
//			String id = parts[0];
//			System.out.println(id);

			
			try {
				File file = new File(path + "/" + id + "/" + filename);
				PDBReader reader = new PDBReader(new FileInputStream(file));
				//IAtomContainer container = reader.read(SilentChemObjectBuilder.getInstance().newInstance(IAtomContainer.class));
				ChemFile crambin = reader.read(new ChemFile());
				List<IAtomContainer> containers= ChemFileManipulator.getAllAtomContainers(crambin);
				String write_filename = path + "/" + id + "/" + id + "_cdk_"+ file_type + ".xyz";
				String atom_filename = path + "/" + id + "/" + id + "_cdk_"+ file_type + ".txt";
				//FileOutputStream outputStream = new FileOutputStream(write_filename);
				BufferedWriter writer = new BufferedWriter(new FileWriter(write_filename));
				BufferedWriter writer1 = new BufferedWriter(new FileWriter(atom_filename));
				for (int i = 0; i< containers.size(); i++) {
					IAtomContainer container = containers.get(i);
//					String description = container.toString();
                    System.out.println(containers.size());
					
//					AdaptiveNumericalSurface new_surface = new AdaptiveNumericalSurface(container, 0, tess_level, tess_type);
					NumericalSurface new_surface = new NumericalSurface(container, 0, tess_level, tess_type);
//					System.out.println(container);
//					System.out.println(description);
//					System.out.println(new_surface);
					try {
					ArrayList<org.openscience.cdk.geometry.surface.Point_Type> point_types = new_surface.getAllPointswithAtomType();
//					System.out.println(points);
					System.out.println(point_types.size());
					for (int j = 0; j < point_types.size(); j++) {
						org.openscience.cdk.geometry.surface.Point_Type point_type = point_types.get(j);
						Point3d coord = (point_type).getCoord();
						int atom = ((Point_Type) point_type).getAtom();
						int atom_index = ((Point_Type) point_type).getIndex();
						String str = coord.x + " " + coord.y + " " + coord.z + "\n";
//						if (j == 1){System.out.println(str);}
						writer.write(str);
//						writer1.write(atom + "\n");
						writer1.write(atom + " " + atom_index + "\n");
					}} catch (Exception ex) {System.out.println(ex);}
				}
				/*IChemObject pdb_mol = reader.read();*/
				reader.close();
				
				writer.close();
				writer1.close();
				System.out.println("Finished " + write_filename);
			}  catch (Exception ex) {System.out.println(ex);}
			//return container;
		}
		
		public static void handle_single_complex(String filename, String path, String id, int tess_level, String tess_type, String file_type) {
			// need revise: to separate ligand/protein.
			try {
				File file = new File(path + "/" + id + "/" + filename);
				PDBReader reader = new PDBReader(new FileInputStream(file));
				ChemFile crambin = reader.read(new ChemFile());
				List<IAtomContainer> containers= ChemFileManipulator.getAllAtomContainers(crambin);
				String write_filename = path + "/" + id + "/" + id + "_cdk_"+ file_type + ".xyz";
				String atom_filename = path + "/" + id + "/" + id + "_cdk_"+ file_type + ".txt";
				BufferedWriter writer = new BufferedWriter(new FileWriter(write_filename));
				BufferedWriter writer1 = new BufferedWriter(new FileWriter(atom_filename));
				for (int i = 0; i< containers.size(); i++) {
					IAtomContainer container = containers.get(i);					
					AdaptiveNumericalSurface new_surface = new AdaptiveNumericalSurface(container, 0, tess_level, tess_type);
					try {
					ArrayList<org.openscience.cdk.geometry.surface.Point_Type> point_types = new_surface.getAllPointswithAtomType();
					System.out.println(point_types.size());
					for (int j = 0; j < point_types.size(); j++) {
						org.openscience.cdk.geometry.surface.Point_Type point_type = point_types.get(j);
						Point3d coord = (point_type).getCoord();
						int atom = ((Point_Type) point_type).getAtom();
						int atom_index = ((Point_Type) point_type).getIndex();
						String str = coord.x + " " + coord.y + " " + coord.z + "\n";
						writer.write(str);
//						writer1.write(atom + "\n");
						writer1.write(atom + " " + atom_index + "\n");
					}} catch (Exception ex) {System.out.println(ex);}
				}
				reader.close();
				
				writer.close();
				writer1.close();
				System.out.println("Finished " + write_filename);
			}  catch (Exception ex) {System.out.println(ex);}
		}
		
		public static void handle_single_id(File complex_id, String path, int tess_level, String tess_type, Boolean protein_flag, String source) {
			//File folder = new File("your/path");
			String id_string = complex_id.getName();
			File[] listOfFiles = complex_id.listFiles();
			
			if (source.equals("pdbbind")) {
				boolean append = false;
				for (int i = 0; i < listOfFiles.length; i++) {
					String file_name = listOfFiles[i].getName();
					
					if (file_name.endsWith("ligand.mol2")){
						System.out.println("Start work on: " + file_name);
						handle_single_mol2(file_name, path, id_string, tess_level, tess_type, "ligand", append);
						System.out.println("Finished: " + file_name);
						append = false;
					}
					else if(file_name.endsWith("pocket.pdb")) {
						System.out.println("Start work on: " + file_name);
						handle_single_pdb(file_name, path, id_string, tess_level, tess_type, "pocket");
						System.out.println("Finished: " + file_name);
					}
					else if(protein_flag && file_name.endsWith("protein.pdb")) {
						System.out.println("Start work on: " + file_name);
						handle_single_pdb(file_name, path, id_string, tess_level-2, tess_type, "protein");
						System.out.println("Finished: " + file_name);
					}
				} // end for
				
			} // end if source
			else if (source.equals("pdbbank")) {
				boolean append = false;
				for (int i = 0; i < listOfFiles.length; i++) {
					String file_name = listOfFiles[i].getName();
					if(file_name.endsWith("_withHs.pdb")) {
						handle_single_complex(file_name, path, id_string, tess_level-2, tess_type, "protein");
					}
				}// end for 
			}
			else if (source.equals("astex")) {
				boolean append = false;
				for (int i = 0; i < listOfFiles.length; i++) {
					String file_name = listOfFiles[i].getName();

					if (file_name.endsWith("ligand.mol2")){
						System.out.println("Start work on: " + file_name);
						handle_single_mol2(file_name, path, id_string, tess_level, tess_type, "ligand", append);
						System.out.println("Finished: " + file_name);
						append = false;
					}
					else if(file_name.endsWith("protein.mol2")) {
						System.out.println("Start work on: " + file_name);
						handle_single_mol2(file_name, path, id_string, tess_level, tess_type, "pocket", append);
						System.out.println("Finished: " + file_name);
					}
				} // end for

			}

			else {
				System.out.println("This File Source is not supported: " + source);
			}
			
		}
		public static void main(String[] args) {
			
			int tess_level = Integer.parseInt(args[0]);
			String tess_type = args[1];
			String source = args[2];
			
			Path currentRelativePath = Paths.get("");
			String s = currentRelativePath.toAbsolutePath().toString(); 
			String[] array = s.split("/");
			
//			String complex_id = array[array.length-1];
			array = Arrays.copyOf(array, array.length - 1);
			String dataset_path = String.join("/", array);
			
			File complex_file = new File(s);
			
			if (complex_file.isDirectory()) {
				String id_string = complex_file.getName();
				handle_single_id(complex_file, dataset_path, tess_level, tess_type, false, source);
			} 
			else {
				System.out.println(s + "is not a folder");
			}
			
		}
}

