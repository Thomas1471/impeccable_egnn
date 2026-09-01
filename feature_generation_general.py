from scipy.spatial.distance import cdist 
from multiprocessing import Pool 
import os 
import argparse 
import numpy as np 
import pandas as pd 
import os, time 
import traceback 
from collections import Counter
import torch
import MDAnalysis as mda

from feature_generation_graph import create_pose_graph

import torch

from feature_generation_graph import create_pose_graph


#Main graph-generation script. It converts IMPECCABLE docking outputs into atom-level protein-ligand graphs.


def collect_file_paths(args):
    master_dict = {} 
    print(args)
    # DCD Files 
    if not os.path.isdir(args.dcd_dir): 
        raise Exception("Directory location for dcd files doesn't exist")

    list_of_files = os.listdir(args.dcd_dir)
    list_of_files.sort()
    for x in list_of_files:
        lig_num = x.split('.')[1]
        path = os.path.join(args.dcd_dir, x)
        if os.path.isfile(path) is False:
            master_dict.pop(x, None)
        else:
            master_dict[lig_num] = [path]

    # PDB Files 
    if not os.path.isdir(args.pdb_dir): 
        raise Exception("Directory location for pdb files doesn't exist") 
    
    list_of_files = os.listdir(args.pdb_dir) 
    list_of_files.sort() 
    for x in list_of_files: 
        lig_num = x.split('.')[1] 
        path = os.path.join(args.pdb_dir, x) 
        if os.path.isfile(path) is False:
            master_dict.pop(x, None)
        else:
            master_dict[lig_num].append(path)

    if args.mode == "training":
        # MMPBSA Files
        if not os.path.isdir(args.mmpbsa_dir):
            raise Exception("Directory location for pdb files doesn't exist")
        list_of_files = os.listdir(args.mmpbsa_dir)
        list_of_files.sort()
        for x in list_of_files:
            path = os.path.join(args.mmpbsa_dir, x)
            path = os.path.join(path, "dg_poses.dat")
            if os.path.isfile(path) is False:
                master_dict.pop(x, None)
            else:
                master_dict[x].append(path)

    # SMILES File 
    with open(args.smiles_file, "r") as f: 
        for line in f: 
            smi = line.split()[0] 
            key = int(line.split()[1])
            if str(key) in master_dict.keys(): 
                master_dict[str(key)].append(smi) 

    output_files = [] 
    for x in master_dict:
        if args.mode == 'training' and len(master_dict[x]) == 4:
            output_files.append(master_dict[x])
        elif args.mode == 'inference' and len(master_dict[x]) == 3:
            output_files.append(master_dict[x])
    print("len(output_files) ", len(output_files))

    chunked_files = [output_files[x:x + args.batch_size] for x in range(0, len(output_files), args.batch_size)]
    return chunked_files 
        

def get_binding_site_protein_atoms(
    univ,
    frame_idx,
    atom_names,
    residue_cutoff,
):
    """
    Select protein H/C/N/O/S atoms belonging to residues within
    residue_cutoff Angstrom of the ligand.

    A residue is included if any of its H/C/N/O/S atoms is within
    residue_cutoff of any ligand H/C/N/O/S atom.
    """
    univ.trajectory[frame_idx]

    ligand_atoms = univ.select_atoms(
        "(not protein) and ("
        + " or ".join([f"name {atom}*" for atom in atom_names])
        + ")"
    )

    protein_atoms = univ.select_atoms(
        "protein and ("
        + " or ".join([f"name {atom}*" for atom in atom_names])
        + ")"
    )

    if len(ligand_atoms) == 0 or len(protein_atoms) == 0:
        return None, None, None, None

    # Residues with any atom within residue_cutoff of the ligand.
    nearby = univ.select_atoms(
        "protein and around "
        f"{residue_cutoff} "
        "("
        "not protein and ("
        + " or ".join([f"name {atom}*" for atom in atom_names])
        + ")"
        ")"
    )

    if len(nearby) == 0:
        return None, None, None, None

    selected_protein_atoms = nearby.residues.atoms.select_atoms(
        "protein and ("
        + " or ".join([f"name {atom}*" for atom in atom_names])
        + ")"
    )

    if len(selected_protein_atoms) == 0:
        return None, None, None, None

    prot_coords = selected_protein_atoms.positions.copy()
    lig_coords = ligand_atoms.positions.copy()

    prot_types = [
        atom.name[0] for atom in selected_protein_atoms
    ]

    lig_types = [
        atom.name[0] for atom in ligand_atoms
    ]

    return prot_coords, lig_coords, prot_types, lig_types

def compute_contacts_training(output_files): 
    G = [] 
    rows = []

    for dcd_file, pdb_file, mmpbsa_file, smiles in output_files: 
        try: 
            # Create mda universe
            
            print(f"Trying pdb_file={pdb_file}")#NEW
            print(f"Trying dcd_file={dcd_file}")#NEW
            univ = mda.Universe(pdb_file, dcd_file) 

            # For UID:
            lig_uid = pdb_file.split('/')[-1].split('.')[1]  
            atom_names = ["H", "C", "N", "O", "S"] 
            
            mmpbsa_values = [] 

            with open(mmpbsa_file, 'r') as f: 
                mmpbsa_values = f.readlines() 
            mmpbsa_values = [x.split()[0] for x in mmpbsa_values]
           

            if len(univ.trajectory) <= len(mmpbsa_values):
                num_poses = len(univ.trajectory)
            else:
                num_poses = len(mmpbsa_values)

         
            for i in range(num_poses): 
               
            
                (
                    prot_coords_all,
                    lig_coords_all,
                    prot_types,
                    lig_types,
                ) = get_binding_site_protein_atoms(
                    univ=univ,
                    frame_idx=i,
                    atom_names=atom_names,
                    residue_cutoff=args.residue_cutoff,
                )

                if prot_coords_all is None or lig_coords_all is None:
                    continue
                

                #Create graph
                graph = create_pose_graph(
                        prot_coords_all,
                        lig_coords_all,
                        prot_types,
                        lig_types,
                        cutoff=args.interaction_cutoff)
        





            



                uid = lig_uid + "_" + str(i)

                graph.y = torch.tensor([float(mmpbsa_values[i])], dtype=torch.float)
                G.append(graph)
            
                
                rows.append({
                    "uid": uid,
                    "compound_num": lig_uid,
                    "frame": i,
                    "mmpbsa": float(mmpbsa_values[i]),
                    "smiles": smiles,
                    "pdb_file": pdb_file,
                    "dcd_file": dcd_file,
                    "graph_idx": len(G)-1
                })
                 
  
        except Exception: 
            print("Exception occurred: ") 
            print(traceback.format_exc()) 
            print("len(mmpbsa_values): ", len(mmpbsa_values), ", num_poses: ", num_poses, "\n")

    df = pd.DataFrame(rows)

    if len(df) > 0:
        uid_counter = Counter(df["uid"])
        keys_greater_than_1 = [key for key, value in uid_counter.items() if value > 1]
        if len(keys_greater_than_1) > 1:
            print("Number of repeated keys: ", len(keys_greater_than_1))
        else:
            print("All clear")

        print(
            "number of unique id numbers: ",
            len(df["uid"].unique()),
            " number of total id numbers: ",
            len(df["uid"])
        )

    return G, df
 



def compute_contacts_inference(output_files): 

    G = [] 
    rows = []

    for dcd_file, pdb_file, smiles in output_files: 
        try: 
            
            if os.path.getsize(pdb_file) == 0 or os.path.getsize(dcd_file) == 0:
                
                print(f"Skipping empty file: {pdb_file}, {dcd_file}")
                continue 
            univ = mda.Universe(pdb_file, dcd_file) 

            # For UID:
            lig_uid = pdb_file.split('/')[-1].split('.')[1]  
            atom_names = ["H", "C", "N", "O", "S"] 
            
           


            num_poses = len(univ.trajectory)


        
            for i in range(num_poses): 
               
                (
                    prot_coords_all,
                    lig_coords_all,
                    prot_types,
                    lig_types,
                ) = get_binding_site_protein_atoms(
                    univ=univ,
                    frame_idx=i,
                    atom_names=atom_names,
                    residue_cutoff=args.residue_cutoff,
                )
               
                if prot_coords_all is None or lig_coords_all is None:
                    continue
                
            

                

                #Create graph
                graph = create_pose_graph(
                        prot_coords_all,
                        lig_coords_all,
                        prot_types,
                        lig_types,
                        cutoff=args.interaction_cutoff)
        


                uid = lig_uid + "_" + str(i)


                G.append(graph)
            
                
                rows.append({
                    "uid": uid,
                    "compound_num": lig_uid,
                    "frame": i,
                    "smiles": smiles,
                    "pdb_file": pdb_file,
                    "dcd_file": dcd_file,
                    "graph_idx": len(G)-1
                })
  
        except Exception: 
            print("Exception occurred: ") 
            print(traceback.format_exc()) 

 
    df = pd.DataFrame(rows)

    if len(df) > 0:
        uid_counter = Counter(df["uid"])
        keys_greater_than_1 = [key for key, value in uid_counter.items() if value > 1]
        if len(keys_greater_than_1) > 1:
            print("Number of repeated keys: ", len(keys_greater_than_1))
        else:
            print("All clear")

        print(
            "number of unique id numbers: ",
            len(df["uid"].unique()),
            " number of total id numbers: ",
            len(df["uid"])
        )

    return G, df


def main(args): 
    output_files = collect_file_paths(args)

    with Pool(processes=args.num_processes) as pool: 
        if args.mode == 'training':
            out = pool.map(compute_contacts_training, output_files)
        elif args.mode == 'inference':
            out = pool.map(compute_contacts_inference, output_files)

    # out is a list of tuples: [(G1, df1), (G2, df2), ...]
    all_graphs = []
    all_dfs = []

    for graphs, df in out:
        all_graphs.extend(graphs)
        if not df.empty:

            all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError("No valid samples were processed.")
    df = pd.concat(all_dfs, axis=0, ignore_index=True)
   

 # Save metadata dataframe
    df.to_csv(args.prepared_data_file, index=False)

    # Save graphs
    graph_file = args.prepared_data_file.replace(".csv", "_graphs.pt")
    tmp_graph_file = graph_file + ".tmp"

    torch.save(
        all_graphs,
        tmp_graph_file,
        _use_new_zipfile_serialization=False,
    )

    os.replace(tmp_graph_file, graph_file)

    print(f"Saved metadata to {args.prepared_data_file}")
    print(f"Saved graphs to {graph_file}")


    df2 = pd.read_csv(args.prepared_data_file)

    print("After saving the file and reading it back in:")

    print("number of unique id numbers: ", len(df2['uid'].unique()), " number of total id numbers: ", len(df2['uid']))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    # Ligands are split into batches and processed in parallel.
    # For each pose, a graph is generated and metadata is collected.
    parser.add_argument("--batch_size", required=False, default=10, type=int)
    #A directory path containing the pdb files
    parser.add_argument("--pdb_dir", required=True, type=str)
    #A directory path containing the dcd files
    parser.add_argument("--dcd_dir", required=True, type=str)
    #A directory path containing the mmpbsa files
    parser.add_argument("--mmpbsa_dir", required=False, type=str)
    #A file path containing the smiles strings for each of the ligands
    parser.add_argument("--smiles_file", required=True, type=str)
    # Output CSV for pose metadata. The corresponding graph objects are saved
    # to a companion .pt file derived from this path.
    parser.add_argument("--prepared_data_file", required=False, default="graph_metadata.csv", type=str)
    #The number of workers to be spawned to generate the features
    parser.add_argument("--num_processes", required=False, type=int, default=32)
    #Flag to distinguish between training and inference mode
    parser.add_argument("--mode", required=True, type=str)

    #Residue Cutoff
    parser.add_argument(
    "--residue_cutoff",
    type=float,
    default=6.0,
    help="Residue inclusion cutoff from ligand in Angstroms",
    )

    parser.add_argument(
        "--interaction_cutoff",
        type=float,
        default=12.0,
        help="Atom-level protein-ligand interaction edge cutoff in Angstroms",
    )
    args = parser.parse_args()

    assert args.mode in ['training', 'inference']

    if args.mode == 'training':
        assert args.mmpbsa_dir

    print("feature gen args: ", args)

    t1 = time.time()
    main(args)
    t2 = time.time()
    print("Run time: ", t2 - t1)