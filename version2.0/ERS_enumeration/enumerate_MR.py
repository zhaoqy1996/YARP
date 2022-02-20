import sys,argparse,os,time,math,subprocess,shutil,fnmatch
import json,pickle
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/utilities')
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/model_reaction')
from utility import return_smi,return_inchikey,parse_smiles
from ERS_types import return_freeze,is_special_atom,break_bonds,form_bonds,bond_to_adj,array_unique,generate_bond_form
from taffi_functions import *
from id_reaction_type import *

def main(argv):

    parser = argparse.ArgumentParser(description='This script will enumerate potential product and generate model reactions for given reactant following one elementary reaction step.')

    parser.add_argument('coord_files', help = 'The program performs on given 1. a txt file contains a list of smiles strings,'  +\
                        '2. a xyz file or 3. a folder of xyz files of reactant and generates all potential products')

    parser.add_argument('-MR', dest='MR_dict', default= '/depot/bsavoie/data/YARP_database/b2f2_model_reaction/MR_dict.txt', help = 'One dictionary save all model reaction info' )

    parser.add_argument('-ff', dest='forcefield', default='mmff94', help = 'force field used to generate product geometry')

    parser.add_argument('-o', dest='output', default='/depot/bsavoie/data/YARP_database/b2f2_model_reaction/xyz_files',\
                        help = 'There are two phases in YARP, phase 1 is unimolecular decomposition/transformation' +\
                        'and phase 2 is bimolecular interaction/combination.')
    
    # parse configuration dictionary and make output dir
    args=parser.parse_args()    
    output_path = args.output  
  
    # performing reaction enumeration
    if os.path.isdir(args.coord_files):
        inputs = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(args.coord_files) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])

    elif os.path.isfile(args.coord_files) and '.txt' in args.coord_files:
        inputs = []
        with open(args.coord_files,"r") as f:
            for lines in f:
                if lines.split()[0] == '#': continue
                inputs.append(lines.split()[0])

    elif os.path.isfile(args.coord_files) and '.xyz' in args.coord_files:
        inputs = [args.coord_files]

    else:
        print("Unexpected inputs, check the input folder/file...")

    for inputF in inputs:
        print("Working on {}...".format(inputF))
        success = b2f2_model_reaction_enumeration(coord_file=inputF,MR_dict_path=args.MR_dict,output_path=output_path,ff=args.forcefield)

### Further todo list
### 1. generate a hash value for model reaction type
### 2. add more options, like b3f3
### 3: better way to store model reactions

# function to enumerate possible products
def b2f2_model_reaction_enumeration(coord_file,MR_dict_path,output_path,ff='mmff94'):

    # initialize smile list
    smile_list = []

    # generate adjacency matrix
    if '.xyz' in coord_file:
        E,G     = xyz_parse(coord_file)
        qt      = parse_q(coord_file)
        adj_mat = Table_generator(E,G)
    else:
        readin,E,G,qt = parse_smiles(coord_file)
        if not readin:
            print("Invalid smiles string ({}), skip this compounds...".format(coord_file))
            return False
        adj_mat = Table_generator(E,G)

    # check fragments of the reactant, if this is a multi-molecular reaction, seperate each reactant
    gs = graph_seps(adj_mat)
    groups = []   # initialize the list for index of all reactants
    loop_ind = []
    for i in range(len(gs)):
        if i not in loop_ind:
            new_group = [count_j for count_j,j in enumerate(gs[i,:]) if j >= 0]
            loop_ind += new_group
            groups   += [new_group]

    # if only one group, this is an unimolecular reactant
    if len(groups) > 1: 
        print("Only support uni-molecular reactant as input, check your input!")
        return False

    # Create a dictionary to store product info
    possible_product = {}
        
    # Check whether exist charge/lone electron contained compounds
    if qt != 0:
        print("Expecct neutral closed-shell reactants, check inputs...")
        return False

    # perform b2f2 enuemration
    MR_b2f2(E,G,adj_mat,MR_dict_path,ff=ff,output=output_path)
            
    return True

# define type one reaction break 2 form 2 (b2f2) elementary reaction step
def MR_b2f2(E,G,adj_mat,MR_dict_path,qt=0,ff='uff',output='/depot/bsavoie/data/YARP_database/b2f2_model_reaction/xyz_files'):

    # apply find lewis to determine bonding information
    lone,bond,core,bond_mat,fc = find_lewis(E,adj_mat,q_tot=qt,return_pref=False,return_FC=True)

    # currently only consider the first bond matrix in the case of resonance structure
    bond_mat,fc,lone = bond_mat[0],fc[0],lone[0]

    # Generate canonical geometry
    E,adj_mat,hash_list,G,bond_mat,N_dup = canon_geo(E,adj_mat,G,bond_mat,dup=[lone,fc],change_group_seq=True)
    lone,fc = N_dup
 
    # determine the atom contains formal charge / radicals
    ion_inds = [ count_i for count_i,i in enumerate(fc) if i != 0 ]
    keep_lone= [ count_i for count_i,i in enumerate(lone) if i % 2 != 0]

    if len(keep_lone) != 0:
        print("Expect closed-shell species...")
        return False

    # Generate a bonding electron (BE) mateix
    BE = np.diag(lone)+bond_mat

    # Generate bond list from BE matrix
    bond_list = []
    for i in range(len(BE)):
        for j in range(len(BE))[i+1:]:
            bond_list += [[i,j]]*int(BE[i][j])
                
    # Create a list of potential products, BE_list and hash_list are used to identify whether a new prodcut is found or not
    BE_list = []
    total_hash_list = []
    BE_list.append(deepcopy(BE))     
    total_hash_list.append(deepcopy(hash_list))
    
    # generate all possible C_N^2 combinations
    comb = [bc for bc in combinations(bond_list, 2)]

    # initialzie some lists
    total_break = []
    bond_change_list = []

    # loop over all bond changes
    for bond_break in comb:

        # load in model reaction database
        if os.path.isfile(MR_dict_path):
            MR_dict = load_MR_dict(MR_dict_path)
        else:
            MR_dict = {}

        # can't break up double-bond in one step
        if bond_break[0] == bond_break[1]: continue

        # if the same bond is aleardy broken, skip this bond
        if bond_break not in total_break: total_break += [bond_break]
        else: continue
        
        # find activate atoms
        atom_list = []
        for bb in bond_break: atom_list += bb
        
        # Calculate the hash value of the atom involved in the bond breaking, will be used to avoid duplicated reactions
        break_hash = sorted([sorted([hash_list[b[0]],hash_list[b[1]]]) for b in bond_break])

        # Determine whether exist common atom
        common_atom = [item for item, count in collections.Counter(atom_list).items() if count > 1]        

        # Determine possible reaction based on number of common atom
        if len(common_atom) == 0:

            # if there is no common atom, two possible new arrangements
            BE_break  = break_bonds(BE,bond_break)
            bonds_form= generate_bond_form(bond_break,atom_list)

            for bond_form in bonds_form:

                # generate product BE matrix
                N_BE = form_bonds(BE_break,bond_form)
                N_adj_mat = bond_to_adj(N_BE)

                # Calculate the hash value of the atom involved in the bond forming
                form_hash  = sorted([sorted([hash_list[f[0]],hash_list[f[1]]]) for f in bond_form])
                change_hash= break_hash + form_hash

                if array_unique(change_hash,bond_change_list)[0]:
                    bond_change_list.append(change_hash)
                    MR_type,bond_break,bond_form = return_MR_types(E,Radj_mat=adj_mat,Padj_mat=N_adj_mat,R_BE=BE,P_BE=N_BE,return_bond_change=True)
                    if MR_type not in MR_dict.keys(): 

                        # update MR_dict
                        index = len(MR_dict) + 1
                        MR_dict[MR_type] = index
                        update_MR_dict(MR_dict_path,MR_type,index)

                        # generate model reaction
                        try:
                            model_reaction = gen_model_reaction(E,G,adj_mat,[BE],bond_break,bond_form,gens=1,canonical=True)
                            xyz_write("{}/MR_{}.xyz".format(output,index),model_reaction['E'],model_reaction['R_geo'])
                            xyz_write("{}/MR_{}.xyz".format(output,index),model_reaction['E'],model_reaction['P_geo'],append_opt=True)
                        except:
                            print("Have trouble generating model reactions, skip this...")
                            pass
                    else:
                        print("model reaction {} already in the databse, skip...".format(MR_type))

        elif len(common_atom) == 1:

            # check "special" atom
            special_flag,connect_atoms = is_special_atom(common_atom[0],atom_list,BE,E,limited=True)
            
            # identify whether it is possible for the common atom to be a "special" atom
            if special_flag:
                
                not_common_atom = [atom for atom in atom_list if atom not in common_atom]

                for c_atom in connect_atoms:

                    # generate BE matrix
                    BE_break = break_bonds(BE,bond_break)
                    N_BE     = form_bonds(BE_break,[not_common_atom,[common_atom[0],c_atom] ] )
                    N_BE[c_atom][c_atom]                         -= 2
                    N_BE[common_atom[0]][common_atom[0]]         += 2
                    N_adj_mat = bond_to_adj(N_BE)

                    # Calculate the hash value of the atom involved in the bond forming
                    form_hash  = sorted([ sorted([hash_list[not_common_atom[0]],hash_list[not_common_atom[1]]]),sorted([hash_list[common_atom[0]],hash_list[c_atom] ]) ])
                    change_hash= break_hash + form_hash

                    if array_unique(change_hash,bond_change_list)[0]:
                        bond_change_list.append(change_hash)
                        MR_type,bond_break,bond_form = return_MR_types(E,Radj_mat=adj_mat,Padj_mat=N_adj_mat,R_BE=BE,P_BE=N_BE,return_bond_change=True)
                        if MR_type not in MR_dict.keys(): 
                            index = len(MR_dict) + 1
                            MR_dict[MR_type] = index
                            update_MR_dict(MR_dict_path,MR_type,index)
                            
                            # generate model reaction
                            try:
                                model_reaction = gen_model_reaction(E,G,adj_mat,[BE],bond_break,bond_form,gens=1,canonical=True)
                                xyz_write("{}/MR_{}.xyz".format(output,index),model_reaction['E'],model_reaction['R_geo'])
                                xyz_write("{}/MR_{}.xyz".format(output,index),model_reaction['E'],model_reaction['P_geo'],append_opt=True)
                            except:
                                print("Have trouble generating model reactions, skip this...")
                                pass

                        else:
                            print("model reaction {} already in the databse, skip...".format(MR_type))
            
    return 

# function to get reaction type and model reaction
def return_MR_types(E,Radj_mat,Padj_mat,R_BE,P_BE,return_bond_change=False):

    # determine the reaction matrix
    BE_change = P_BE - R_BE
    
    # determine bonds break and bonds form from Reaction matrix
    bond_break = []
    bond_form  = []
    for i in range(len(E)):
        for j in range(i+1,len(E)):
            if BE_change[i][j] == -1:
                bond_break += [(i,j)]
                
            if BE_change[i][j] == 1:
                bond_form += [(i,j)]
            
    # id reaction type
    try:
        reaction_type,seq,bond_dis = id_reaction_types(E,[Radj_mat,Padj_mat],bond_changes=[bond_break,bond_form],gens=1,algorithm="matrix",return_bond_dis=True)
        if return_bond_change: return reaction_type,bond_break,bond_form
        else: return reaction_type

    except:
        print("Have trouble getting reaction type, skip...")
        print([bond_break,bond_form])
        return False

def load_MR_dict(MR_dict_file):
    MR_dict = {}
    with open(MR_dict_file,'r') as f:
        for lc,lines in enumerate(f):
            if lc == 0: continue
            fields = lines.split()
            if len(fields) != 2: continue
            MR_dict[fields[1]] = fields[0]

    return MR_dict

def update_MR_dict(MR_dict_file,MR_type,index):
    with open(MR_dict_file,'a') as f:
        f.write("{}\t{}\n".format(index,MR_type))

if __name__ == "__main__":
    main(sys.argv[1:])
