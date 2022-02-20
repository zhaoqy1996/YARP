import sys,argparse,os,time,math,subprocess,shutil,fnmatch
from copy import deepcopy
import numpy as np
import ast
import collections
#import cPickle as pickle  # python2
import pickle,json
from itertools import combinations 

# all function in taffi
sys.path.append('../utilities')
from taffi_functions import *
from utility import return_smi,return_inchikey,parse_smiles

# import ERS definitions
from ERS_types import *

def main(argv):

    parser = argparse.ArgumentParser(description='This script will enumerate potential product for given reactant following one elementary reaction step.')

    parser.add_argument('coord_files', help = 'The program performs on given 1. a txt file contains a list of smiles strings,'  +\
                        '2. a xyz file or 3. a folder of xyz files of reactant and generates all potential products')

    parser.add_argument('-c', dest='config', default='TCIT-config.txt',
                        help = 'The program expects a configuration file for running TCIT jobs')

    parser.add_argument('-rd', dest='reactant_dict', help = 'One dictionary save all reactant decomposition/transformation info' )

    parser.add_argument('-ff', dest='forcefield', default='mmff94', help = 'force field used to generate product geometry')

    parser.add_argument('-P', dest='phase', default=None, help = 'There are two phases in YARP, phase 1 is unimolecular decomposition/transformation' +\
                        'and phase 2 is bimolecular interaction/combination.')
    
    parser.add_argument('-t', dest='truncate', default='[]',
                        help = '1 refers to removing 3-atom ring compounds,' +\
                               '2 refers to removing 4-atom ring compounds,' +\
                               '3 refers to removing complex ring (bridge) compounds')

    parser.add_argument('--force_update', dest='force_update', default=False, action='store_const', const=True, help = 'When this flag is on, redo ERS enumeration and update db (be careful)')

    parser.add_argument('--apply_TCIT', dest='apply_TCIT', default=False, action='store_const', const=True, help = 'When this flag is on, perform TCIT Hf_298k calculations for the reactant and products')

    parser.add_argument('--b3f3', dest='b3f3', default=False, action='store_const', const=True, help = 'when set, b3f3 enuemration will be performed rather than b2f2' )

    parser.add_argument('--partial_b3f3', dest='Pb3f3', default=False, action='store_const', const=True, help = 'when set, partial b3f3 enuemration will be performed rather than b2f2' )

    # parse configuration dictionary and make output dir
    args=parser.parse_args()    

    if args.phase in ['1','2']:
        phase = int(args.phase)
        
    elif args.phase in [1,2]:
        phase = args.phase
    
    else:
        print("phase must be eigher 1 (unimolecular mode) or 2 (bimolecular mode), it will set to be None if not applicable")
        phase = None

    truncate = ast.literal_eval(args.truncate)

    # load in reactant database
    if os.path.isfile(args.reactant_dict):
        with open(args.reactant_dict,'rb') as f:
            reactant_dict = pickle.load(f)

    else:
        reactant_dict = {}

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
        changed,inchi_ind,reactant_dict = reaction_enumeration(coord_file=inputF,reactant_dict=reactant_dict,ff=args.forcefield,truncate=truncate,phase=phase,\
                                                               force_update=args.force_update,apply_TCIT=args.apply_TCIT,config=args.config,b3f3=args.b3f3,Pb3f3=args.Pb3f3)

        # write dictionary into pickle
        if changed:
            with open(args.reactant_dict, 'wb') as fp:
                pickle.dump(reactant_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


# function to enumerate possible products
def reaction_enumeration(coord_file,reactant_dict,ff,truncate,phase=None,force_update=False,apply_TCIT=False,b3f3=False,Pb3f3=False,config=None):

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
            return False,_,reactant_dict
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
    if len(groups) == 1:

        # obatin the inchikey to check whether this reactant already in database
        inchikey  = return_inchikey(E,G)
        inchi_ind = inchikey.split('-')[0]
        inchi_list= [inchikey]
        print("Inchi index is {}".format(inchi_ind))
        if phase is None: phase = 1

    elif len(groups) == 2:
        
        # get inchikey for each compound seperately
        inchi_list = []
        for group in groups:

            # numver of atoms in this reactant
            N_atom = len(group)

            # Elements and geometry of this reactant 
            frag_E = [E[ind] for ind in group]
            frag_G = np.zeros([N_atom,3])
            for count_i,i in enumerate(group):
                frag_G[count_i,:] = G[i,:]

            # Determine the inchikey of this reactant and inchi_ind
            inchikey   = return_inchikey(frag_E,frag_G)
            inchi_list+= [inchikey]

        inchi_ind  = '-'.join([inchi.split('-')[0] for inchi in sorted(inchi_list)])
        print("Inchi index is {}".format(inchi_ind))
        if phase is None: phase = 2

    else:
        print("Only support 1/2 reactant(s) as input, check your input xyz file")
        quit()

    if inchi_ind in reactant_dict.keys() and "possible_products" in reactant_dict[inchi_ind].keys() and force_update is False:

        changed = False
        print("{}/(inchi index: {}) already in reactant dictionary, direactly take it for next step...".format(coord_file,inchi_ind))

    else:

        # will append content in the reactant dict, turn on the flag
        changed = True

        # create reactant sub-dictionary if not exists 
        if inchi_ind not in reactant_dict:
            reactant_dict[inchi_ind] = {}

        # obatin smiles string
        Rsmiles = return_smi(E,G,adj_mat)

        # apply find lewis to determine bonding information
        lone,bond,core,bond_mat,fc = find_lewis(E,adj_mat,q_tot=qt,return_pref=False,return_FC=True)

        # currently only consider the first bond matrix in the case of resonance structure
        bond_mat= bond_mat[0]
        fc      = fc[0]
        lone    = lone[0]

        # Generate canonical geometry
        N_E,N_adj_mat,N_hash_list,N_G,N_bond_mat,N_dup= canon_geo(E,adj_mat,G,bond_mat,dup=[lone,fc],change_group_seq=True)
        N_lone,N_fc = N_dup
 
        # determine the atom contains formal charge / radicals
        ion_inds = [ count_i for count_i,i in enumerate(N_fc) if i !=0 ]
        keep_lone= [ count_i for count_i,i in enumerate(N_lone) if i%2 != 0]

        # Generate a bonding electron (BE) mateix
        N_BE = np.diag(N_lone)+N_bond_mat
    
        # Generate bond list from BE matrix
        bond_list = []
        for i in range(len(N_BE)):
            for j in range(len(N_BE))[i+1:]:
                bond_list += [[i,j]]*int(N_BE[i][j])
                
        # save related properties in reactant dictionary
        reactant_dict[inchi_ind]["prop"] = {}
        reactant_dict[inchi_ind]["prop"]["E"] = N_E
        reactant_dict[inchi_ind]["prop"]["G"] = N_G
        reactant_dict[inchi_ind]["prop"]["fc"] = N_fc
        reactant_dict[inchi_ind]["prop"]["lone"] = N_lone
        reactant_dict[inchi_ind]["prop"]["smiles"] = Rsmiles
        reactant_dict[inchi_ind]["prop"]["BE_mat"] = N_BE
        reactant_dict[inchi_ind]["prop"]["adj_mat"] = N_adj_mat
        reactant_dict[inchi_ind]["prop"]["charge"] = sum(N_fc)
        reactant_dict[inchi_ind]["prop"]["unpair"] = len(keep_lone)
        reactant_dict[inchi_ind]["prop"]["multiplicity"] = 1+len(keep_lone)
        reactant_dict[inchi_ind]["prop"]["bond_list"] = bond_list
        reactant_dict[inchi_ind]["prop"]["hash_list"] = N_hash_list
        reactant_dict[inchi_ind]["prop"]["inchi_list"]= inchi_list

        # add smiles into the list
        smile_list += Rsmiles.split('.')

        # To regenerate canonical input reactant file
        #xyz_write('canon-reactant.xyz',N_E,N_G)

        # Create a dictionary to store product info
        possible_product = {}
        
        # Type 1 reaction: identify whether exist charge/lone electron contained compounds
        if len(ion_inds) == 0 and len(keep_lone) == 0:

            print("This will be a type 1 reaction")

            if b3f3: possible_product,separated_product = R1_b3f3(reactant_dict[inchi_ind]["prop"],truncate=truncate,ff=ff,phase=phase)
            elif Pb3f3: possible_product,separated_product = R1_b3f3(reactant_dict[inchi_ind]["prop"],truncate=truncate,ff=ff,phase=phase,limit_b3f3=True)
            else: possible_product,separated_product = R1_b2f2(reactant_dict[inchi_ind]["prop"],truncate=truncate,ff=ff,phase=phase)
            
            # apply TCIT
            if apply_TCIT:
                
                # get smiles that need TCIT predictions
                for _,pp in possible_product.items():
                    smile_list += pp["name"].split('.')

                # return TCIT predictions
                TCIT_dict = TCIT_prediction(list(set(smile_list)),config)
                
                # append TCIT predictions into dictionaries
                R_Hf_298 = sum([TCIT_dict[smi]["Hf_298"] for smi in Rsmiles.split('.')])
                reactant_dict[inchi_ind]["prop"]["TCIT_Hf"] = R_Hf_298
                for _,pp in possible_product.items():
                    try:
                        pp["TCIT_Hf"] = sum([TCIT_dict[smi]["Hf_298"] for smi in pp["name"].split('.')])
                    except:
                        # if missing CAVs, use R_Hf_298 for instead
                        print(pp["name"])
                        pp["TCIT_Hf"] = R_Hf_298

            # append possible product in reactant_dict
            reactant_dict[inchi_ind]["possible_products"] = possible_product
            reactant_dict[inchi_ind]["separated_product"] = separated_product

            ''' check output when debuging
            with open('possible_product.p', 'wb') as fp:
                pickle.dump(possible_product, fp, protocol=pickle.HIGHEST_PROTOCOL)

            with open('separated_product.p', 'wb') as fp:
                pickle.dump(separated_product, fp, protocol=pickle.HIGHEST_PROTOCOL)
                
            for pp in possible_product.keys():
                for ind,G in enumerate(possible_product[pp]['G_list']):
                    xyz_write("P.xyz",E,G)
                    os.system("cp {} P_{}_{}.xyz; cat P.xyz >> P_{}_{}.xyz; rm P.xyz".format(args.coord_files,pp,ind,pp,ind))
            '''

        elif len(ion_inds) == 2 and len(keep_lone) == 0 and sum(N_fc) == 0:

            print("This will be a type 2 reaction (zwritter-ionic) ")
            
            possible_product,separated_product = R2_zwitterionic(reactant_dict[inchi_ind]["prop"],anion_ind=N_fc.index(-1),cation_ind=N_fc.index(1),truncate=truncate,ff=ff,phase=phase)

            # append possible product in reactant_dict
            reactant_dict[inchi_ind]["possible_products"] = possible_product
            reactant_dict[inchi_ind]["separated_product"] = separated_product
        elif len(keep_lone)==1 and len(ion_inds)==0: # for uni-radical case
            print('ERS for an uni-radical compound')
            possible_product, separated_product=R1_b1f1(reactant_dict[inchi_ind]['prop'], truncate=truncate, ff=ff, phase=phase)
            p,s = R1_b2f2(reactant_dict[inchi_ind]['prop'], truncate=truncate, ff=ff, phase=phase)
            keys=p.keys()
            n_len=len(possible_product.keys())
            for i in keys:
                possible_product[int(i)+n_len]=p[i]                                                                                                                                                               
            separated_product.update(s)
            reactant_dict[inchi_ind]['possible_products']=possible_product
            reactant_dict[inchi_ind]['separated_product']=separated_product
        else:
            print("Current version only supports neutral and no radical systems...")
            quit()
            # Type 2: lone electron case. For this kind of reaction, perform b1f1 for three times  
            # Type 3: ion-containing case.  For this kind of reaction, perform R3_b1f1 for three times  
            # Type 4: hybrid case 

    return changed,inchi_ind,reactant_dict

# Function to call TCIT to give thermochemistry predictions (in kJ/mol)
def TCIT_prediction(smiles_list,config_file):

    # load in configs
    configs = parse_configuration(config_file)

    # load in TCIT database
    if os.path.isfile(configs["tcit_db"]) is True:
        with open(configs["tcit_db"],"r") as f:
            TCIT_dict = json.load(f) 
    else:
        TCIT_dict = {}
        
    # check the compounds without TCIT predictions
    smiles_list = [smi for smi in smiles_list if smi not in TCIT_dict.keys()]

    # write smiles into config file
    with open(configs["target_file"],'w') as f:
        for M_smiles in smiles_list:
            f.write("{}\n".format(M_smiles))
            
    # run TCIT calculation
    output=subprocess.Popen("python {} -c {}".format(configs["tcit_path"],config_file),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')

    # re-load in TCIT database
    with open(configs["tcit_db"],"r") as f:
        TCIT_dict = json.load(f) 
        
    return TCIT_dict

# load in config
def parse_configuration(config_file):

    # Convert inputs to the proper data type
    if os.path.isfile(config_file) is False:
        print("ERROR in python_driver: the configuration file {} does not exist.".format(config_file))
        quit()

    # Process configuration file for keywords
    keywords = ["TCIT_path","input_type","target_file","database","G4_db","TCIT_db","ring_db","xyz_task"]
    keywords = [ _.lower() for _ in keywords ]

    list_delimiters = [ "," ]  # values containing any delimiters in this list will be split into lists based on the delimiter
    space_delimiters = [ "&" ] # values containing any delimiters in this list will be turned into strings with spaces replacing delimites
    configs = { i:None for i in keywords }    

    with open(config_file,'r') as f:
        for lines in f:
            fields = lines.split()
            
            # Delete comments
            if "#" in fields:
                del fields[fields.index("#"):]
            
            # Parse keywords
            l_fields = [ _.lower() for _ in fields ] 
 
            for i in keywords:
                if i in l_fields:

                    # Parse keyword value pair
                    ind = l_fields.index(i) + 1
                    if len(fields) >= ind + 1:
                        configs[i] = fields[ind]

                        # Handle delimiter parsing of lists
                        for j in space_delimiters:
                            if j in configs[i]:
                                configs[i] = " ".join([ _ for _ in configs[i].split(j) ])
                        for j in list_delimiters:
                            if j in configs[i]:
                                configs[i] = configs[i].split(j)
                                break
                                
                    # Break if keyword is encountered in a non-comment token without an argument
                    else:
                        print("ERROR in python_driver: enountered a keyword ({}) without an argument.".format(i))
                        quit()

    # Makesure detabase folder exits
    if os.path.isfile(configs["database"]) is False: 
        print("No such data base")
        quit()
            
    if os.path.isfile(configs["ring_db"]) is False:
        print("No such ring correction database, please check config file, existing...")
        quit()

    if os.path.isfile(configs["g4_db"]) is False:
        print("No such G4 result database, please check config file, existing...")
        quit()

    if len(os.listdir(configs["xyz_task"]) ) > 0:
        subprocess.Popen("rm {}/*".format(configs["xyz_task"]),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0]
   
    return configs

if __name__ == "__main__":
    main(sys.argv[1:])
