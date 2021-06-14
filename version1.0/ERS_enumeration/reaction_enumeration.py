import sys,argparse,os,time,math,subprocess,shutil
from copy import deepcopy
import numpy as np
import ast
import collections
from itertools import combinations 

# all function in taffi
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/utilities')
from taffi_functions import *

# import ERS definitions
from ERS_types import *

def main(argv):

    parser = argparse.ArgumentParser(description='This script will enumerate all of potential product for given reactant following elementary reaction steps.' +\
                                     'run it as [python reaction_enumeration.py <reactant.xyz> -o <output_folder name> -t [1,2] --generate_image]')

    parser.add_argument('coord_files', help = 'The program performs on given xyz file of reactant and generates all potential products' +\
                        '(make sure the reactive part and surface part are seperate in sequence)')

    parser.add_argument('-o', dest='output', default='product',help = 'Controls the output folder name' )

    parser.add_argument('-N', dest='N_loop', default=None, help = 'Performing elementary reaction for N_loop times' )

    parser.add_argument('-ff', dest='forcefield', default='uff', help = 'Which force-field used to generate 3D geometry' )

    parser.add_argument('-t', dest='truncate', default='[]',
                        help = '1 refers to removing 3-atom ring compounds,' +\
                               '2 refers to removing 4-atom ring compounds,' +\
                               '3 refers to removing complex ring (bridge) compounds,' +\
                               '4 refers to applying check_ring function to remove non-common ring sturctures')

    parser.add_argument('--b3f3', dest='b3f3', default=0, action='store_const', const=1, 
                        help = 'when set, b3f3 enuemration will be performed' )

    parser.add_argument('--generate_image', dest='generate_image', default=0, action='store_const', const=1, 
                        help = 'when set, png files will be generated' )

    # parse configuration dictionary and make output dir
    args=parser.parse_args()    
    
    if args.generate_image == 1: generate_image = True
    else: generate_image = False

    if args.b3f3 == 1: apply_b3f3 = True
    else: apply_b3f3 = False

    ff = args.forcefield
        
    truncate = ast.literal_eval(args.truncate)
    
    # create working directory
    if os.path.exists(args.output) is False:
        os.makedirs(args.output)
        os.makedirs("{}/mol_files".format(args.output))
        os.makedirs("{}/reaction_channel".format(args.output))
        if generate_image:
            os.makedirs("{}/png_files".format(args.output))
    
    # create a txt file to record the product
    with open("{}/product_record.txt".format(args.output),"w") as f:
        f.write("Product list for {}\n".format(args.coord_files))
        f.write("{:<10s} {:<30s} {:<15s} {:<100s}\n".format('Index',"Indicator","N-adj_mats","Pathways"))

    # generate adj & bond matrix
    E,G   = xyz_parse(args.coord_files)
    qt    = parse_q(args.coord_files)  
    adj_mat = Table_generator(E,G)
    lones,bonds,cores,bond_mat,fc=find_lewis(E,adj_mat,q_tot=qt,return_pref=False,return_FC=True)

    # only consider the first bond matrix in the case of resonance structure
    bond_mat= bond_mat[0]
    fc      = fc[0]
    lone    = lones[0]

    # Generate canonical geometry for the reactant
    E,adj_mat,hash_list,G,bond_mat,N_dup= canon_geo(E,adj_mat,G,bond_mat,dup=[lone,fc],change_group_seq=False)
    lone_electron,fc = N_dup
 
    # determine the atom contains formal charge / radicals
    ion_inds = [ count_i for count_i,i in enumerate(fc) if i !=0 ]
    keep_lone= [ count_i for count_i,i in enumerate(lone_electron) if i%2 != 0]

    # Generate a bonding electron (BE) mateix
    BE=np.diag(lone_electron)+bond_mat

    # Generate bond list from BE matrix
    bond_list = []
    for i in range(len(BE)):
        for j in range(len(BE))[i+1:]:
            bond_list += [[i,j]]*int(BE[i][j])

    # To regenerate canonical input reactant file
    print("Canonical reactant geometry is written into canon-reactant.xyz in the working folder, please make sure use this file in GSM step!") 
    xyz_write('canon-reactant.xyz',E,G)

    # Create a list of potential products, BE_list and hash_list are used to identify whether a new prodcut is found or not
    BE_list = []
    total_hash_list = []
    BE_list.append(deepcopy(BE))     
    total_hash_list.append(deepcopy(hash_list))

    # Create a dictionary to store product info
    potential_product = {}
    potential_product[0]={}
    potential_product[0]["E"]             = deepcopy(E)
    potential_product[0]["G_list"]        =[deepcopy(G)]
    potential_product[0]["adj_mat_list"]  =[deepcopy(adj_mat)]
    potential_product[0]["bond_mat_list"] =[deepcopy(BE)]
    potential_product[0]["fc"]            = fc
    potential_product[0]["depth"]         = 0

    # initialize network nodes and edges, each element is edges is (a,b,i) which means from a to b and i refers to the corresponding adj_mat
    edges = []
    
    # Type 1 reaction: identify whether exist charge/lone electron contained compounds
    # For this kind of reaction, perform b2f2 twice for default 
    if len(ion_inds) == 0 and len(keep_lone) == 0:
        print("This will be a type 1 reaction")

        # First loop: break two bonds and enumerate all of possible cases
        if os.path.isdir("{}/reaction_channel/pp_0".format(args.output)) is False:
            os.makedirs("{}/reaction_channel/pp_0".format(args.output))

        if not apply_b3f3:
            edges = R1_b2f2(0,BE_list,total_hash_list,potential_product,outputname=args.output,bond_list=bond_list,truncate=truncate,generate_png=generate_image,edge_list=edges,ff=ff)        
        else:
            edges = R1_b3f3(0,BE_list,total_hash_list,potential_product,outputname=args.output,bond_list=bond_list,truncate=truncate,generate_png=generate_image,edge_list=edges,ff=ff)        

        first_layer_product = deepcopy(potential_product)

        # Set N_loop
        if args.N_loop is None:
            N_loop = 1
        else:
            N_loop = int(args.N_loop)

        # if N >= 2, continue to next loop 
        N_count = 1
        for loop in range(N_loop-1):

            N_depth = len(BE_list) - N_count

            for start_ind in range(N_depth):
                print("Loop over depth={} compounds: {}/{}".format(loop+1,start_ind+1,N_depth))

                if os.path.isdir("{}/reaction_channel/pp_{}".format(args.output,start_ind+N_count)) is False:
                    os.makedirs("{}/reaction_channel/pp_{}".format(args.output,start_ind+N_count))

                if not apply_b3f3:
                    edges = R1_b2f2(start_ind+N_count,BE_list,total_hash_list,potential_product,outputname=args.output,G=potential_product[start_ind+N_count]["G_list"][0],\
                                    bond_list=[],truncate=truncate,generate_png=generate_image,edge_list=edges,ff=ff)
                else:
                    edges = R1_b3f3(start_ind+N_count,BE_list,total_hash_list,potential_product,outputname=args.output,G=potential_product[start_ind+N_count]["G_list"][0],\
                                    bond_list=[],truncate=truncate,generate_png=generate_image,edge_list=edges,ff=ff)

                print("Now {} potential product are found...".format(len(potential_product.keys()) - 1))

            N_count += N_depth
        
        edge_list = sorted(set([(i[0],i[1]) for i in edges]))
        #print("All of reaction channels are: {}".format(edge_list))

    else:
        print("Now only support neutral and no radical systems...")
        quit()
        # Type 2: lone electron case. For this kind of reaction, perform b1f1 for three times  
        # Type 3: ion-containing case.  For this kind of reaction, perform R3_b1f1 for three times  
        # Type 4: hybrid case 

    # Generate product list
    #'''
    if os.path.isdir("{}/frag_compounds".format(args.output)) is False:
        os.makedirs("{}/frag_compounds".format(args.output))
        os.makedirs("{}/frag_compounds/xyz_files".format(args.output))
        os.makedirs("{}/frag_compounds/mol_files".format(args.output))

    # loop over all of the products and get inchikey for each component
    frag_inchi = []
    for N_p in list(potential_product.keys())[1:]:

        # obatin the adjacency matrix
        adj_mat= potential_product[N_p]["adj_mat_list"][0]

        # apply graph_seps to determine whether all atoms (vertices) are connected
        gs     = graph_seps(adj_mat)
        groups = []
        loop_ind = []

        # loop over all atoms, if not connected, write the atom index in a new group
        for i in range(len(gs)):
            if i not in loop_ind:
                new_group = [count_j for count_j,j in enumerate(gs[i,:]) if j >= 0]
                loop_ind += new_group
                groups   += [new_group]

        # generate smiles string and inchikey for all of the components
        smile_list  = []
        inchi_list  = []

        # each group in "groups" list is a separated molecule
        for group in groups:

            # determine the number of atoms, Element list and geometry of this molecule
            N_atom = len(group)
            frag_E = [potential_product[N_p]["E"][ind] for ind in group]
            frag_G = np.zeros([N_atom,3])
            for count_i,i in enumerate(group):
                frag_G[count_i,:] = potential_product[N_p]["G_list"][0][i,:]

            # determine the adjacency matrix and write a mol file 
            frag_adj_mat= adj_mat[group,:][:,group]
            mol_write("{}/frag_compounds/check.mol".format(args.output),frag_E,frag_G,frag_adj_mat)

            # apply openbabel to obatin the inchikeys
            substring = "obabel -imol {}/frag_compounds/check.mol -oinchikey".format(args.output)
            output    = subprocess.Popen(substring,shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0] 
            inchikey  = output.replace('\n','')
            inchi_list+= [inchikey]

            # if this molecule not shown before, copy the mol file to target folder
            if inchikey not in frag_inchi:
                frag_inchi += [inchikey]
                xyz_write("{}/frag_compounds/xyz_files/{}.xyz".format(args.output,inchikey),frag_E,frag_G)
                shutil.copy2("{}/frag_compounds/check.mol".format(args.output),"{}/frag_compounds/mol_files/{}.mol".format(args.output,inchikey))

            # apply openbabel to obatin the smiles string            
            substring= "obabel -imol {}/frag_compounds/check.mol -ocan".format(args.output)
            output   = subprocess.Popen(substring,shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0] 
            smile = output.replace('\n','').split()[0]
            smile_list += [smile]

        # name this product, which might be multiple molecules by combining the smiles strings together
        name = '.'.join(smile_list)

        # number of different adj_mats are counted
        N_adj    = len(potential_product[N_p]["adj_mat_list"])

        # all of the pathways leading to this product are counted
        pathways = [(edge[0],edge[2]) for edge in edges if edge[1] == N_p]
        start_inds = sorted(list(set([i[0] for i in pathways])))
        adj_dict={}
        for i in start_inds:
            adj_dict[i]=sorted([j[1] for j in pathways if j[0]==i])
        pathways = [(i,adj_dict[i]) for i in sorted(adj_dict.keys())]
        
        with open("{}/product_record.txt".format(args.output),"a") as f:
            f.write("{:<10s} {:<30s} {:<15s} {:<100s}\n".format(str(N_p),name,str(N_adj),str(pathways)))
        
        with open("{}/frag_compounds/frag_inchi.txt".format(args.output),"a") as g:
            g.write("pp_{}\t\t".format(N_p))
            for inchi in inchi_list:
                g.write("{}\t".format(inchi))
            g.write("\n")
    #'''

    return

if __name__ == "__main__":
    main(sys.argv[1:])
