import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
import pickle,json
import numpy as np

# Load modules in same folder        
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/utilities')
from xtb_functions import xtb_energy,xtb_geo_opt
from taffi_functions import *
from job_submit import *
from utility import *

def main(argv):

    parser = argparse.ArgumentParser(description='This script will work on a YARP working folder and generate possible catalysis input files')

    #optional arguments                                             
    parser.add_argument('-i', dest='input_folder',
                        help = 'The program expects a specific input xyz file folder')

    parser.add_argument('-o', dest='out_folder',
                        help = 'The program expects a specific output xyz file folder')

    parser.add_argument('-c', dest='catalyst', default='H20', help = 'Select a catalyst' )

    parser.add_argument('-ff', dest='forcefield', default='mmff94', help = 'Force field used to generate product geometry')

    parser.add_argument('--exclude_H2', dest='excludeH2', default=False, action='store_const', const=True, help = 'When set, all H2 formation reaction will be excluded' )
    
    parser.add_argument('--exclude_b3f3', dest='excludeb3', default=False, action='store_const', const=True, help = 'When set, also generate water catalyzed pathways for b3f3 reaction')

    # parse configuration dictionary (c)
    print("parsing calculation configuration...")
    args=parser.parse_args()
    generate_catalysis(args.input_folder,args.out_folder,catalyst=args.catalyst,ff=args.forcefield,excludeH2=args.excludeH2,excludeb3=args.excludeb3)

    return

def generate_catalysis(input_xyz,output_xyz,catalyst='H20',ff='uff',excludeH2=False,excludeb3=False):

    # create folders
    if os.path.isdir(input_xyz) is False:
        print("Expected a working folder with previous YARP calculation results, quit...")
        quit()

    # parse input xyz 
    input_xyzs = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(input_xyz) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])

    for input_file in input_xyzs:

        print("Working on {}...".format(input_file))
        # parse reactant, product geometries
        E,RG,PG = parse_input(input_file)

        # determien bond changes
        bond_break,bond_form,center_atoms,R_adj_mat,R_BE = return_reactive(E,RG,PG,return_Rmats=True)
        H_index = [i for i in center_atoms if E[i] == 'H']
        H2_form = True in [[E[i[0]],E[i[1]]].count('H') == 2 for i in bond_form ]

        if len(bond_break) != 2 and len(bond_break) != 3: 
            print("Only accecpt b2f2 and b3f3 reactions, skip {}...".format(input_file))
            continue

        if len(bond_break) > 2 and excludeb3: 
            print("Exclude a b3f3 reaction")
            continue

        if H2_form and excludeH2: 
            print("Exclude a H2 forming reaction")
            continue

        # generate catalysis pathway
        if len(H_index) == 1:

            # For this case, A-H1, B-C break,and B-H1, A-C form
            # In the new catalysis pathway, A-H1, B-C, R-H2 break, B-H2, R-H1, A-C form (add R-H2 in break list and R-H1 in form list; replace B-H2 by B-H1 in form list)
            N_atoms = len(E)
            cataE,cataG= xyz_parse('catalyst/{}.xyz'.format(catalyst))
            cata_adj   = Table_generator(cataE,cataG)    
            cata_lone,_,_,cata_bond,_ = find_lewis(cataE,cata_adj,q_tot=0,keep_lone=[],return_pref=False,return_FC=True)
            cata_BE = np.diag(cata_lone[0])+cata_bond[0]

            # add catalyst into initial molecules
            E += cataE
            G = np.vstack([RG,cataG])
            add_mat = np.zeros([N_atoms,len(cataE)])
            adj_mat = np.asarray(np.bmat([[R_adj_mat, add_mat], [add_mat.T, cata_adj]]))
            BE_mat  = np.asarray(np.bmat([[R_BE, add_mat], [add_mat.T, cata_BE]]))

            # determine the bond break and bond form 
            new_bond_break = bond_break + [(N_atoms,N_atoms+1)]
            new_bond_form  = []
            for bond in bond_form:
                if H_index[0] not in bond: 
                    new_bond_form += [bond]
                else:
                    B_ind = [ind for ind in bond if ind != H_index[0]][0]

            new_bond_form += [(B_ind, N_atoms+1)]
            new_bond_form += [(H_index[0], N_atoms)]

            # update BE_matrix
            BE_break = break_bonds(BE_mat,new_bond_break)
            N_BE     = form_bonds(BE_break,new_bond_form)
            P_adj_mat = bond_to_adj(N_BE)
            
            # update product geometry
            PG = opt_geo(deepcopy(G),P_adj_mat,E,ff=ff,step=400)
            RG,PG = align_geo(E,G,PG,adj_mat,P_adj_mat,working_folder=os.getcwd(),ff=ff,iter_step=3)

            # write xyz file
            outputname = '_'.join(input_file.split('/')[-1].split('.xyz')[0].split('_')[:3])
            xyz_write('{}/{}-cata.xyz'.format(output_xyz,outputname),E,RG)
            xyz_write('{}/{}-cata.xyz'.format(output_xyz,outputname),E,PG,append_opt=True)

        elif len(H_index) == 2:

            # For b2f2 reaction: A-H1, B-H2 break, A-B, H1-H2 form
            # In the new catalysis pathway, A-H1, B-H2, R-H3 break, two possible new rearrangement A-B, H1-H3, R-H2 form or A-B, H2-H3, R-H1 form
            # For a conditioned b3f3 reaction, A-H1, B-H2, C-D (double bond) break, and an additional R-H3 bond will be added to the list
            # Assume X1 and X2 are originally connected, X1-X2 bond will form, X3-H1, X4-H2 bond will also form, add R-H1 form and replace X3-H1 by X3-H3; or add R-H2 and replace X4-H2 by X4-H3
            N_atoms = len(E)
            cataE,cataG= xyz_parse('catalyst/{}.xyz'.format(catalyst))
            cata_adj   = Table_generator(cataE,cataG)    
            cata_lone,_,_,cata_bond,_ = find_lewis(cataE,cata_adj,q_tot=0,keep_lone=[],return_pref=False,return_FC=True)
            cata_BE = np.diag(cata_lone[0])+cata_bond[0]

            # add catalyst into initial molecules
            E += cataE
            G = np.vstack([RG,cataG])
            add_mat = np.zeros([N_atoms,len(cataE)])
            adj_mat = np.asarray(np.bmat([[R_adj_mat, add_mat], [add_mat.T, cata_adj]]))
            BE_mat  = np.asarray(np.bmat([[R_BE, add_mat], [add_mat.T, cata_BE]]))

            # add R-H3 bond to the break list
            new_bond_break = bond_break + [(N_atoms,N_atoms+1)]

            if len(bond_break) == 2:

                # determine the bond form list 
                atom_ind = []
                for bond in bond_break: atom_ind += [ind for ind in bond if ind not in H_index]

                new_bond_form_1, new_bond_form_2  = [(atom_ind[0], atom_ind[1])], [(atom_ind[0], atom_ind[1])]
                new_bond_form_1 += [(H_index[0], N_atoms), (H_index[1], N_atoms+1)]
                new_bond_form_2 += [(H_index[1], N_atoms), (H_index[0], N_atoms+1)]

            if len(bond_break) == 3:

                # For a conditioned b3f3 reaction, A-H1, B-H2, C-D (double bond) break, and an additional R-H3 bond will be added to the list
                # Assume X1 and X2 are originally connected, X1-X2 bond will form, X3-H1, X4-H2 bond will also form, add R-H1 form and replace X3-H1 by X3-H3; or add R-H2 and replace X4-H2 by X4-H3
                atom_ind = []
                for bond in bond_break: atom_ind += [ind for ind in bond if ind not in H_index]

                # locate the originally bonded atom pair
                X12_ind = [bond for bond in bond_form if R_adj_mat[bond[0]][bond[1]] == 1][0]
                X34_ind = [ind for ind in atom_ind if ind not in X12_ind]
                new_bond_form_1, new_bond_form_2  = [(X12_ind[0], X12_ind[1])], [(X12_ind[0], X12_ind[1])]
                
                if len(X34_ind) != 2: 
                    print("Unexpected bond changes, skip {}...".format(input_file))
                    continue

                if [X34_ind[0],H_index[0]] in bond_form or [H_index[0],X34_ind[0]] in bond_form: X3,X4 = X34_ind[0],X34_ind[1]
                else: X3,X4 = X34_ind[1],X34_ind[0]

                new_bond_form_1 += [(H_index[0], N_atoms), (H_index[1], X4), (N_atoms+1, X3)]
                new_bond_form_2 += [(H_index[1], N_atoms), (H_index[0], X3), (N_atoms+1, X4)]

            for count,new_bond_form in enumerate([new_bond_form_1, new_bond_form_2]):

                # update BE_matrix
                BE_break = break_bonds(BE_mat,new_bond_break)
                N_BE     = form_bonds(BE_break,new_bond_form)
                P_adj_mat = bond_to_adj(N_BE)
            
                # update product geometry
                PG = opt_geo(deepcopy(G),P_adj_mat,E,ff=ff,step=400)
                RG,PG = align_geo(E,G,PG,adj_mat,P_adj_mat,working_folder=os.getcwd(),ff=ff,iter_step=3)

                # write xyz file
                outputname = '_'.join(input_file.split('/')[-1].split('.xyz')[0].split('_')[:3])
                xyz_write('{}/{}-cata{}.xyz'.format(output_xyz,outputname,count+1),E,RG)
                xyz_write('{}/{}-cata{}.xyz'.format(output_xyz,outputname,count+1),E,PG,append_opt=True)

    return

# function to align reactant and product geometry
def align_geo(E,RG,PG,R_adj_mat,P_adj_mat,working_folder='.',ff='uff',iter_step=2,qt=0,unpair=0):
    
    # create opt and xTB folder if is not exist
    if os.path.isdir(working_folder+'/opt-folder') is False:
        os.mkdir(working_folder+'/opt-folder')

    if os.path.isdir(working_folder+'/xTB-folder') is False:
        os.mkdir(working_folder+'/xTB-folder')

    # optimize geometries
    for i in range(iter_step):
        
        # optimize product geometries
        PG = opt_geo(deepcopy(RG),P_adj_mat,E,ff=ff,step=200)
        #product_opt = working_folder+'/opt-folder/product-opt.xyz'
        #xyz_write(product_opt,E,PG)
        #Energy,opted_geo,finish = xtb_geo_opt(product_opt,charge=qt,unpair=unpair,namespace='product',workdir=working_folder+'/xTB-folder',level='normal',output_xyz=product_opt)
        #_,PG  = xyz_parse(opted_geo)
        
        # optimize reactant geometries 
        RG = opt_geo(deepcopy(PG),R_adj_mat,E,ff=ff,step=200)
        #reactant_opt = working_folder+'/opt-folder/reactant-opt.xyz'
        #xyz_write(reactant_opt,E,PG)
        #Energy,opted_geo,finish = xtb_geo_opt(reactant_opt,charge=qt,unpair=unpair,namespace='reactant',workdir=working_folder+'/xTB-folder',level='normal',output_xyz=reactant_opt)
        #E,RG  = xyz_parse(opted_geo)

    return RG,PG


# function to return reactive atoms and bond changes
def return_reactive(E,RG,PG,return_Rmats=False):
    
    # get adjacency matrix
    adj_mat_1 = Table_generator(E,RG)    
    adj_mat_2 = Table_generator(E,PG)    
    
    # apply find lewis
    lone_1,_,_,bond_mat_1,fc_1 = find_lewis(E,adj_mat_1,q_tot=0,keep_lone=[],return_pref=False,return_FC=True)
    lone_2,_,_,bond_mat_2,fc_2 = find_lewis(E,adj_mat_2,q_tot=0,keep_lone=[],return_pref=False,return_FC=True)

    # contruct BE matrix
    BE_1   = np.diag(lone_1[0])+bond_mat_1[0]
    BE_2   = np.diag(lone_2[0])+bond_mat_2[0]
    BE_change = BE_2 - BE_1

    # determine bonds break and bonds form from Reaction matrix
    bond_break = []
    bond_form  = []
    for i in range(len(E)):
        for j in range(i+1,len(E)):
            if BE_change[i][j] < 0:
                bond_break += [(i,j)]
            if BE_change[i][j] > 0:
                bond_form += [(i,j)]
    center_atoms = list(set(sum(bond_break, ())+sum(bond_form, ())))

    if return_Rmats:
        return bond_break,bond_form,center_atoms,adj_mat_1,BE_1 
    else:
        return bond_break,bond_form,center_atoms

# function to transfer bond_mat to adj_mat
def bond_to_adj(BE):
    adj_mat = deepcopy(BE)
    for i in range(len(BE)):
        for j in range(len(BE)):
            if BE[i][j] != 0: adj_mat[i][j] = 1
            
    for i in range(len(BE)): adj_mat[i][i] = 0

    return adj_mat

# Function to break given bond list
def break_bonds(BE,bond_break):
    new_BE = deepcopy(BE)
    if type(bond_break[0])==list or type(bond_break[0])==tuple:
        for bb in bond_break:
            new_BE[bb[0]][bb[1]] -= 1
            new_BE[bb[1]][bb[0]] -= 1
            new_BE[bb[0]][bb[0]] += 1
            new_BE[bb[1]][bb[1]] += 1
    else:
        new_BE[bond_break[0]][bond_break[1]]-=1
        new_BE[bond_break[1]][bond_break[0]]-=1
        new_BE[bond_break[0]][bond_break[0]]+=1
        new_BE[bond_break[1]][bond_break[1]]+=1

    return new_BE

# Function to form given bond list
def form_bonds(BE,bond_form):
    new_BE = deepcopy(BE)
    if type(bond_form[0])==list or type(bond_form[0])==tuple:
        for bf in bond_form:
            new_BE[bf[0]][bf[1]] += 1
            new_BE[bf[1]][bf[0]] += 1
            new_BE[bf[0]][bf[0]] -= 1
            new_BE[bf[1]][bf[1]] -= 1
    else:
        new_BE[bond_form[0]][bond_form[1]]+=1
        new_BE[bond_form[1]][bond_form[0]]+=1
        new_BE[bond_form[0]][bond_form[0]]-=1
        new_BE[bond_form[1]][bond_form[1]]-=1
    return new_BE

if __name__ == "__main__":
    main(sys.argv[1:])

