###### This file contains different ERS reaction types ######
###### Currently only R1_B2f2 is valid #######
import sys

# all function in taffi
sys.path.append('../utilities')
from taffi_functions import *

################ utilize functions ############
# function to transfer bond_mat to adj_mat
def bond_to_adj(BE):
    adj_mat = deepcopy(BE)
    for i in range(len(BE)):
        for j in range(len(BE)):
            if BE[i][j] != 0: adj_mat[i][j] = 1
            
    for i in range(len(BE)): adj_mat[i][i] = 0

    return adj_mat

# function to identify whether this is a "special" atom. A special case in b2f2 is when an atom is shared between the two bonds,
# we have to determien if there is potential bond rearrangement
def is_special_atom(ind,atom_list,BE,E,limited=False):
    
    flag = False

    # determine the nerghboring atom of target atom
    connect = [count_i for count_i,i in enumerate(BE[ind]) if (i > 0 and ind != count_i)]
    
    # determine the nerghboring atoms which don't participate in the bond changes
    other_atoms = [i for i in connect if i not in atom_list]
    
    # initialize the connect_atom list which is nerghboring atom of elements in other_atoms
    connect_atom = []

    # we set two criteria, one is loose criteria which is related to limited=False and the other one is strict.
    if len(other_atoms) > 0 and not limited:
        # loose criteria: as long as the "other nerghboring atoms" can offer additional lone elctron to the target atom
        for other_atom in other_atoms:
            new_connect = [count_i for count_i,i in enumerate(BE[other_atom]) if ( i > 0 and count_i != other_atom and count_i != ind) ]  
            new_connect_E = [E[count_i] for count_i in new_connect if E[count_i] != 'H']
            if len(new_connect) < 3 and len(new_connect_E)==0 and E[other_atom] != E[ind] and BE[other_atom][other_atom] > 0:
                flag = True
                connect_atom += [other_atom]

    elif len(other_atoms) > 0 and limited:
        # strict criteria: "carbon monoxide" criteria, only if the "other nerghboring atoms" is an end atom 
        for other_atom in other_atoms:
            new_connect = [count_i for count_i,i in enumerate(BE[other_atom]) if ( i > 0 and count_i != other_atom and count_i != ind) ]  
            if len(new_connect) == 0 and BE[other_atom][other_atom] > 0:
                flag = True
                connect_atom += [other_atom]

    return flag,connect_atom

# Function to break given bond list
def break_bonds(BE,bond_break):
    new_BE = deepcopy(BE)
    for bb in bond_break:
        new_BE[bb[0]][bb[1]] -= 1
        new_BE[bb[1]][bb[0]] -= 1
        new_BE[bb[0]][bb[0]] += 1
        new_BE[bb[1]][bb[1]] += 1
    return new_BE

# Function to form given bond list
def form_bonds(BE,bond_form):
    new_BE = deepcopy(BE)
    for bf in bond_form:
        new_BE[bf[0]][bf[1]] += 1
        new_BE[bf[1]][bf[0]] += 1
        new_BE[bf[0]][bf[0]] -= 1
        new_BE[bf[1]][bf[1]] -= 1
    return new_BE

# Function to generate all possible bond rearrangement (new bond forming)
def generate_bond_form(bond_break,atom_list,lone=[]):

    bond_form = []

    # neutral, no radical case
    if len(lone) == 0:

        if len(bond_break) == 2:
            bond_form += [ [ [bond_break[0][0],bond_break[1][0]],[bond_break[0][1],bond_break[1][1]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[1][1]],[bond_break[0][1],bond_break[1][0]] ] ]

        elif len(bond_break) == 3:
            bond_form += [ [ [bond_break[0][0],bond_break[1][0]],[bond_break[0][1],bond_break[2][0]],[bond_break[1][1],bond_break[2][1]] ] ] 
            bond_form += [ [ [bond_break[0][0],bond_break[1][0]],[bond_break[0][1],bond_break[2][1]],[bond_break[1][1],bond_break[2][0]] ] ] 
            bond_form += [ [ [bond_break[0][0],bond_break[1][1]],[bond_break[0][1],bond_break[2][0]],[bond_break[1][0],bond_break[2][1]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[1][1]],[bond_break[0][1],bond_break[2][1]],[bond_break[1][0],bond_break[2][0]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[2][0]],[bond_break[0][1],bond_break[1][0]],[bond_break[1][1],bond_break[2][1]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[2][0]],[bond_break[0][1],bond_break[1][1]],[bond_break[1][0],bond_break[2][1]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[2][1]],[bond_break[0][1],bond_break[1][0]],[bond_break[1][1],bond_break[2][0]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[2][1]],[bond_break[0][1],bond_break[1][1]],[bond_break[1][0],bond_break[2][0]] ] ]
      
        else:
            print("expect two/three bonds to be broken due to ERS definition, exit...")
            quit()
            
    elif len(lone) == 1:
        if len(bond_break) != 2:
            print("Can't count such case, quit...")
            quit()

        if lone[0] not in atom_list:
            # normal case
            bond_form += [ [ [bond_break[0][0],bond_break[1][0]],[bond_break[0][1],bond_break[1][1]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[1][1]],[bond_break[0][1],bond_break[1][0]] ] ]

            # radical involved case
            bond_form += [ [ [bond_break[0][0],lone[0]],[bond_break[0][1],bond_break[1][0]] ] ] 
            bond_form += [ [ [bond_break[0][0],lone[0]],[bond_break[0][1],bond_break[1][1]] ] ] 
            bond_form += [ [ [bond_break[0][1],lone[0]],[bond_break[0][0],bond_break[1][0]] ] ] 
            bond_form += [ [ [bond_break[0][1],lone[0]],[bond_break[0][0],bond_break[1][1]] ] ] 
            bond_form += [ [ [bond_break[1][0],lone[0]],[bond_break[0][0],bond_break[1][1]] ] ] 
            bond_form += [ [ [bond_break[1][0],lone[0]],[bond_break[0][1],bond_break[1][1]] ] ] 
            bond_form += [ [ [bond_break[1][1],lone[0]],[bond_break[0][0],bond_break[1][0]] ] ] 
            bond_form += [ [ [bond_break[1][1],lone[0]],[bond_break[0][1],bond_break[1][0]] ] ] 
         
        else:
            # normal case
            bond_form += [ [ [bond_break[0][0],bond_break[1][0]],[bond_break[0][1],bond_break[1][1]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[1][1]],[bond_break[0][1],bond_break[1][0]] ] ]

            # radical involved case
            nonconnect = [ bb for bb in bond_break if lone[0] not in bb][0]
            bond_form += [ [ [lone[0],nonconnect[0]],[lone[0],nonconnect[1]] ] ]

    return bond_form

# This function is used to identify number of rings and number of atoms in each ring
def identify_rings(E,adj_mat):
    rings=[]
    ring_size_list=range(9)[3:] # identify up to 8-atom ring structure
    for ring_size in ring_size_list:
        for j,Ej in enumerate(E):
            is_ring,ring_ind = ring_atom(adj_mat,j,ring_size=ring_size)
            if is_ring and ring_ind not in rings:
                rings += [ring_ind]
    rings=[list(ring_inds) for ring_inds in rings]
    
    return rings

# Determine freezing atom list (benzene,...,etc)
def return_freeze(E,adj_mat,BE):
    rings_index = identify_rings(E,adj_mat)
    freeze = []
    for ring_index in rings_index:
        conj_flag=True
        for ind in ring_index:
            n_bond = sorted([ int(BE[ind][oind]) for oind in ring_index if adj_mat[ind][oind] == 1])
            if n_bond != [1,2]:
                conj_flag=False
        if conj_flag:
            freeze += ring_index
        
    return freeze
            
# Determine whether exist 4 bonds in BE matrix
def check_bond_condition(BE):
    flag= True
    max_bonds = [max([j for count_j,j in enumerate(i) if count_j != count_i] ) for count_i,i in enumerate(BE) ]
    if max(max_bonds) > 3:
        flag=False
        
    return flag

# Determine whether 
def check_fused_condition(ring_index,BE):
    flag = True
    ring_index = [ind for ind in ring_index if len(ind) <= 4]
    if len(ring_index) >= 2:
        combs = combinations(range(len(ring_index)), 2)
        for comb in combs:
            ring_atoms = ring_index[comb[0]]+ring_index[comb[1]]
            N_common = len(ring_atoms) - len(set(ring_atoms))
            if N_common >= 2:
                flag = False
                return flag

    return flag
    
# Determine whether a ring contained compounds satisfies following rules:
# 1. No triple bond in a ring 
# 2. No ring atom has two double bonds
# 3. No double bond in a 3-members ring
# 
def check_ring_condition(ring_index,bond_mat):
    flag = True
    ring_atoms = []
    three_ring = []
    for ring in ring_index:
        ring_atoms += ring
        if len(ring) == 3:
            three_ring += [ring]

    ring_atoms = list(set(ring_atoms))
    for atom in ring_atoms:
        bond_list = [int(i) for count_i,i in enumerate(bond_mat[atom]) if count_i != atom]
        if 3 in bond_list or bond_list.count(2) == 2:
            flag = False
            return flag

    if len(three_ring) >= 1:
        for ring in three_ring:
            comb = combinations(ring, 2) 
            for pair in comb:
                if bond_mat[pair[0]][pair[1]] >= 2.0:
                    flag = False
                    return flag
    return flag

# Determine whether this is a three members ring
# is return flag is false, such product will be removed
def check_3members_ring(ring_index):
    flag = True
    ring_number = [len(ring) for ring in ring_index]
    #if 3 in ring_number or 4 in ring_number:
    if 3 in ring_number:
        flag = False
        return flag

    return flag

# Determine whether this is a four members ring
# is return flag is false, such product will be removed
def check_4members_ring(ring_index):
    flag = True
    ring_number = [len(ring) for ring in ring_index]
    if 4 in ring_number:
        flag = False
        return flag

    return flag

# Determine whether this is a complex fused ring (bridge ring)
def check_bridge(ring_index):
    flag = True
    if len(ring_index) >= 2:
        combs = combinations(range(len(ring_index)), 2)
        for comb in combs:
            ring_atoms = ring_index[comb[0]]+ring_index[comb[1]]
            N_common = len(ring_atoms) - len(set(ring_atoms))
            if N_common >= 3 and N_common < min(len(ring_index[comb[0]]),len(ring_index[comb[1]])):
                flag = False
                return flag

    return flag

# return hash value function
def return_hash(elements,adj_mat):

    # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
    mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                 'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                 'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                 'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                 'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                 'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                 'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                 'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    # Canonicalize by sorting the elements based on hashing
    masses = [ mass_dict[i] for i in elements ]
    hash_list = [ atom_hash(i,adj_mat,masses) for i in range(len(elements)) ]

    return hash_list

# define type one reaction break 2 form 2 (b2f2) elementary reaction step
def R1_b2f2(start_ind,new_BE_list,total_hash_list,potential_product,outputname,G=[],bond_list=[],truncate=[],generate_png=True,edge_list=[],reactive_list=[],ff='uff'):

    # obatin elements, adj_mat, BE matrix
    E        = potential_product[start_ind]["E"]
    adj_mat  = potential_product[start_ind]["adj_mat_list"][0]
    BE       = potential_product[start_ind]["bond_mat_list"][0]
    depth    = potential_product[start_ind]["depth"] + 1

    # get hash list for the starting compound
    hash_list= return_hash(E,adj_mat)

    # determine the index of next product being enumerated
    begin_ind= len(new_BE_list)

    # get freezed atoms index (like the benzene ring), those bonds involved will not breake  
    freeze   = return_freeze(E,adj_mat,BE)

    # If no geometry is given, the start geometry directly take the first one
    if G==[]:
        G    = potential_product[start_ind]["G_list"][0]

    # Generate geo-opt needed list
    geo_opt_list = []
    
    # Generate breakable bonds list
    if bond_list == []:
        for i in reactive_list:
            for j in reactive_list[i+1:]:
                bond_list += [[i,j]]*int(BE[i][j])

    # generate all possible C_N^2 combinations
    comb = combinations(bond_list, 2)

    # initialzie some lists
    total_break = []
    bond_change_list = []

    # loop over all bond changes
    for bond_break in comb:

        if bond_break not in total_break:
            total_break += [bond_break]
        else:
            continue

        # can't break up double-bond in one step
        if bond_break[0] == bond_break[1]: 
            continue

        # can't break up freezed bonds
        if (bond_break[0][0] in freeze and bond_break[0][1] in freeze) or (bond_break[1][0] in freeze and bond_break[1][1] in freeze):
            continue
        
        atom_list = []
        for bb in bond_break: 
            atom_list += bb
        
        # Calculate the hash value of the atom involved in the bond breaking
        break_hash=sorted([sorted([hash_list[b[0]],hash_list[b[1]]]) for b in bond_break])

        # Determine whether exist common atom
        common_atom = [item for item, count in collections.Counter(atom_list).items() if count > 1]        

        # Determine possible reaction based on number of common atom
        if len(common_atom) == 0:

            # if there is no common atom, two possible new arrangements
            BE_break = break_bonds(BE,bond_break)
            bonds_form = generate_bond_form(bond_break,atom_list)

            for bond_form in bonds_form:

                N_BE = form_bonds(BE_break,bond_form)
                N_adj_mat = bond_to_adj(N_BE)

                # Apply canonical operation only to detemine whether this is a new molecule or a duolicated one
                N_E,N_Adj_mat,N_hash_list,N_BE_canon=canon_geo(E,N_adj_mat,bond_mat=N_BE)

                # Calculate the hash value of the atom involved in the bond forming
                form_hash=sorted([sorted([hash_list[f[0]],hash_list[f[1]]]) for f in bond_form])
                change_hash = break_hash + form_hash

                # check whether all of the conditions are satisfied
                ring_index = identify_rings(N_E,N_adj_mat)
                if reactive_list != []:
                    ring_index =[ring_ind for ring_ind in ring_index if False not in [index in reactive_list for index in ring_ind] ]
                    NR_BE= N_BE[reactive_list,:][:,reactive_list]
                    check_flag = check_bond_condition(NR_BE) and check_ring_condition(ring_index,NR_BE) and check_fused_condition(ring_index,NR_BE) 
                    
                else:
                    check_flag = check_bond_condition(N_BE) and check_ring_condition(ring_index,N_BE) and check_fused_condition(ring_index,N_BE) 

                if 1 in truncate:
                    check_flag = (check_flag and check_3members_ring(ring_index))
                
                if 2 in truncate:
                    check_flag = (check_flag and check_4members_ring(ring_index))

                if 3 in truncate:
                    check_flag = (check_flag and check_bridge(ring_index))

                # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
                check_flag = (check_flag and  array_unique(change_hash,bond_change_list)[0])

                if array_unique(N_BE_canon,new_BE_list)[0] is True and array_unique(N_hash_list,total_hash_list)[0] is True and check_flag:

                    # A new product is found !
                    new_BE_list.append(deepcopy(N_BE_canon))
                    total_hash_list.append(deepcopy(N_hash_list))
                    bond_change_list.append(change_hash)

                    N_p = len(new_BE_list) - 1
                    potential_product[N_p]                 = {}
                    potential_product[N_p]["E"]            = deepcopy(E)
                    potential_product[N_p]["G_list"]       =[deepcopy(G)]
                    potential_product[N_p]["adj_mat_list"] =[deepcopy(N_adj_mat)]
                    potential_product[N_p]["bond_mat_list"]=[deepcopy(N_BE)]
                    potential_product[N_p]["depth"]        = depth
                    potential_product[N_p]["start_ind"]    = start_ind
                    edge_list    += [(start_ind,N_p,0)]
                    geo_opt_list += [(N_p,0)]

                elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:

                    # append change_hash into the change list
                    bond_change_list.append(change_hash)

                    # This is be same product but allows for different reaction pathways
                    N_p = total_hash_list.index(N_hash_list)

                    # if N_p = start, self reaction, ignore it
                    if N_p != start_ind and N_p != 0:
                        # check whether this adj_mat exist in adj_mat_list or not
                        if array_unique(N_adj_mat,potential_product[N_p]["adj_mat_list"])[0]:
                            potential_product[N_p]["G_list"]       += [deepcopy(G)]
                            potential_product[N_p]["adj_mat_list"] += [deepcopy(N_adj_mat)] 
                            potential_product[N_p]["bond_mat_list"]+= [deepcopy(N_BE)]
                            edge_list += [(start_ind,N_p,len(potential_product[N_p]["adj_mat_list"])-1)]
                            geo_opt_list += [(N_p,len(potential_product[N_p]["adj_mat_list"])-1)]
        
                        else:
                            ind = array_unique(N_adj_mat,potential_product[N_p]["adj_mat_list"])[1]
                            edge_list += [(start_ind,N_p,ind)]
                            geo_opt_list += [(N_p,ind)]

        elif len(common_atom) == 1:

            special_flag,connect_atoms = is_special_atom(common_atom[0],atom_list,BE,E,limited=True)
            
            # identify whether it is possible for the common atom to be a "special" atom
            if special_flag:

                not_common_atom = [atom for atom in atom_list if atom not in common_atom]

                for c_atom in connect_atoms:
                    BE_break = break_bonds(BE,bond_break)
                    N_BE     = form_bonds(BE_break,[not_common_atom,[common_atom[0],c_atom] ] )
                    N_BE[c_atom][c_atom]                         -= 2
                    N_BE[common_atom[0]][common_atom[0]]         += 2
                    N_adj_mat = bond_to_adj(N_BE)
                    N_E,N_Adj_mat,N_hash_list,N_G,N_BE_canon=canon_geo(E,N_adj_mat,G,bond_mat=N_BE)

                    # Calculate the hash value of the atom involved in the bond forming
                    form_hash=sorted([ sorted([hash_list[not_common_atom[0]],hash_list[not_common_atom[1]]]),sorted([hash_list[common_atom[0]],hash_list[c_atom] ]) ])
                    change_hash = break_hash + form_hash

                    # check whether all of the conditions are satisfied
                    ring_index = identify_rings(N_E,N_adj_mat)
                    check_flag = check_ring_condition(ring_index,N_BE)
                    if 1 in truncate:
                        check_flag = (check_flag and check_3members_ring(ring_index))

                    if 2 in truncate:
                        check_flag = (check_flag and check_4members_ring(ring_index))
                
                    if 3 in truncate:
                        check_flag = (check_flag and check_bridge(ring_index))

                    # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
                    check_flag = (check_flag and  array_unique(change_hash,bond_change_list)[0])

                    if array_unique(N_BE_canon,new_BE_list)[0] is True and array_unique(N_hash_list,total_hash_list)[0] is True and check_flag:

                        new_BE_list.append(deepcopy(N_BE_canon))
                        total_hash_list.append(deepcopy(N_hash_list))
                        N_p = len(new_BE_list) - 1
                        potential_product[N_p]                 = {}
                        potential_product[N_p]["E"]            = deepcopy(E)
                        potential_product[N_p]["G_list"]       =[deepcopy(G)]
                        potential_product[N_p]["adj_mat_list"] =[deepcopy(N_adj_mat)]
                        potential_product[N_p]["bond_mat_list"]=[deepcopy(N_BE)]
                        potential_product[N_p]["depth"]        = depth
                        potential_product[N_p]["start_ind"]    = start_ind
                        edge_list += [(start_ind,N_p,0)]
                        geo_opt_list += [(N_p,0)]

                    elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:
                        # This is be same product but allows for different reaction pathways
                        N_p = total_hash_list.index(N_hash_list)
                        if N_p != start_ind and N_p != 0:
                            # check whether this adj_mat exist in adj_mat_list or not
                            if array_unique(N_adj_mat,potential_product[N_p]["adj_mat_list"])[0] is True:
                                potential_product[N_p]["G_list"]       += [deepcopy(G)]
                                potential_product[N_p]["adj_mat_list"] += [deepcopy(N_adj_mat)]
                                potential_product[N_p]["bond_mat_list"]+= [deepcopy(N_BE)]
                                edge_list += [(start_ind,N_p,len(potential_product[N_p]["adj_mat_list"])-1)]
                                geo_opt_list += [(N_p,len(potential_product[N_p]["adj_mat_list"])-1)]
                                
                            else:
                                ind = array_unique(N_adj_mat,potential_product[N_p]["adj_mat_list"])[1]
                                edge_list += [(start_ind,N_p,ind)]
                                geo_opt_list += [(N_p,ind)]

    # Apply force field geometry optimization for each product and adj_mat
    for geo_opt_ind in geo_opt_list:
        product = potential_product[geo_opt_ind[0]]
        ind     = geo_opt_ind[1]

        N_opt_G = opt_geo(deepcopy(G),product["adj_mat_list"][ind],product["E"],ff=ff,step=500)
        potential_product[geo_opt_ind[0]]["G_list"][ind] = N_opt_G

        # generate mol file
        xyz_write('{}/reaction_channel/pp_{}/pp_{}_{}.xyz'.format(outputname,start_ind,geo_opt_ind[0],ind),product["E"],N_opt_G)
        # write one mol file to generate png file
        if ind == 0 and os.path.isfile("{}/mol_files/pp_{}.mol".format(outputname,geo_opt_ind[0])) is False:
            mol_write('{}/mol_files/pp_{}.mol'.format(outputname,geo_opt_ind[0]),product["E"],N_opt_G,product["adj_mat_list"][ind])

    if generate_png:
        for N_p in list(potential_product.keys())[begin_ind:]:
            if os.path.isfile("{}/png_files/pp_{}.png".format(outputname,N_p)) is False:
                # generate png file
                substring = "obabel -imol {}/mol_files/pp_{}.mol -O {}/png_files/pp_{}.png -xO molfile".format(outputname,N_p,outputname,N_p)
                output = subprocess.Popen(substring,shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0] 

    return edge_list

# define type one reaction break 3 form 3 (b3f3, including b2f2) elementary reaction step
def R1_b3f3(start_ind,new_BE_list,total_hash_list,potential_product,outputname,G=[],bond_list=[],truncate=[],generate_png=True,edge_list=[],reactive_list=[],ff='uff'):

    # obatin elements, adj_mat, BE matrix
    E        = potential_product[start_ind]["E"]
    adj_mat  = potential_product[start_ind]["adj_mat_list"][0]
    BE       = potential_product[start_ind]["bond_mat_list"][0]
    depth    = potential_product[start_ind]["depth"] + 1

    # get hash list for the starting compound
    hash_list= return_hash(E,adj_mat)

    # determine the index of next product being enumerated
    begin_ind= len(new_BE_list)

    # get freezed atoms index (like the benzene ring), those bonds involved will not breake  
    freeze   = return_freeze(E,adj_mat,BE)

    # If no geometry is given, the start geometry directly take the first one
    if G==[]:
        G    = potential_product[start_ind]["G_list"][0]

    # Generate geo-opt needed list
    geo_opt_list = []
    
    # Generate breakable bonds list
    if bond_list == []:
        for i in reactive_list:
            for j in reactive_list[i+1:]:
                bond_list += [[i,j]]*int(BE[i][j])

    # generate all possible C_N^2 combinations
    comb2 = combinations(bond_list, 2)
    comb3 = combinations(bond_list, 3)

    # initialzie some lists
    total_break = []
    bond_change_list = []

    # loop over all b2 bond changes
    for bond_break in comb2:

        if bond_break not in total_break:
            total_break += [bond_break]
        else:
            continue

        # can't break up double-bond in one step
        if bond_break[0] == bond_break[1]: 
            continue

        # can't break up freezed bonds
        if (bond_break[0][0] in freeze and bond_break[0][1] in freeze) or (bond_break[1][0] in freeze and bond_break[1][1] in freeze):
            continue
        
        atom_list = []
        for bb in bond_break: atom_list += bb
        
        # Calculate the hash value of the atom involved in the bond breaking
        break_hash=sorted([sorted([hash_list[b[0]],hash_list[b[1]]]) for b in bond_break])

        # Determine whether exist common atom
        common_atom = [item for item, count in collections.Counter(atom_list).items() if count > 1]        

        # Determine possible reaction based on number of common atom
        if len(common_atom) == 0:

            # if there is no common atom, two possible new arrangements
            BE_break = break_bonds(BE,bond_break)
            bonds_form = generate_bond_form(bond_break,atom_list)

            for bond_form in bonds_form:

                N_BE = form_bonds(BE_break,bond_form)
                N_adj_mat = bond_to_adj(N_BE)

                # Apply canonical operation only to detemine whether this is a new molecule or a duolicated one
                N_E,N_Adj_mat,N_hash_list,N_BE_canon=canon_geo(E,N_adj_mat,bond_mat=N_BE)

                # Calculate the hash value of the atom involved in the bond forming
                form_hash=sorted([sorted([hash_list[f[0]],hash_list[f[1]]]) for f in bond_form])
                change_hash = break_hash + form_hash

                # check whether all of the conditions are satisfied
                ring_index = identify_rings(N_E,N_adj_mat)
                if reactive_list != []:
                    ring_index =[ring_ind for ring_ind in ring_index if False not in [index in reactive_list for index in ring_ind] ]
                    NR_BE= N_BE[reactive_list,:][:,reactive_list]
                    check_flag = check_bond_condition(NR_BE) and check_ring_condition(ring_index,NR_BE) and check_fused_condition(ring_index,NR_BE) 
                    
                else:
                    check_flag = check_bond_condition(N_BE) and check_ring_condition(ring_index,N_BE) and check_fused_condition(ring_index,N_BE) 

                if 1 in truncate:
                    check_flag = (check_flag and check_3members_ring(ring_index))
                
                if 2 in truncate:
                    check_flag = (check_flag and check_4members_ring(ring_index))

                if 3 in truncate:
                    check_flag = (check_flag and check_bridge(ring_index))

                # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
                check_flag = (check_flag and  array_unique(change_hash,bond_change_list)[0])

                if array_unique(N_BE_canon,new_BE_list)[0] is True and array_unique(N_hash_list,total_hash_list)[0] is True and check_flag:

                    # A new product is found !
                    new_BE_list.append(deepcopy(N_BE_canon))
                    total_hash_list.append(deepcopy(N_hash_list))
                    bond_change_list.append(change_hash)

                    N_p = len(new_BE_list) - 1
                    potential_product[N_p]                 = {}
                    potential_product[N_p]["E"]            = deepcopy(E)
                    potential_product[N_p]["G_list"]       =[deepcopy(G)]
                    potential_product[N_p]["adj_mat_list"] =[deepcopy(N_adj_mat)]
                    potential_product[N_p]["bond_mat_list"]=[deepcopy(N_BE)]
                    potential_product[N_p]["depth"]        = depth
                    potential_product[N_p]["start_ind"]    = start_ind
                    edge_list    += [(start_ind,N_p,0)]
                    geo_opt_list += [(N_p,0)]

                elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:

                    # append change_hash into the change list
                    bond_change_list.append(change_hash)

                    # This is be same product but allows for different reaction pathways
                    N_p = total_hash_list.index(N_hash_list)

                    # if N_p = start, self reaction, ignore it
                    if N_p != start_ind and N_p != 0:
                        # check whether this adj_mat exist in adj_mat_list or not
                        if array_unique(N_adj_mat,potential_product[N_p]["adj_mat_list"])[0]:
                            potential_product[N_p]["G_list"]       += [deepcopy(G)]
                            potential_product[N_p]["adj_mat_list"] += [deepcopy(N_adj_mat)] 
                            potential_product[N_p]["bond_mat_list"]+= [deepcopy(N_BE)]
                            edge_list += [(start_ind,N_p,len(potential_product[N_p]["adj_mat_list"])-1)]
                            geo_opt_list += [(N_p,len(potential_product[N_p]["adj_mat_list"])-1)]
        
                        else:
                            ind = array_unique(N_adj_mat,potential_product[N_p]["adj_mat_list"])[1]
                            edge_list += [(start_ind,N_p,ind)]
                            geo_opt_list += [(N_p,ind)]

        elif len(common_atom) == 1:

            special_flag,connect_atoms = is_special_atom(common_atom[0],atom_list,BE,E,limited=True)
            
            # identify whether it is possible for the common atom to be a "special" atom
            if special_flag:

                not_common_atom = [atom for atom in atom_list if atom not in common_atom]

                for c_atom in connect_atoms:
                    BE_break = break_bonds(BE,bond_break)
                    N_BE     = form_bonds(BE_break,[not_common_atom,[common_atom[0],c_atom] ] )
                    N_BE[c_atom][c_atom]                         -= 2
                    N_BE[common_atom[0]][common_atom[0]]         += 2
                    N_adj_mat = bond_to_adj(N_BE)
                    N_E,N_Adj_mat,N_hash_list,N_G,N_BE_canon=canon_geo(E,N_adj_mat,G,bond_mat=N_BE)

                    # Calculate the hash value of the atom involved in the bond forming
                    form_hash=sorted([ sorted([hash_list[not_common_atom[0]],hash_list[not_common_atom[1]]]),sorted([hash_list[common_atom[0]],hash_list[c_atom] ]) ])
                    change_hash = break_hash + form_hash

                    # check whether all of the conditions are satisfied
                    ring_index = identify_rings(N_E,N_adj_mat)
                    check_flag = check_ring_condition(ring_index,N_BE)
                    if 1 in truncate:
                        check_flag = (check_flag and check_3members_ring(ring_index))

                    if 2 in truncate:
                        check_flag = (check_flag and check_4members_ring(ring_index))
                
                    if 3 in truncate:
                        check_flag = (check_flag and check_bridge(ring_index))

                    # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
                    check_flag = (check_flag and  array_unique(change_hash,bond_change_list)[0])

                    if array_unique(N_BE_canon,new_BE_list)[0] is True and array_unique(N_hash_list,total_hash_list)[0] is True and check_flag:

                        new_BE_list.append(deepcopy(N_BE_canon))
                        total_hash_list.append(deepcopy(N_hash_list))
                        N_p = len(new_BE_list) - 1
                        potential_product[N_p]                 = {}
                        potential_product[N_p]["E"]            = deepcopy(E)
                        potential_product[N_p]["G_list"]       =[deepcopy(G)]
                        potential_product[N_p]["adj_mat_list"] =[deepcopy(N_adj_mat)]
                        potential_product[N_p]["bond_mat_list"]=[deepcopy(N_BE)]
                        potential_product[N_p]["depth"]        = depth
                        potential_product[N_p]["start_ind"]    = start_ind
                        edge_list += [(start_ind,N_p,0)]
                        geo_opt_list += [(N_p,0)]

                    elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:
                        # This is be same product but allows for different reaction pathways
                        N_p = total_hash_list.index(N_hash_list)
                        if N_p != start_ind and N_p != 0:
                            # check whether this adj_mat exist in adj_mat_list or not
                            if array_unique(N_adj_mat,potential_product[N_p]["adj_mat_list"])[0] is True:
                                potential_product[N_p]["G_list"]       += [deepcopy(G)]
                                potential_product[N_p]["adj_mat_list"] += [deepcopy(N_adj_mat)]
                                potential_product[N_p]["bond_mat_list"]+= [deepcopy(N_BE)]
                                edge_list += [(start_ind,N_p,len(potential_product[N_p]["adj_mat_list"])-1)]
                                geo_opt_list += [(N_p,len(potential_product[N_p]["adj_mat_list"])-1)]
                                
                            else:
                                ind = array_unique(N_adj_mat,potential_product[N_p]["adj_mat_list"])[1]
                                edge_list += [(start_ind,N_p,ind)]
                                geo_opt_list += [(N_p,ind)]


    # loop over all b3 bond changes
    for bond_break in comb3:

        if bond_break not in total_break: total_break += [bond_break]
        else: continue

        # can't break up double-bond in one step
        if bond_break[0] == bond_break[1] or bond_break[0] == bond_break[2] or bond_break[1] == bond_break[2]: continue

        # can't break up freezed bonds
        if (bond_break[0][0] in freeze and bond_break[0][1] in freeze) or (bond_break[1][0] in freeze and bond_break[1][1] in freeze) or (bond_break[2][0] in freeze and bond_break[2][1] in freeze): continue
        
        atom_list = []
        for bb in bond_break: atom_list += bb
        
        # Calculate the hash value of the atom involved in the bond breaking
        break_hash=sorted([sorted([hash_list[b[0]],hash_list[b[1]]]) for b in bond_break])

        # Determine whether exist common atom
        common_atom = [item for item, count in collections.Counter(atom_list).items() if count > 1]        

        # Determine possible reaction based on number of common atom
        if len(common_atom) != 0: continue

        # if there is no common atom, two possible new arrangements
        BE_break = break_bonds(BE,bond_break)
        bonds_form = generate_bond_form(bond_break,atom_list)

        for bond_form in bonds_form:

            N_BE = form_bonds(BE_break,bond_form)
            N_adj_mat = bond_to_adj(N_BE)

            # Apply canonical operation only to detemine whether this is a new molecule or a duolicated one
            N_E,N_Adj_mat,N_hash_list,N_BE_canon=canon_geo(E,N_adj_mat,bond_mat=N_BE)

            # Calculate the hash value of the atom involved in the bond forming
            form_hash=sorted([sorted([hash_list[f[0]],hash_list[f[1]]]) for f in bond_form])
            change_hash = break_hash + form_hash

            # check whether all of the conditions are satisfied
            ring_index = identify_rings(N_E,N_adj_mat)
            if reactive_list != []:
                ring_index =[ring_ind for ring_ind in ring_index if False not in [index in reactive_list for index in ring_ind] ]
                NR_BE= N_BE[reactive_list,:][:,reactive_list]
                check_flag = check_bond_condition(NR_BE) and check_ring_condition(ring_index,NR_BE) and check_fused_condition(ring_index,NR_BE) 
                    
            else:
                check_flag = check_bond_condition(N_BE) and check_ring_condition(ring_index,N_BE) and check_fused_condition(ring_index,N_BE) 

            if 1 in truncate:
                check_flag = (check_flag and check_3members_ring(ring_index))
                
            if 2 in truncate:
                check_flag = (check_flag and check_4members_ring(ring_index))

            if 3 in truncate:
                check_flag = (check_flag and check_bridge(ring_index))

            # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
            check_flag = (check_flag and  array_unique(change_hash,bond_change_list)[0])

            if array_unique(N_BE_canon,new_BE_list)[0] is True and array_unique(N_hash_list,total_hash_list)[0] is True and check_flag:

                # A new product is found !
                new_BE_list.append(deepcopy(N_BE_canon))
                total_hash_list.append(deepcopy(N_hash_list))
                bond_change_list.append(change_hash)

                N_p = len(new_BE_list) - 1
                potential_product[N_p]                 = {}
                potential_product[N_p]["E"]            = deepcopy(E)
                potential_product[N_p]["G_list"]       =[deepcopy(G)]
                potential_product[N_p]["adj_mat_list"] =[deepcopy(N_adj_mat)]
                potential_product[N_p]["bond_mat_list"]=[deepcopy(N_BE)]
                potential_product[N_p]["depth"]        = depth
                potential_product[N_p]["start_ind"]    = start_ind
                edge_list    += [(start_ind,N_p,0)]
                geo_opt_list += [(N_p,0)]

            elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:

                # append change_hash into the change list
                bond_change_list.append(change_hash)

                # This is be same product but allows for different reaction pathways
                N_p = total_hash_list.index(N_hash_list)

                # if N_p = start, self reaction, ignore it
                if N_p != start_ind and N_p != 0:
                    # check whether this adj_mat exist in adj_mat_list or not
                    if array_unique(N_adj_mat,potential_product[N_p]["adj_mat_list"])[0]:
                        potential_product[N_p]["G_list"]       += [deepcopy(G)]
                        potential_product[N_p]["adj_mat_list"] += [deepcopy(N_adj_mat)] 
                        potential_product[N_p]["bond_mat_list"]+= [deepcopy(N_BE)]
                        edge_list += [(start_ind,N_p,len(potential_product[N_p]["adj_mat_list"])-1)]
                        geo_opt_list += [(N_p,len(potential_product[N_p]["adj_mat_list"])-1)]
        
                    else:
                        ind = array_unique(N_adj_mat,potential_product[N_p]["adj_mat_list"])[1]
                        edge_list += [(start_ind,N_p,ind)]
                        geo_opt_list += [(N_p,ind)]

    # Apply force field geometry optimization for each product and adj_mat
    for geo_opt_ind in geo_opt_list:
        product = potential_product[geo_opt_ind[0]]
        ind     = geo_opt_ind[1]

        N_opt_G = opt_geo(deepcopy(G),product["adj_mat_list"][ind],product["E"],ff=ff,step=500)
        potential_product[geo_opt_ind[0]]["G_list"][ind] = N_opt_G

        # generate mol file
        xyz_write('{}/reaction_channel/pp_{}/pp_{}_{}.xyz'.format(outputname,start_ind,geo_opt_ind[0],ind),product["E"],N_opt_G)
        # write one mol file to generate png file
        if ind == 0 and os.path.isfile("{}/mol_files/pp_{}.mol".format(outputname,geo_opt_ind[0])) is False:
            mol_write('{}/mol_files/pp_{}.mol'.format(outputname,geo_opt_ind[0]),product["E"],N_opt_G,product["adj_mat_list"][ind])

    if generate_png:
        for N_p in list(potential_product.keys())[begin_ind:]:
            if os.path.isfile("{}/png_files/pp_{}.png".format(outputname,N_p)) is False:
                # generate png file
                substring = "obabel -imol {}/mol_files/pp_{}.mol -O {}/png_files/pp_{}.png -xO molfile".format(outputname,N_p,outputname,N_p)
                output = subprocess.Popen(substring,shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0] 

    return edge_list

