###### This file contains different ERS reaction types ######
###### Currently only R1_B2f2 is valid #######
import sys

# all function in taffi
sys.path.append('../utilities')
from taffi_functions import *
from utility import return_smi,return_inchikey

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
    if type(bond_break[0])==list:
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
    if type(bond_form[0])==list: # form 2 case
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

# Function to generate all possible bond rearrangement (new bond forming)
# lone refers to atom index of lone electron
# zw_ind should be [a,b] where a is cation index and b is anion index 
def generate_bond_form(bond_break,atom_list,lone=[],zw_ind=[],number_of_formation=2):

    bond_form = []
    
    # check lone and zw_ind
    if len(lone) > 1:
        print("Error! Only support close-shell or uni-radical cases...")
        quit()
        
    if len(lone) == 1 and len(zw_ind) > 0:
        print("Error! Can't deal with radical and zwritterionic mixed cases...")
        quit()

    #if len(bond_break) != 2 and len(bond_break) != 3:
    #    print("Expect two/three bonds to be broken due to ERS definition, exit...")
    #    quit()

    # Generate possible bonds form for different situations
    # neutral, no radical case
    if len(lone) == 0 and sorted(zw_ind) not in [sorted(b) for b in bond_break]:

        if len(bond_break) == 2 and number_of_formation==2: # for b2f2 case
            bond_form += [ [ [bond_break[0][0],bond_break[1][0]],[bond_break[0][1],bond_break[1][1]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[1][1]],[bond_break[0][1],bond_break[1][0]] ] ]

        elif len(bond_break)==2 and number_of_formation==1: # For b2f1 case
            bond_form+=[[bond_break[0][0],bond_break[1][0]]]
            bond_form+=[[bond_break[0][1],bond_break[1][1]]]
            bond_form+=[[bond_break[0][0],bond_break[1][1]]]
            bond_form+=[[bond_break[0][1],bond_break[1][0]]]
        else:
            bond_form += [ [ [bond_break[0][0],bond_break[1][0]],[bond_break[0][1],bond_break[2][0]],[bond_break[1][1],bond_break[2][1]] ] ] 
            bond_form += [ [ [bond_break[0][0],bond_break[1][0]],[bond_break[0][1],bond_break[2][1]],[bond_break[1][1],bond_break[2][0]] ] ] 
            bond_form += [ [ [bond_break[0][0],bond_break[1][1]],[bond_break[0][1],bond_break[2][0]],[bond_break[1][0],bond_break[2][1]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[1][1]],[bond_break[0][1],bond_break[2][1]],[bond_break[1][0],bond_break[2][0]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[2][0]],[bond_break[0][1],bond_break[1][0]],[bond_break[1][1],bond_break[2][1]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[2][0]],[bond_break[0][1],bond_break[1][1]],[bond_break[1][0],bond_break[2][1]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[2][1]],[bond_break[0][1],bond_break[1][0]],[bond_break[1][1],bond_break[2][0]] ] ]
            bond_form += [ [ [bond_break[0][0],bond_break[2][1]],[bond_break[0][1],bond_break[1][1]],[bond_break[1][0],bond_break[2][0]] ] ]

    elif len(lone) == 1:
        if len(bond_break)==1:
            # For b1f1 for uni-radical system
            if lone[0] not in atom_list:
                bond_form+=[[bond_break[0][0], lone[0]]]
                bond_form+=[[bond_break[0][1], lone[0]]]
        elif len(bond_break)==2 and lone[0] not in atom_list and number_of_formation==2:
        
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

    else:
        
        bond_form += [ [ [bond_break[0][0],bond_break[1][0]],[bond_break[0][1],bond_break[1][1]] ] ]
        bond_form += [ [ [bond_break[0][0],bond_break[1][1]],[bond_break[0][1],bond_break[1][0]] ] ]

        # one more bond rearrangement, the atom with anion forms two bonds
        # first identify which bond is the "special" one
        if zw_ind[0] in bond_break[0]:
            bond_form += [ [ [zw_ind[1],bond_break[1][0] ],[zw_ind[1],bond_break[1][1]] ] ]
        else:
            bond_form += [ [ [zw_ind[1],bond_break[0][0] ],[zw_ind[1],bond_break[0][1]] ] ]

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

# Determine whether a fused ring exists
def check_fused_condition(ring_index):

    if len(ring_index) >= 2:
        combs = combinations(range(len(ring_index)), 2)
        for comb in combs:
            ring_atoms = ring_index[comb[0]]+ring_index[comb[1]]
            N_common = len(ring_atoms) - len(set(ring_atoms))
            if N_common == 1 and min(len(ring_index[comb[0]]),len(ring_index[comb[1]])) == 3: return False
            if N_common >= 2: return False

    return True
    
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

# Function to split all compounds
def split_compounds(E,G,adj_mat):

    # loop over all of the products and get inchikey for each component
    frag_inchi = []
    
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
        frag_E = [E[ind] for ind in group]
        frag_G = np.zeros([N_atom,3])
        for count_i,i in enumerate(group):
            frag_G[count_i,:] = G[i,:]

        # determine the adjacency matrix and write a mol file 
        frag_adj_mat= adj_mat[group,:][:,group]
        inchikey = return_inchikey(frag_E,frag_G,frag_adj_mat)
        smiles   = return_smi(frag_E,frag_G,frag_adj_mat)
        inchi_list+= [inchikey]
        smile_list += [smiles]

    return smile_list,inchi_list,groups

# define type one reaction break 2 form 2 (b2f2) elementary reaction step
def R1_b2f2(reactant,truncate=[],ff='uff',phase=1):

    # create a dictionary for products and a dictionary for separated uni-products
    possible_product = {}
    separated_product= {}

    # obatin elements, adj_mat, BE matrix
    E        = reactant["E"]
    G        = reactant["G"]
    BE       = reactant["BE_mat"]
    adj_mat  = reactant["adj_mat"]
    bond_list= reactant["bond_list"]
    hash_list= reactant["hash_list"]
    radical_part=find_radical(E, BE)
    # Create a list of potential products, BE_list and hash_list are used to identify whether a new prodcut is found or not
    BE_list = []
    total_hash_list = []
    BE_list.append(deepcopy(BE))     
    total_hash_list.append(deepcopy(hash_list))
    
    # get freezed atoms index (like the benzene ring), those bonds involved will not breake 
    freeze   = return_freeze(E,adj_mat,BE)

    # return number of reactant(s) in this structure. If there are multi-reactants, also return the index
    smile_list,inchi_list,groups = split_compounds(E,G,adj_mat)

    # determine which phase will this enumeration be and whether the input reactant (uni/bi) matches with the phase
    if phase == 1 and len(groups) == 1:

        print("Phase 1 enumeration: unimolecular transformation...")

        # generate all possible C_N^2 combinations
        comb = [bc for bc in combinations(bond_list, 2)]

    # if phase is 2 and number of reactant is 2
    elif phase == 2 and len(groups) == 2:

        # generate all possible C_N^2 combinations, only keep it when break two bonds in two reactant
        comb = [bc for bc in combinations(bond_list, 2) if [bc[0][0] in group for group in groups].index(True) != [bc[1][0] in group for group in groups].index(True) ]

    else:
        print("Only two phases available, make sure phase and number of reactant matches, quit...")
        quit()

    # initialzie some lists
    total_break = []
    bond_change_list = []

    # loop over all bond changes
    for bond_break in comb:

        # can't break up double-bond in one step
        if bond_break[0] == bond_break[1]: 
            continue

        # can't break up freezed bonds
        if (bond_break[0][0] in freeze and bond_break[0][1] in freeze) or (bond_break[1][0] in freeze and bond_break[1][1] in freeze):
            continue

        # if the same bond is aleardy broken, skip this bond
        if bond_break not in total_break:
            total_break += [bond_break]

        else:
            continue
        
        atom_list = []
        for bb in bond_break: 
            atom_list += bb
        
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

                N_BE = form_bonds(BE_break,bond_form)
                N_adj_mat = bond_to_adj(N_BE)

                # Apply canonical operation only to detemine whether this is a new molecule or a duolicated one
                _,_,N_hash_list,N_BE_canon=canon_geo(E,N_adj_mat,bond_mat=N_BE)

                # Calculate the hash value of the atom involved in the bond forming
                form_hash  = sorted([sorted([hash_list[f[0]],hash_list[f[1]]]) for f in bond_form])
                change_hash= break_hash + form_hash

                # check whether all of the conditions are satisfied
                ring_index = identify_rings(E,N_adj_mat)
                check_flag = check_bond_condition(N_BE) and check_ring_condition(ring_index,N_BE) and check_fused_condition(ring_index) 

                if 1 in truncate: check_flag = (check_flag and check_3members_ring(ring_index))
                if 2 in truncate: check_flag = (check_flag and check_4members_ring(ring_index))
                if 3 in truncate: check_flag = (check_flag and check_bridge(ring_index))

                # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
                check_flag = (check_flag and  array_unique(change_hash,bond_change_list)[0])

                if array_unique(N_BE_canon,BE_list)[0] and array_unique(N_hash_list,total_hash_list)[0] and check_flag:

                    # A new product is found!
                    BE_list.append(deepcopy(N_BE_canon))
                    total_hash_list.append(deepcopy(N_hash_list))
                    bond_change_list.append(change_hash)
                    
                    # The index of new product
                    N_p = len(BE_list) - 1
                    possible_product[N_p] = {}
                    
                    # Apply force field geometry optimization for each product and adj_mat
                    N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                    Psmile_list,Pinchi_list,Pgroups = split_compounds(E,N_G,N_adj_mat)
                    name = '.'.join(Psmile_list)

                    # write properties into product dictionary
                    possible_product[N_p]["E"]        = E
                    possible_product[N_p]["name"]     = name
                    possible_product[N_p]["hash_list"]= N_hash_list
                    possible_product[N_p]["G_list"]   =[deepcopy(N_G)] 
                    possible_product[N_p]["adj_list"] =[deepcopy(N_adj_mat)]
                    possible_product[N_p]["BE_list"]  =[deepcopy(N_BE)]
                        
                    # write info into separated product dictionary
                    for ind,group in enumerate(Pgroups):
                        inchi_ind = Pinchi_list[ind].split('-')[0]
                        # add into dictionary if this compound not shown in dict
                        if inchi_ind not in separated_product.keys():
                            separated_product[inchi_ind] = {}
                            separated_product[inchi_ind]["E"] = [E[ind] for ind in group]
                            separated_product[inchi_ind]["G"] = N_G[group,:]
                            separated_product[inchi_ind]["adj_mat"]= N_adj_mat[group,:][:,group]
                            separated_product[inchi_ind]["smiles"] = return_smi(separated_product[inchi_ind]["E"],separated_product[inchi_ind]["G"],separated_product[inchi_ind]["adj_mat"])
                            separated_product[inchi_ind]["source"] = [deepcopy(N_p)]
                                
                        else:
                            # if this compound already exists in dict, add another source of it
                            if N_p not in separated_product[inchi_ind]["source"]:
                                separated_product[inchi_ind]["source"] += [deepcopy(N_p)]

                elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:
                        
                    # append change_hash into the change list
                    bond_change_list.append(change_hash)
                    
                    # This is be same product but allows for different reaction pathways
                    N_p = total_hash_list.index(N_hash_list)
                    
                    # exclude conformational change
                    if N_p == 0:
                        continue

                    # check whether this adj_mat exist in adj_mat_list or not
                    if array_unique(N_adj_mat,possible_product[N_p]["adj_list"])[0]:

                        # Apply force field geometry optimization for each product and adj_mat
                        N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                        possible_product[N_p]["G_list"]   += [deepcopy(N_G)]
                        possible_product[N_p]["adj_list"] += [deepcopy(N_adj_mat)] 
                        possible_product[N_p]["BE_list"]  += [deepcopy(N_BE)]
        
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
                    form_hash  = sorted([ sorted([hash_list[not_common_atom[0]],hash_list[not_common_atom[1]]]),sorted([hash_list[common_atom[0]],hash_list[c_atom] ]) ])
                    change_hash= break_hash + form_hash

                    # check whether all of the conditions are satisfied
                    ring_index = identify_rings(N_E,N_adj_mat)
                    check_flag = check_ring_condition(ring_index,N_BE)

                    if 1 in truncate: check_flag = (check_flag and check_3members_ring(ring_index))
                    if 2 in truncate: check_flag = (check_flag and check_4members_ring(ring_index))
                    if 3 in truncate: check_flag = (check_flag and check_bridge(ring_index))

                    # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
                    check_flag = (check_flag and array_unique(change_hash,bond_change_list)[0])

                    if array_unique(N_BE_canon,BE_list)[0] is True and array_unique(N_hash_list,total_hash_list)[0] is True and check_flag:
                        
                        # A new product is found!
                        BE_list.append(deepcopy(N_BE_canon))
                        total_hash_list.append(deepcopy(N_hash_list))
                        bond_change_list.append(change_hash)
                            
                        # The index of new product
                        N_p = len(BE_list) - 1
                        possible_product[N_p] = {}
                            
                        # Apply force field geometry optimization for each product and adj_mat
                        N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                        if len(N_G) != len(G):
                            N_G = N_G[:len(G)]

                        Psmile_list,Pinchi_list,Pgroups = split_compounds(E,N_G,N_adj_mat)
                        name = '.'.join(Psmile_list)
                            
                        # write properties into product dictionary
                        possible_product[N_p]["E"]        = E
                        possible_product[N_p]["name"]     = name
                        possible_product[N_p]["hash_list"]= N_hash_list
                        possible_product[N_p]["G_list"]   =[deepcopy(N_G)] 
                        possible_product[N_p]["adj_list"] =[deepcopy(N_adj_mat)]
                        possible_product[N_p]["BE_list"]  =[deepcopy(N_BE)]
                            
                        # write info into separated product dictionary
                        for ind,group in enumerate(Pgroups):
                            inchi_ind = Pinchi_list[ind].split('-')[0]
                                
                            # add into dictionary if this compound not shown in dict
                            if inchi_ind not in separated_product.keys():
                                separated_product[inchi_ind] = {}
                                separated_product[inchi_ind]["E"] = [E[ind] for ind in group]
                                separated_product[inchi_ind]["G"] = N_G[group,:]
                                separated_product[inchi_ind]["adj_mat"]= N_adj_mat[group,:][:,group]
                                separated_product[inchi_ind]["smiles"] = return_smi(separated_product[inchi_ind]["E"],separated_product[inchi_ind]["G"],separated_product[inchi_ind]["adj_mat"])
                                separated_product[inchi_ind]["source"] = [deepcopy(N_p)]
                                
                            else:
                                # if this compound already exists in dict, add another source of it
                                if N_p not in separated_product[inchi_ind]["source"]:
                                    separated_product[inchi_ind]["source"] += [deepcopy(N_p)]

                    elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:

                        # This is be same product but allows for different reaction pathways
                        N_p = total_hash_list.index(N_hash_list)

                        # exclude conformational change
                        if N_p == 0:
                            continue

                        # check whether this adj_mat exist in adj_mat_list or not
                        if array_unique(N_adj_mat,possible_product[N_p]["adj_list"])[0]:
                        
                            # Apply force field geometry optimization for each product and adj_mat
                            N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                            if len(N_G) != len(G):
                                N_G = N_G[:len(G)]
                            possible_product[N_p]["G_list"]   += [deepcopy(N_G)]
                            possible_product[N_p]["adj_list"] += [deepcopy(N_adj_mat)] 
                            possible_product[N_p]["BE_list"]  += [deepcopy(N_BE)]
                                
    return possible_product,separated_product

# define type one reaction break 3 form 3 (b3f3, including b2f2) elementary reaction step
# when include_b2f2 is true, enuemrate both b2f2 and b3f3 reactions
# when limit_b3f3 is true, doing general b2f2, where b3f3 must involve a single/double bond transform in both side  
def R1_b3f3(reactant,truncate=[],ff='uff',phase=1,include_b2f2=True,limit_b3f3=False):

    # create a dictionary for products and a dictionary for separated uni-products
    possible_product = {}
    separated_product= {}

    # obatin elements, adj_mat, BE matrix
    E        = reactant["E"]
    G        = reactant["G"]
    BE       = reactant["BE_mat"]
    adj_mat  = reactant["adj_mat"]
    bond_list= reactant["bond_list"]
    hash_list= reactant["hash_list"]

    # Create a list of potential products, BE_list and hash_list are used to identify whether a new prodcut is found or not
    BE_list = []
    total_hash_list = []
    BE_list.append(deepcopy(BE))     
    total_hash_list.append(deepcopy(hash_list))

    # get freezed atoms index (like the benzene ring), those bonds involved will not breake  
    freeze   = return_freeze(E,adj_mat,BE)

    # return number of reactant(s) in this structure. If there are multi-reactants, also return the index
    smile_list,inchi_list,groups = split_compounds(E,G,adj_mat)

    # determine which phase will this enumeration be and whether the input reactant (uni/bi) matches with the phase
    if phase == 1 and len(groups) == 1:

        print("Phase 1 enumeration: unimolecular transformation...")

        # generate all possible C_N^2 combinations
        comb2 = [bc for bc in combinations(bond_list, 2)]
        comb3 = [bc for bc in combinations(bond_list, 3)]

    # if phase is 2 and number of reactant is 2
    elif phase == 2 and len(groups) == 2:

        # generate all possible C_N^2 and C_N^3 combinations, only keep it when break bonds in two reactant
        comb2 = [bc for bc in combinations(bond_list, 2) if [bc[0][0] in group for group in groups].index(True) != [bc[1][0] in group for group in groups].index(True) ]
        comb3 = [bc for bc in combinations(bond_list, 3) if ([bc[0][0] in group for group in groups].index(True) != [bc[1][0] in group for group in groups].index(True) or \
                                                             [bc[0][0] in group for group in groups].index(True) != [bc[2][0] in group for group in groups].index(True))]

    else:
        print("Only two phases available, make sure phase and number of reactant matches, quit...")
        quit()

    # remove b2f2 if is not needed
    if not include_b2f2: comb2 = []

    # further select comb3 based on limit
    if limit_b3f3: comb3 = [i for i in comb3 if max([bond_list.count(bc) for bc in i]) > 1 and len(set([tuple(j) for j in i])) == 3 ]

    # initialzie some lists
    total_break = []
    bond_change_list = []

    # loop over all b2 bond changes
    for bond_break in comb2:

        # can't break up double-bond in one step
        if bond_break[0] == bond_break[1]: 
            continue

        # can't break up freezed bonds
        if (bond_break[0][0] in freeze and bond_break[0][1] in freeze) or (bond_break[1][0] in freeze and bond_break[1][1] in freeze):
            continue

        # if the same bond is aleardy broken, skip this bond
        if bond_break not in total_break: total_break += [bond_break]
        else: continue
        
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
            bonds_form = generate_bond_form(bond_break, atom_list, number_of_formation=2)

            for bond_form in bonds_form:

                N_BE = form_bonds(BE_break,bond_form)
                N_adj_mat = bond_to_adj(N_BE)

                # Apply canonical operation only to detemine whether this is a new molecule or a duolicated one
                _,_,N_hash_list,N_BE_canon=canon_geo(E,N_adj_mat,bond_mat=N_BE)

                # Calculate the hash value of the atom involved in the bond forming
                form_hash  = sorted([sorted([hash_list[f[0]],hash_list[f[1]]]) for f in bond_form])
                change_hash= break_hash + form_hash

                # check whether all of the conditions are satisfied
                ring_index = identify_rings(E,N_adj_mat)
                check_flag = check_bond_condition(N_BE) and check_ring_condition(ring_index,N_BE) and check_fused_condition(ring_index) 

                if 1 in truncate: check_flag = (check_flag and check_3members_ring(ring_index))
                if 2 in truncate: check_flag = (check_flag and check_4members_ring(ring_index))
                if 3 in truncate: check_flag = (check_flag and check_bridge(ring_index))

                # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
                check_flag = (check_flag and  array_unique(change_hash,bond_change_list)[0])

                if array_unique(N_BE_canon,BE_list)[0] and array_unique(N_hash_list,total_hash_list)[0] and check_flag:

                    # A new product is found!
                    BE_list.append(deepcopy(N_BE_canon))
                    total_hash_list.append(deepcopy(N_hash_list))
                    bond_change_list.append(change_hash)
                    
                    # The index of new product
                    N_p = len(BE_list) - 1
                    possible_product[N_p] = {}
                    
                    # Apply force field geometry optimization for each product and adj_mat
                    N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                    Psmile_list,Pinchi_list,Pgroups = split_compounds(E,N_G,N_adj_mat)
                    name = '.'.join(Psmile_list)

                    # write properties into product dictionary
                    possible_product[N_p]["E"]        = E
                    possible_product[N_p]["name"]     = name
                    possible_product[N_p]["hash_list"]= N_hash_list
                    possible_product[N_p]["G_list"]   =[deepcopy(N_G)] 
                    possible_product[N_p]["adj_list"] =[deepcopy(N_adj_mat)]
                    possible_product[N_p]["BE_list"]  =[deepcopy(N_BE)]
                        
                    # write info into separated product dictionary
                    for ind,group in enumerate(Pgroups):
                        inchi_ind = Pinchi_list[ind].split('-')[0]
                        # add into dictionary if this compound not shown in dict
                        if inchi_ind not in separated_product.keys():
                            separated_product[inchi_ind] = {}
                            separated_product[inchi_ind]["E"] = [E[ind] for ind in group]
                            separated_product[inchi_ind]["G"] = N_G[group,:]
                            separated_product[inchi_ind]["adj_mat"]= N_adj_mat[group,:][:,group]
                            separated_product[inchi_ind]["smiles"] = return_smi(separated_product[inchi_ind]["E"],separated_product[inchi_ind]["G"],separated_product[inchi_ind]["adj_mat"])
                            separated_product[inchi_ind]["source"] = [deepcopy(N_p)]
                                
                        else:
                            # if this compound already exists in dict, add another source of it
                            if N_p not in separated_product[inchi_ind]["source"]:
                                separated_product[inchi_ind]["source"] += [deepcopy(N_p)]

                elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:
                        
                    # append change_hash into the change list
                    bond_change_list.append(change_hash)
                    
                    # This is be same product but allows for different reaction pathways
                    N_p = total_hash_list.index(N_hash_list)
                    
                    # exclude conformational change
                    if N_p == 0:
                        continue

                    # check whether this adj_mat exist in adj_mat_list or not
                    if array_unique(N_adj_mat,possible_product[N_p]["adj_list"])[0]:

                        # Apply force field geometry optimization for each product and adj_mat
                        N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                        possible_product[N_p]["G_list"]   += [deepcopy(N_G)]
                        possible_product[N_p]["adj_list"] += [deepcopy(N_adj_mat)] 
                        possible_product[N_p]["BE_list"]  += [deepcopy(N_BE)]
        
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
                    form_hash  = sorted([ sorted([hash_list[not_common_atom[0]],hash_list[not_common_atom[1]]]),sorted([hash_list[common_atom[0]],hash_list[c_atom] ]) ])
                    change_hash= break_hash + form_hash

                    # check whether all of the conditions are satisfied
                    ring_index = identify_rings(N_E,N_adj_mat)
                    check_flag = check_ring_condition(ring_index,N_BE)
                    if 1 in truncate: check_flag = (check_flag and check_3members_ring(ring_index))
                    if 2 in truncate: check_flag = (check_flag and check_4members_ring(ring_index))
                    if 3 in truncate: check_flag = (check_flag and check_bridge(ring_index))

                    # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
                    check_flag = (check_flag and array_unique(change_hash,bond_change_list)[0])

                    if array_unique(N_BE_canon,BE_list)[0] is True and array_unique(N_hash_list,total_hash_list)[0] is True and check_flag:
                        
                        # A new product is found!
                        BE_list.append(deepcopy(N_BE_canon))
                        total_hash_list.append(deepcopy(N_hash_list))
                        bond_change_list.append(change_hash)
                            
                        # The index of new product
                        N_p = len(BE_list) - 1
                        possible_product[N_p] = {}
                            
                        # Apply force field geometry optimization for each product and adj_mat
                        N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                        if len(N_G) != len(G):
                            N_G = N_G[:len(G)]

                        Psmile_list,Pinchi_list,Pgroups = split_compounds(E,N_G,N_adj_mat)
                        name = '.'.join(Psmile_list)
                            
                        # write properties into product dictionary
                        possible_product[N_p]["E"]        = E
                        possible_product[N_p]["name"]     = name
                        possible_product[N_p]["hash_list"]= N_hash_list
                        possible_product[N_p]["G_list"]   =[deepcopy(N_G)] 
                        possible_product[N_p]["adj_list"] =[deepcopy(N_adj_mat)]
                        possible_product[N_p]["BE_list"]  =[deepcopy(N_BE)]
                            
                        # write info into separated product dictionary
                        for ind,group in enumerate(Pgroups):
                            inchi_ind = Pinchi_list[ind].split('-')[0]
                                
                            # add into dictionary if this compound not shown in dict
                            if inchi_ind not in separated_product.keys():
                                separated_product[inchi_ind] = {}
                                separated_product[inchi_ind]["E"] = [E[ind] for ind in group]
                                separated_product[inchi_ind]["G"] = N_G[group,:]
                                separated_product[inchi_ind]["adj_mat"]= N_adj_mat[group,:][:,group]
                                separated_product[inchi_ind]["smiles"] = return_smi(separated_product[inchi_ind]["E"],separated_product[inchi_ind]["G"],separated_product[inchi_ind]["adj_mat"])
                                separated_product[inchi_ind]["source"] = [deepcopy(N_p)]
                                
                            else:
                                # if this compound already exists in dict, add another source of it
                                if N_p not in separated_product[inchi_ind]["source"]:
                                    separated_product[inchi_ind]["source"] += [deepcopy(N_p)]

                    elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:

                        # This is be same product but allows for different reaction pathways
                        N_p = total_hash_list.index(N_hash_list)

                        # exclude conformational change
                        if N_p == 0:
                            continue

                        # check whether this adj_mat exist in adj_mat_list or not
                        if array_unique(N_adj_mat,possible_product[N_p]["adj_list"])[0]:
                        
                            # Apply force field geometry optimization for each product and adj_mat
                            N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                            if len(N_G) != len(G):
                                N_G = N_G[:len(G)]
                            possible_product[N_p]["G_list"]   += [deepcopy(N_G)]
                            possible_product[N_p]["adj_list"] += [deepcopy(N_adj_mat)] 
                            possible_product[N_p]["BE_list"]  += [deepcopy(N_BE)]
                                

    # loop over all b3 bond changes
    for bond_break in comb3:

        # can't break up double-bond in one step
        if bond_break[0] == bond_break[1] or bond_break[0] == bond_break[2] or bond_break[1] == bond_break[2]: continue

        # can't break up freezed bonds
        if (bond_break[0][0] in freeze and bond_break[0][1] in freeze) or (bond_break[1][0] in freeze and bond_break[1][1] in freeze) or (bond_break[2][0] in freeze and bond_break[2][1] in freeze): continue

        # if the same bond is aleardy broken, skip this bond
        if bond_break not in total_break: total_break += [bond_break]
        else: continue

        # create atom list
        atom_list = []
        for bb in bond_break: atom_list += bb
        
        # Calculate the hash value of the atom involved in the bond breaking
        break_hash = sorted([sorted([hash_list[b[0]],hash_list[b[1]]]) for b in bond_break])

        # Determine whether exist common atom
        common_atom = [item for item, count in collections.Counter(atom_list).items() if count > 1]        

        # Determine possible reaction based on number of common atom
        BE_break = break_bonds(BE,bond_break)
        if len(common_atom) == 0: 
            bonds_form = generate_bond_form(bond_break,atom_list,number_of_formation=3)
        elif len(common_atom) == 1 and atom_list.count(common_atom[0]) == 2: 
            common_atom = common_atom[0]
            connect_atoms = [ind for ind in atom_list if [ind,common_atom] in bond_break or [common_atom,ind] in bond_break]
            disconnet_atoms = [ind for ind in atom_list if ind not in connect_atoms and ind != common_atom]
            bonds_form = [[connect_atoms,[disconnet_atoms[0],common_atom],[disconnet_atoms[1],common_atom]]]
        else:
            continue

        # further select bond form based on limit
        if limit_b3f3: bonds_form = [bf for bf in bonds_form if max([bond_list.count(b) for b in bf]) > 0]

        # if there is no common atom, 8 possible new arrangements
        for bond_form in bonds_form:

            N_BE = form_bonds(BE_break,bond_form)
            N_adj_mat = bond_to_adj(N_BE)

            # Apply canonical operation only to detemine whether this is a new molecule or a duolicated one
            _,_,N_hash_list,N_BE_canon=canon_geo(E,N_adj_mat,bond_mat=N_BE)

            # Calculate the hash value of the atom involved in the bond forming
            form_hash  = sorted([sorted([hash_list[f[0]],hash_list[f[1]]]) for f in bond_form])
            change_hash= break_hash + form_hash

            # check whether all of the conditions are satisfied
            ring_index = identify_rings(E,N_adj_mat)
            check_flag = check_bond_condition(N_BE) and check_ring_condition(ring_index,N_BE) and check_fused_condition(ring_index) 

            if 1 in truncate: check_flag = (check_flag and check_3members_ring(ring_index))
            if 2 in truncate: check_flag = (check_flag and check_4members_ring(ring_index))
            if 3 in truncate: check_flag = (check_flag and check_bridge(ring_index))

            # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
            check_flag = (check_flag and array_unique(change_hash,bond_change_list)[0])

            if array_unique(N_BE_canon,BE_list)[0] and array_unique(N_hash_list,total_hash_list)[0] and check_flag:

                # A new product is found !
                BE_list.append(deepcopy(N_BE_canon))
                total_hash_list.append(deepcopy(N_hash_list))
                bond_change_list.append(change_hash)

                # The index of new product
                N_p = len(BE_list) - 1
                possible_product[N_p] = {}
                    
                # Apply force field geometry optimization for each product and adj_mat
                N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                Psmile_list,Pinchi_list,Pgroups = split_compounds(E,N_G,N_adj_mat)
                name = '.'.join(Psmile_list)

                # write properties into product dictionary
                possible_product[N_p]["E"]        = E
                possible_product[N_p]["name"]     = name
                possible_product[N_p]["hash_list"]= N_hash_list
                possible_product[N_p]["G_list"]   =[deepcopy(N_G)] 
                possible_product[N_p]["adj_list"] =[deepcopy(N_adj_mat)]
                possible_product[N_p]["BE_list"]  =[deepcopy(N_BE)]
                        
                # write info into separated product dictionary
                for ind,group in enumerate(Pgroups):
                    inchi_ind = Pinchi_list[ind].split('-')[0]
                    # add into dictionary if this compound not shown in dict
                    if inchi_ind not in separated_product.keys():
                        separated_product[inchi_ind] = {}
                        separated_product[inchi_ind]["E"] = [E[ind] for ind in group]
                        separated_product[inchi_ind]["G"] = N_G[group,:]
                        separated_product[inchi_ind]["adj_mat"]= N_adj_mat[group,:][:,group]
                        separated_product[inchi_ind]["smiles"] = return_smi(separated_product[inchi_ind]["E"],separated_product[inchi_ind]["G"],separated_product[inchi_ind]["adj_mat"])
                        separated_product[inchi_ind]["source"] = [deepcopy(N_p)]
                                
                    else:
                        # if this compound already exists in dict, add another source of it
                        if N_p not in separated_product[inchi_ind]["source"]:
                            separated_product[inchi_ind]["source"] += [deepcopy(N_p)]

            elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:
                        
                # append change_hash into the change list
                bond_change_list.append(change_hash)
                    
                # This is be same product but allows for different reaction pathways
                N_p = total_hash_list.index(N_hash_list)
                    
                # exclude conformational change
                if N_p == 0: continue

                # check whether this adj_mat exist in adj_mat_list or not
                if array_unique(N_adj_mat,possible_product[N_p]["adj_list"])[0]:

                    # Apply force field geometry optimization for each product and adj_mat
                    N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                    possible_product[N_p]["G_list"]   += [deepcopy(N_G)]
                    possible_product[N_p]["adj_list"] += [deepcopy(N_adj_mat)] 
                    possible_product[N_p]["BE_list"]  += [deepcopy(N_BE)]

    return possible_product,separated_product

# define type two reaction for zwitter-ionic species, involving b2f2, 
def R2_zwitterionic(reactant,anion_ind,cation_ind,truncate=[],ff='uff',phase=1):

    # create a dictionary for products and a dictionary for seperated uni-products
    possible_product = {}
    separated_product= {}

    # obatin elements, adj_mat, BE matrix
    E        = reactant["E"]
    G        = reactant["G"]
    BE       = reactant["BE_mat"]
    adj_mat  = reactant["adj_mat"]
    bond_list= reactant["bond_list"]
    hash_list= reactant["hash_list"]

    # Create a list of potential products, BE_list and hash_list are used to identify whether a new prodcut is found or not
    BE_list = []
    total_hash_list = []
    BE_list.append(deepcopy(BE))     
    total_hash_list.append(deepcopy(hash_list))
    
    # get freezed atoms index (like the benzene ring), those bonds involved will not breake 
    freeze   = return_freeze(E,adj_mat,BE)

    # return number of reactant(s) in this structure. If there are multi-reactants, also return the index
    smile_list,inchi_list,groups = split_compounds(E,G,adj_mat)

    # determine which phase will this enumeration be and whether the input reactant (uni/bi) matches with the phase
    if phase == 1 and len(groups) == 1:

        print("Phase 1 enumeration: unimolecular transformation...")

        # generate all possible C_N^2 combinations
        comb  = [[i] for i in set(tuple(x) for x in bond_list) if anion_ind not in i and cation_ind in i]
        comb += [bc for bc in combinations(bond_list, 2)]

    # if phase is 2 and number of reactant is 2
    elif phase == 2 and len(groups) == 2:

        # generate all possible C_N^2 combinations, only keep it when break two bonds in two reactant
        comb = [bc for bc in combinations(bond_list, 2) if [bc[0][0] in group for group in groups].index(True) != [bc[1][0] in group for group in groups].index(True) ]

    else:
        print("Only two phases available, make sure phase and number of reactant matches, quit...")
        quit()

    # initialzie some lists
    total_break = []
    bond_change_list = []

    # loop over all bond changes
    for bond_break in comb:

        # calculate bond change hash
        atom_list = []
        for bb in bond_break: 
            atom_list += bb

        # Calculate the hash value of the atom involved in the bond breaking, will be used to avoid duplicated reactions
        break_hash = sorted([sorted([hash_list[b[0]],hash_list[b[1]]]) for b in bond_break])

        if len(bond_break) == 1:

            # For b1f1, one atom transfer from cation atom to anion atom
            BE_break  = break_bonds(BE,bond_break[0])
            bond_form = [(anion_ind,[ind for ind in bond_break[0] if ind != cation_ind][0])]
            N_BE      = form_bonds(BE_break,bond_form[0])
            N_adj_mat = bond_to_adj(N_BE)

            # Apply canonical operation only to detemine whether this is a new molecule or a duolicated one
            _,_,N_hash_list,N_BE_canon=canon_geo(E,N_adj_mat,bond_mat=N_BE)

            # Calculate the hash value of the atom involved in the bond forming
            form_hash  = sorted([sorted([hash_list[f[0]],hash_list[f[1]]]) for f in bond_form])
            change_hash= break_hash + form_hash

            # check whether all of the conditions are satisfied
            ring_index = identify_rings(E,N_adj_mat)
            check_flag = check_bond_condition(N_BE) and check_ring_condition(ring_index,N_BE) and check_fused_condition(ring_index) 

            if 1 in truncate:
                check_flag = (check_flag and check_3members_ring(ring_index))
                
            if 2 in truncate:
                check_flag = (check_flag and check_4members_ring(ring_index))

            if 3 in truncate:
                check_flag = (check_flag and check_bridge(ring_index))

            # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
            check_flag = (check_flag and  array_unique(change_hash,bond_change_list)[0])

            if array_unique(N_BE_canon,BE_list)[0] and array_unique(N_hash_list,total_hash_list)[0] and check_flag:

                # A new product is found!
                BE_list.append(deepcopy(N_BE_canon))
                total_hash_list.append(deepcopy(N_hash_list))
                bond_change_list.append(change_hash)
                    
                # The index of new product
                N_p = len(BE_list) - 1
                possible_product[N_p] = {}
                    
                # Apply force field geometry optimization for each product and adj_mat
                N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                Psmile_list,Pinchi_list,Pgroups = split_compounds(E,N_G,N_adj_mat)
                name = '.'.join(Psmile_list)

                # write properties into product dictionary
                possible_product[N_p]["E"]        = E
                possible_product[N_p]["name"]     = name
                possible_product[N_p]["hash_list"]= N_hash_list
                possible_product[N_p]["G_list"]   =[deepcopy(N_G)] 
                possible_product[N_p]["adj_list"] =[deepcopy(N_adj_mat)]
                possible_product[N_p]["BE_list"]  =[deepcopy(N_BE)]
                    
                # write info into separated product dictionary
                for ind,group in enumerate(Pgroups):
                    inchi_ind = Pinchi_list[ind].split('-')[0]
                
                # add into dictionary if this compound not shown in dict
                if inchi_ind not in separated_product.keys():
                    separated_product[inchi_ind] = {}
                    separated_product[inchi_ind]["E"] = [E[ind] for ind in group]
                    separated_product[inchi_ind]["G"] = N_G[group,:]
                    separated_product[inchi_ind]["adj_mat"]= N_adj_mat[group,:][:,group]
                    separated_product[inchi_ind]["source"] = [deepcopy(N_p)]
                                
                else:
                    # if this compound already exists in dict, add another source of it
                    if N_p not in separated_product[inchi_ind]["source"]:
                        separated_product[inchi_ind]["source"] += [deepcopy(N_p)]

            elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:
                        
                # append change_hash into the change list
                bond_change_list.append(change_hash)
                
                # This is be same product but allows for different reaction pathways
                N_p = total_hash_list.index(N_hash_list)
                    
                # exclude conformational change
                if N_p == 0: continue

                # check whether this adj_mat exist in adj_mat_list or not
                if array_unique(N_adj_mat,possible_product[N_p]["adj_list"])[0]:

                    # Apply force field geometry optimization for each product and adj_mat
                    N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                    possible_product[N_p]["G_list"]   += [deepcopy(N_G)]
                    possible_product[N_p]["adj_list"] += [deepcopy(N_adj_mat)] 
                    possible_product[N_p]["BE_list"]  += [deepcopy(N_BE)]
                    
        else:

            # can't break up double-bond in one step
            if bond_break[0] == bond_break[1]: 
                continue

            # can't break up freezed bonds
            if (bond_break[0][0] in freeze and bond_break[0][1] in freeze) or (bond_break[1][0] in freeze and bond_break[1][1] in freeze):
                continue

            # if the same bond is aleardy broken, skip this bond
            if bond_break not in total_break: total_break += [bond_break]
            else: continue
        
            # Determine whether exist common atom
            common_atom = [item for item, count in collections.Counter(atom_list).items() if count > 1]        
            if len(common_atom) > 0: 
                continue

            # if there is no common atom, two possible new arrangements
            BE_break  = break_bonds(BE,bond_break)
            bonds_form= generate_bond_form(bond_break,atom_list,zw_ind=[cation_ind,anion_ind])

            # loop over possible new bond rearangements
            for bond_form in bonds_form:

                N_BE = form_bonds(BE_break,bond_form)
                N_adj_mat = bond_to_adj(N_BE)

                # Apply canonical operation only to detemine whether this is a new molecule or a duolicated one
                _,_,N_hash_list,N_BE_canon=canon_geo(E,N_adj_mat,bond_mat=N_BE)

                # Calculate the hash value of the atom involved in the bond forming
                form_hash  = sorted([sorted([hash_list[f[0]],hash_list[f[1]]]) for f in bond_form])
                change_hash= break_hash + form_hash

                # check whether all of the conditions are satisfied
                ring_index = identify_rings(E,N_adj_mat)
                check_flag = check_bond_condition(N_BE) and check_ring_condition(ring_index,N_BE) and check_fused_condition(ring_index) 

                if 1 in truncate:
                    check_flag = (check_flag and check_3members_ring(ring_index))
                
                if 2 in truncate:
                    check_flag = (check_flag and check_4members_ring(ring_index))

                if 3 in truncate:
                    check_flag = (check_flag and check_bridge(ring_index))

                # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
                check_flag = (check_flag and  array_unique(change_hash,bond_change_list)[0])

                if array_unique(N_BE_canon,BE_list)[0] and array_unique(N_hash_list,total_hash_list)[0] and check_flag:

                    # A new product is found!
                    BE_list.append(deepcopy(N_BE_canon))
                    total_hash_list.append(deepcopy(N_hash_list))
                    bond_change_list.append(change_hash)
                    
                    # The index of new product
                    N_p = len(BE_list) - 1
                    possible_product[N_p] = {}
                    
                    # Apply force field geometry optimization for each product and adj_mat
                    N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                    Psmile_list,Pinchi_list,Pgroups = split_compounds(E,N_G,N_adj_mat)
                    name = '.'.join(Psmile_list)

                    # write properties into product dictionary
                    possible_product[N_p]["E"]        = E
                    possible_product[N_p]["name"]     = name
                    possible_product[N_p]["hash_list"]= N_hash_list
                    possible_product[N_p]["G_list"]   =[deepcopy(N_G)] 
                    possible_product[N_p]["adj_list"] =[deepcopy(N_adj_mat)]
                    possible_product[N_p]["BE_list"]  =[deepcopy(N_BE)]
                    
                    # write info into separated product dictionary
                    for ind,group in enumerate(Pgroups):
                        inchi_ind = Pinchi_list[ind].split('-')[0]
                
                    # add into dictionary if this compound not shown in dict
                    if inchi_ind not in separated_product.keys():
                        separated_product[inchi_ind] = {}
                        separated_product[inchi_ind]["E"] = [E[ind] for ind in group]
                        separated_product[inchi_ind]["G"] = N_G[group,:]
                        separated_product[inchi_ind]["adj_mat"]= N_adj_mat[group,:][:,group]
                        separated_product[inchi_ind]["source"] = [deepcopy(N_p)]
                                
                    else:
                        # if this compound already exists in dict, add another source of it
                        if N_p not in separated_product[inchi_ind]["source"]:
                            separated_product[inchi_ind]["source"] += [deepcopy(N_p)]

                elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:
                        
                    # append change_hash into the change list
                    bond_change_list.append(change_hash)
                
                    # This is be same product but allows for different reaction pathways
                    N_p = total_hash_list.index(N_hash_list)
                    
                    # exclude conformational change
                    if N_p == 0: continue

                    # check whether this adj_mat exist in adj_mat_list or not
                    if array_unique(N_adj_mat,possible_product[N_p]["adj_list"])[0]:

                        # Apply force field geometry optimization for each product and adj_mat
                        N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                        possible_product[N_p]["G_list"]   += [deepcopy(N_G)]
                        possible_product[N_p]["adj_list"] += [deepcopy(N_adj_mat)] 
                        possible_product[N_p]["BE_list"]  += [deepcopy(N_BE)]        

    return possible_product,separated_product

# define type one reaction for radical species
def R1_b1f1(reactant,truncate=[],ff='uff',phase=1):
    # create a dictionary for products and a dictionary for seperated uni-products
    possible_product = {}
    separated_product= {}

    # obatin elements, adj_mat, BE matrix
    E        = reactant["E"]
    G        = reactant["G"]
    BE       = reactant["BE_mat"]
    adj_mat  = reactant["adj_mat"]
    bond_list= reactant["bond_list"]
    hash_list= reactant["hash_list"]
    # determine the partical part
    radical_part=find_radical(E, BE)
    # Create a list of potential products, BE_list and hash_list are used to identify whether a new prodcut is found or not
    BE_list = []
    total_hash_list = []
    BE_list.append(deepcopy(BE))     
    total_hash_list.append(deepcopy(hash_list))
    
    # get freezed atoms index (like the benzene ring), those bonds involved will not breake 
    freeze   = return_freeze(E,adj_mat,BE)

    # return number of reactant(s) in this structure. If there are multi-reactants, also return the index
    smile_list,inchi_list,groups = split_compounds(E,G,adj_mat)
    
    comb=[bc for bc in combinations(bond_list,1)]
    # initialzie some lists
    total_break = []
    bond_change_list = []
    # loop over all bond changes
    for bond_break in comb:

        # can't break up double-bond in one step
        if bond_break[0][0] == bond_break[0][1]: 
            continue

        # can't break up freezed bonds
        if (bond_break[0][0] in freeze and bond_break[0][1] in freeze):
            continue

        # if the same bond is aleardy broken, skip this bond
        if bond_break not in total_break:
            total_break += [bond_break]     
        else:
            continue
        
        atom_list = []
        for bb in bond_break: 
            atom_list += bb
        
        # Calculate the hash value of the atom involved in the bond breaking, will be used to avoid duplicated reactions
        break_hash = sorted([sorted([hash_list[b[0]],hash_list[b[1]]]) for b in bond_break])

        # Determine whether exist common atom
        common_atom = [item for item, count in collections.Counter(atom_list).items() if count > 1]        

        # Determine possible reaction based on number of common atom                                  
        if len(common_atom) == 0:
            # if there is no common atom, two possible new arrangements
            BE_break  = break_bonds(BE,bond_break)
            bonds_form= generate_bond_form(bond_break,atom_list,lone=radical_part, number_of_formation=1)
            for bond_form in bonds_form:

                N_BE = form_bonds(BE_break,bond_form)
                N_adj_mat = bond_to_adj(N_BE)
                # Apply canonical operation only to detemine whether this is a new molecule or a duolicated one
                _,_,N_hash_list,N_BE_canon=canon_geo(E,N_adj_mat,bond_mat=N_BE)

                # Calculate the hash value of the atom involved in the bond forming
                form_hash  = sorted([sorted([hash_list[bond_form[0]],hash_list[bond_form[1]]])])
                change_hash= break_hash + form_hash

                # check whether all of the conditions are satisfied
                ring_index = identify_rings(E,N_adj_mat)
                check_flag = check_bond_condition(N_BE) and check_ring_condition(ring_index,N_BE) and check_fused_condition(ring_index) 

                if 1 in truncate:
                    check_flag = (check_flag and check_3members_ring(ring_index))
                
                if 2 in truncate:
                    check_flag = (check_flag and check_4members_ring(ring_index))

                if 3 in truncate:
                    check_flag = (check_flag and check_bridge(ring_index))

                # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
                check_flag = (check_flag and  array_unique(change_hash,bond_change_list)[0])

                if array_unique(N_BE_canon,BE_list)[0] and array_unique(N_hash_list,total_hash_list)[0] and check_flag:

                    # A new product is found!
                    BE_list.append(deepcopy(N_BE_canon))
                    total_hash_list.append(deepcopy(N_hash_list))
                    bond_change_list.append(change_hash)
                    
                    # The index of new product
                    N_p = len(BE_list) - 1
                    possible_product[N_p] = {}
                    
                    # Apply force field geometry optimization for each product and adj_mat
                    N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                    Psmile_list,Pinchi_list,Pgroups = split_compounds(E,N_G,N_adj_mat)
                    name = '.'.join(Psmile_list)

                    # write properties into product dictionary
                    possible_product[N_p]["E"]        = E
                    possible_product[N_p]["name"]     = name
                    possible_product[N_p]["hash_list"]= N_hash_list
                    possible_product[N_p]["G_list"]   =[deepcopy(N_G)] 
                    possible_product[N_p]["adj_list"] =[deepcopy(N_adj_mat)]
                    possible_product[N_p]["BE_list"]  =[deepcopy(N_BE)]
                    possible_product[N_p]['type']=['b1f1']
                    # write info into separated product dictionary
                    for ind,group in enumerate(Pgroups):                           
                        inchi_ind = Pinchi_list[ind].split('-')[0]

                        # add into dictionary if this compound not shown in dict
                        if inchi_ind not in separated_product.keys():
                            separated_product[inchi_ind] = {}
                            separated_product[inchi_ind]["E"] = [E[ind] for ind in group]
                            separated_product[inchi_ind]["G"] = N_G[group,:]
                            separated_product[inchi_ind]["adj_mat"]= N_adj_mat[group,:][:,group]
                            separated_product[inchi_ind]["source"] = [deepcopy(N_p)]

                        else:
                            # if this compound already exists in dict, add another source of it
                            if N_p not in separated_product[inchi_ind]["source"]:
                                separated_product[inchi_ind]["source"] += [deepcopy(N_p)]

                elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:
                        
                    # append change_hash into the change list
                    bond_change_list.append(change_hash)
                    
                    # This is be same product but allows for different reaction pathways
                    N_p = total_hash_list.index(N_hash_list)
                    
                    # exclude conformational change
                    if N_p == 0:
                        continue

                    # check whether this adj_mat exist in adj_mat_list or not
                    if array_unique(N_adj_mat,possible_product[N_p]["adj_list"])[0]:

                        # Apply force field geometry optimization for each product and adj_mat
                        N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                        possible_product[N_p]["G_list"]   += [deepcopy(N_G)]
                        possible_product[N_p]["adj_list"] += [deepcopy(N_adj_mat)] 
                        possible_product[N_p]["BE_list"]  += [deepcopy(N_BE)]
                        possible_product[N_p]['type']+=['b1f1']
                        print('b1f1')
                        print(len(N_G))
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
                    form_hash  = sorted([ sorted([hash_list[not_common_atom[0]],hash_list[not_common_atom[1]]]),sorted([hash_list[common_atom[0]],hash_list[c_atom] ]) ])
                    change_hash= break_hash + form_hash 
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
                    check_flag = (check_flag and array_unique(change_hash,bond_change_list)[0])

                    if array_unique(N_BE_canon,BE_list)[0] is True and array_unique(N_hash_list,total_hash_list)[0] is True and check_flag:
                        
                        # A new product is found!
                        BE_list.append(deepcopy(N_BE_canon))
                        total_hash_list.append(deepcopy(N_hash_list))
                        bond_change_list.append(change_hash)
                            
                        # The index of new product
                        N_p = len(BE_list) - 1
                        possible_product[N_p] = {}
                            
                        # Apply force field geometry optimization for each product and adj_mat
                        N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                        if len(N_G) != len(G):
                            N_G = N_G[:len(G)]

                        Psmile_list,Pinchi_list,Pgroups = split_compounds(E,N_G,N_adj_mat)
                        name = '.'.join(Psmile_list)
                            
                        # write properties into product dictionary
                        possible_product[N_p]["E"]        = E
                        possible_product[N_p]["name"]     = name
                        possible_product[N_p]["hash_list"]= N_hash_list
                        possible_product[N_p]["G_list"]   =[deepcopy(N_G)] 
                        possible_product[N_p]["adj_list"] =[deepcopy(N_adj_mat)]
                        possible_product[N_p]["BE_list"]  =[deepcopy(N_BE)]
                        possible_product[N_p]['type']+=['b1f1']
                        # write info into separated product dictionary
                        for ind,group in enumerate(Pgroups):
                            inchi_ind = Pinchi_list[ind].split('-')[0]
                                
                            # add into dictionary if this compound not shown in dict
                            if inchi_ind not in separated_product.keys():
                                separated_product[inchi_ind] = {}
                                separated_product[inchi_ind]["E"] = [E[ind] for ind in group]
                                separated_product[inchi_ind]["G"] = N_G[group,:]
                                separated_product[inchi_ind]["adj_mat"]= N_adj_mat[group,:][:,group]
                                separated_product[inchi_ind]["source"] = [deepcopy(N_p)]

                            else:
                                # if this compound already exists in dict, add another source of it
                                if N_p not in separated_product[inchi_ind]["source"]:
                                    separated_product[inchi_ind]["source"] += [deepcopy(N_p)]  
                    elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:

                        # This is be same product but allows for different reaction pathways
                        N_p = total_hash_list.index(N_hash_list)

                        # exclude conformational change
                        if N_p == 0:
                            continue

                        # check whether this adj_mat exist in adj_mat_list or not
                        if array_unique(N_adj_mat,possible_product[N_p]["adj_list"])[0]:

                            # Apply force field geometry optimization for each product and adj_mat
                            N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                            if len(N_G) != len(G):
                                N_G = N_G[:len(G)]
                            possible_product[N_p]["G_list"]   += [deepcopy(N_G)]
                            possible_product[N_p]["adj_list"] += [deepcopy(N_adj_mat)]
                            possible_product[N_p]["BE_list"]  += [deepcopy(N_BE)]
                            possible_product[N_p]['type']+=['b1f1']
    return possible_product,separated_product

def R1_b2f1(reactant, truncate=[], ff='uff', phase=1, start_ind=0):
    # Use break 2 form 1 process to generate the radical products from neutral compounds
    possible_product = {}                                                                                                                                                                           
    separated_product= {}

    # obatin elements, adj_mat, BE matrix
    E        = reactant["E"]
    G        = reactant["G"]
    BE       = reactant["BE_mat"]
    adj_mat  = reactant["adj_mat"]
    bond_list= reactant["bond_list"]
    hash_list= reactant["hash_list"]
    radical_part=find_radical(E, BE)
    # Create a list of potential products, BE_list and hash_list are used to identify whether a new prodcut is found or not
    BE_list = []
    total_hash_list = []
    BE_list.append(deepcopy(BE))     
    total_hash_list.append(deepcopy(hash_list))
    
    # get freezed atoms index (like the benzene ring), those bonds involved will not breake 
    freeze   = return_freeze(E,adj_mat,BE)

    # return number of reactant(s) in this structure. If there are multi-reactants, also return the index
    smile_list,inchi_list,groups = split_compounds(E,G,adj_mat)

    # determine which phase will this enumeration be and whether the input reactant (uni/bi) matches with the phase
    if phase == 1 and len(groups) == 1:

        print("Phase 1 enumeration: unimolecular transformation...")

        # generate all possible C_N^2 combinations
        comb = [bc for bc in combinations(bond_list, 2)]

    # if phase is 2 and number of reactant is 2
    elif phase == 2 and len(groups) == 2:

        # generate all possible C_N^2 combinations, only keep it when break two bonds in two reactant
        comb = [bc for bc in combinations(bond_list, 2) if [bc[0][0] in group for group in groups].index(True) != [bc[1][0] in group for group in groups].index(True) ]

    else:
        print("Only two phases available, make sure phase and number of reactant matches, quit...")
        quit()

    # initialzie some lists
    total_break = []
    bond_change_list = []

    # loop over all bond changes
    for bond_break in comb:

        # can't break up double-bond in one step
        if bond_break[0] == bond_break[1]: 
            continue

        # can't break up freezed bonds
        if (bond_break[0][0] in freeze and bond_break[0][1] in freeze) or (bond_break[1][0] in freeze and bond_break[1][1] in freeze):
            continue                                 
        # if the same bond is aleardy broken, skip this bond
        if bond_break not in total_break:
            total_break += [bond_break]
        else:
            continue
        
        atom_list = []
        for bb in bond_break: 
            atom_list += bb
        
        # Calculate the hash value of the atom involved in the bond breaking, will be used to avoid duplicated reactions
        break_hash = sorted([sorted([hash_list[b[0]],hash_list[b[1]]]) for b in bond_break])

        # Determine whether exist common atom
        common_atom = [item for item, count in collections.Counter(atom_list).items() if count > 1]        

        # Determine possible reaction based on number of common atom
        if len(common_atom) == 0:

            # if there is no common atom, two possible new arrangements
            BE_break  = break_bonds(BE,bond_break)
            bonds_form= generate_bond_form(bond_break,atom_list,lone=radical_part, number_of_formation=1)

            for bond_form in bonds_form:

                N_BE = form_bonds(BE_break,bond_form)
                N_adj_mat = bond_to_adj(N_BE)

                # Apply canonical operation only to detemine whether this is a new molecule or a duolicated one
                _,_,N_hash_list,N_BE_canon=canon_geo(E,N_adj_mat,bond_mat=N_BE)

                # Calculate the hash value of the atom involved in the bond forming
                form_hash  = form_hash  = sorted([sorted([hash_list[bond_form[0]],hash_list[bond_form[1]]])])
                change_hash= break_hash + form_hash

                # check whether all of the conditions are satisfied
                ring_index = identify_rings(E,N_adj_mat)
                check_flag = check_bond_condition(N_BE) and check_ring_condition(ring_index,N_BE) and check_fused_condition(ring_index) 

                if 1 in truncate:
                    check_flag = (check_flag and check_3members_ring(ring_index))
                
                if 2 in truncate:
                    check_flag = (check_flag and check_4members_ring(ring_index))

                if 3 in truncate:
                    check_flag = (check_flag and check_bridge(ring_index))

                # determine whether change_hash is unique, if not, identical/equivalent bond changing occurs
                check_flag = (check_flag and  array_unique(change_hash,bond_change_list)[0])

                if array_unique(N_BE_canon,BE_list)[0] and array_unique(N_hash_list,total_hash_list)[0] and check_flag:

                    # A new product is found!
                    BE_list.append(deepcopy(N_BE_canon))
                    total_hash_list.append(deepcopy(N_hash_list))
                    bond_change_list.append(change_hash)                              
                    # The index of new product
                    N_p = len(BE_list) - 1
                    possible_product[N_p] = {}
                    # Apply force field geometry optimization for each product and adj_mat
                    N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                    Psmile_list,Pinchi_list,Pgroups = split_compounds(E,N_G,N_adj_mat)
                    name = '.'.join(Psmile_list)

                    # write properties into product dictionary
                    possible_product[N_p]["E"]        = E
                    possible_product[N_p]["name"]     = name
                    possible_product[N_p]["hash_list"]= N_hash_list
                    possible_product[N_p]["G_list"]   =[deepcopy(N_G)]
                    possible_product[N_p]["adj_list"] =[deepcopy(N_adj_mat)]
                    possible_product[N_p]["BE_list"]  =[deepcopy(N_BE)]
                        
                    # write info into separated product dictionary
                    for ind,group in enumerate(Pgroups):
                        inchi_ind = Pinchi_list[ind].split('-')[0]

                        # add into dictionary if this compound not shown in dict
                        if inchi_ind not in separated_product.keys():
                            separated_product[inchi_ind] = {}
                            separated_product[inchi_ind]["E"] = [E[ind] for ind in group]
                            separated_product[inchi_ind]["G"] = N_G[group,:]
                            separated_product[inchi_ind]["adj_mat"]= N_adj_mat[group,:][:,group]
                            separated_product[inchi_ind]["source"] = [deepcopy(N_p)]
                                
                        else:
                            # if this compound already exists in dict, add another source of it
                            if N_p not in separated_product[inchi_ind]["source"]:
                                separated_product[inchi_ind]["source"] += [deepcopy(N_p)]

                elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:
                        
                    # append change_hash into the change list
                    bond_change_list.append(change_hash)
                    
                    # This is be same product but allows for different reaction pathways
                    N_p = total_hash_list.index(N_hash_list)

                    # exclude conformational change
                    if N_p == 0:
                        continue

                    # check whether this adj_mat exist in adj_mat_list or not
                    if array_unique(N_adj_mat,possible_product[N_p]["adj_list"])[0]:

                        # Apply force field geometry optimization for each product and adj_mat
                        N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                        possible_product[N_p]["G_list"]   += [deepcopy(N_G)]
                        possible_product[N_p]["adj_list"] += [deepcopy(N_adj_mat)] 
                        possible_product[N_p]["BE_list"]  += [deepcopy(N_BE)]
        
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
                    form_hash  = sorted([ sorted([hash_list[not_common_atom[0]],hash_list[not_common_atom[1]]]),sorted([hash_list[common_atom[0]],hash_list[c_atom] ]) ])
                    change_hash= break_hash + form_hash

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
                    check_flag = (check_flag and array_unique(change_hash,bond_change_list)[0])
                    if array_unique(N_BE_canon,BE_list)[0] is True and array_unique(N_hash_list,total_hash_list)[0] is True and check_flag:
                        
                        # A new product is found!
                        BE_list.append(deepcopy(N_BE_canon))
                        total_hash_list.append(deepcopy(N_hash_list))
                        bond_change_list.append(change_hash)

                        # The index of new product
                        N_p = len(BE_list) - 1
                        possible_product[N_p] = {}
                            
                        # Apply force field geometry optimization for each product and adj_mat
                        N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                        if len(N_G) != len(G):
                            N_G = N_G[:len(G)]

                        Psmile_list,Pinchi_list,Pgroups = split_compounds(E,N_G,N_adj_mat)
                        name = '.'.join(Psmile_list)
                            
                        # write properties into product dictionary
                        possible_product[N_p]["E"]        = E
                        possible_product[N_p]["name"]     = name
                        possible_product[N_p]["hash_list"]= N_hash_list
                        possible_product[N_p]["G_list"]   =[deepcopy(N_G)] 
                        possible_product[N_p]["adj_list"] =[deepcopy(N_adj_mat)]
                        possible_product[N_p]["BE_list"]  =[deepcopy(N_BE)]
                            
                        # write info into separated product dictionary      
                        # write info into separated product dictionary
                        for ind,group in enumerate(Pgroups):
                            inchi_ind = Pinchi_list[ind].split('-')[0]
                            # add into dictionary if this compound not shown in dict
                            if inchi_ind not in separated_product.keys():
                                separated_product[inchi_ind] = {}
                                separated_product[inchi_ind]["E"] = [E[ind] for ind in group]
                                separated_product[inchi_ind]["G"] = N_G[group,:]
                                separated_product[inchi_ind]["adj_mat"]= N_adj_mat[group,:][:,group]
                                separated_product[inchi_ind]["source"] = [deepcopy(N_p)]

                            else:
                                # if this compound already exists in dict, add another source of it
                                if N_p not in separated_product[inchi_ind]["source"]:
                                    separated_product[inchi_ind]["source"] += [deepcopy(N_p)]

                    elif check_flag and array_unique(N_hash_list,total_hash_list)[0] is False:

                        # This is be same product but allows for different reaction pathways
                        N_p = total_hash_list.index(N_hash_list)

                        # exclude conformational change
                        if N_p == 0:
                            continue

                        # check whether this adj_mat exist in adj_mat_list or not
                        if array_unique(N_adj_mat,possible_product[N_p]["adj_list"])[0]:
                        
                            # Apply force field geometry optimization for each product and adj_mat
                            N_G = opt_geo(deepcopy(G),N_adj_mat,E,ff=ff,step=1000)
                            if len(N_G) != len(G):
                                N_G = N_G[:len(G)]
                            possible_product[N_p]["G_list"]   += [deepcopy(N_G)]
                            possible_product[N_p]["adj_list"] += [deepcopy(N_adj_mat)] 
                            possible_product[N_p]["BE_list"]  += [deepcopy(N_BE)]
                                
    return possible_product,separated_product

def find_radical(element,bond_mat):   
    radical_part=[]
    for count_i, i in enumerate(bond_mat):
        summation=0.0                                                                                                                                                                                             
        for count_j, j in enumerate(i):
            if count_j==count_i: summation+=j
            else: summation+=j*2.0
        if (element[count_i]!='H' or element[count_i]!='h') and (summation%8.0)==0.0: continue
        elif (element[count_i]=='H' or element[count_i]=='h') and summation==2.0: continue
        else: radical_part.append(count_i)

    return radical_part
