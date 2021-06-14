import sys,argparse,os,time,math,subprocess
import random
import ast
import collections
import numpy as np
from scipy.spatial.distance import cdist
from copy import deepcopy
from itertools import combinations 

# Generates the adjacency matrix based on UFF bond radii
# Inputs:       Elements: N-element List strings for each atom type
#               Geometry: Nx3 array holding the geometry of the molecule
#               File:  Optional. If Table_generator encounters a problem then it is often useful to have the name of the file the geometry came from printed. 
# Outputs:      Adjacency matrix
#
def Table_generator(Elements,Geometry,File=None):

    # Initialize UFF bond radii (Rappe et al. JACS 1992)
    # NOTE: Units of angstroms 
    # NOTE: These radii neglect the bond-order and electronegativity corrections in the original paper. Where several values exist for the same atom, the largest was used. 
    Radii = {  'H':0.354, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.244, 'Si':1.117,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.473, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    # SAME AS ABOVE BUT WITH A SMALLER VALUE FOR THE Al RADIUS ( I think that it tends to predict a bond where none are expected
    Radii = {  'H':0.39, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.15,  'Si':1.050,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.473, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    Max_Bonds = {  'H':2,    'He':1,\
                  'Li':None, 'Be':None,                                                                                                                'B':4,     'C':4,     'N':4,     'O':2,     'F':1,    'Ne':1,\
                  'Na':None, 'Mg':None,                                                                                                               'Al':4,    'Si':4,  'P':None,  'S':None, 'Cl':1,    'Ar':1,\
                   'K':None, 'Ca':None, 'Sc':None, 'Ti':None,  'V':None, 'Cr':None, 'Mn':None, 'Fe':None, 'Co':None, 'Ni':None, 'Cu':None, 'Zn':None, 'Ga':None, 'Ge':None, 'As':None, 'Se':None, 'Br':1,    'Kr':None,\
                  'Rb':None, 'Sr':None,  'Y':None, 'Zr':None, 'Nb':None, 'Mo':None, 'Tc':None, 'Ru':None, 'Rh':None, 'Pd':None, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':1,    'Xe':None,\
                  'Cs':None, 'Ba':None, 'La':None, 'Hf':None, 'Ta':None,  'W':None, 'Re':None, 'Os':None, 'Ir':None, 'Pt':None, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }
                     
    # Scale factor is used for determining the bonding threshold. 1.2 is a heuristic that give some lattitude in defining bonds since the UFF radii correspond to equilibrium lengths. 
    scale_factor = 1.2

    # Print warning for uncoded elements.
    for i in Elements:
        if i not in Radii.keys():
            print("ERROR in Table_generator: The geometry contains an element ({}) that the Table_generator function doesn't have bonding information for. This needs to be directly added to the Radii".format(i)+\
                  " dictionary before proceeding. Exiting...")
            quit()

    # Generate distance matrix holding atom-atom separations (only save upper right)
    Dist_Mat = np.triu(cdist(Geometry,Geometry))
    
    # Find plausible connections
    x_ind,y_ind = np.where( (Dist_Mat > 0.0) & (Dist_Mat < max([ Radii[i]**2.0 for i in Radii.keys() ])) )

    # Initialize Adjacency Matrix
    Adj_mat = np.zeros([len(Geometry),len(Geometry)])

    # Iterate over plausible connections and determine actual connections
    for count,i in enumerate(x_ind):
        
        # Assign connection if the ij separation is less than the UFF-sigma value times the scaling factor
        if Dist_Mat[i,y_ind[count]] < (Radii[Elements[i]]+Radii[Elements[y_ind[count]]])*scale_factor:            
            Adj_mat[i,y_ind[count]]=1
    
    # Hermitize Adj_mat
    Adj_mat=Adj_mat + Adj_mat.transpose()

    # Perform some simple checks on bonding to catch errors
    problem_dict = { i:0 for i in Radii.keys() }
    conditions = { "H":1, "C":4, "F":1, "Cl":1, "Br":1, "I":1, "O":2, "N":4, "B":4 }
    for count_i,i in enumerate(Adj_mat):

        if Max_Bonds[Elements[count_i]] is not None and sum(i) > Max_Bonds[Elements[count_i]]:
            problem_dict[Elements[count_i]] += 1
            cons = sorted([ (Dist_Mat[count_i,count_j],count_j) if count_j > count_i else (Dist_Mat[count_j,count_i],count_j) for count_j,j in enumerate(i) if j == 1 ])[::-1]
            while sum(Adj_mat[count_i]) > Max_Bonds[Elements[count_i]]:
                sep,idx = cons.pop(0)
                Adj_mat[count_i,idx] = 0
                Adj_mat[idx,count_i] = 0
#        if Elements[count_i] in conditions.keys():
#            if sum(i) > conditions[Elements[count_i]]:


    # Print warning messages for obviously suspicious bonding motifs.
    if sum( [ problem_dict[i] for i in problem_dict.keys() ] ) > 0:
        print("Table Generation Warnings:")
        for i in sorted(problem_dict.keys()):
            if problem_dict[i] > 0:
                if File is None:
                    if i == "H":  print("WARNING in Table_generator: {} hydrogen(s) have more than one bond.".format(problem_dict[i]))
                    if i == "C":  print("WARNING in Table_generator: {} carbon(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "Si": print("WARNING in Table_generator: {} silicons(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "F":  print("WARNING in Table_generator: {} fluorine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "Cl": print("WARNING in Table_generator: {} chlorine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "Br": print("WARNING in Table_generator: {} bromine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "I":  print("WARNING in Table_generator: {} iodine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "O":  print("WARNING in Table_generator: {} oxygen(s) have more than two bonds.".format(problem_dict[i]))
                    if i == "N":  print("WARNING in Table_generator: {} nitrogen(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "B":  print("WARNING in Table_generator: {} bromine(s) have more than four bonds.".format(problem_dict[i]))
                else:
                    if i == "H": print("WARNING in Table_generator: parsing {}, {} hydrogen(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "C": print("WARNING in Table_generator: parsing {}, {} carbon(s) have more than four bonds.".format(File,problem_dict[i]))
                    if i == "Si": print("WARNING in Table_generator: parsing {}, {} silicons(s) have more than four bonds.".format(File,problem_dict[i]))
                    if i == "F": print("WARNING in Table_generator: parsing {}, {} fluorine(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "Cl": print("WARNING in Table_generator: parsing {}, {} chlorine(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "Br": print("WARNING in Table_generator: parsing {}, {} bromine(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "I": print("WARNING in Table_generator: parsing {}, {} iodine(s) have more than one bond.".format(File,problem_dict[i]))
                    if i == "O": print("WARNING in Table_generator: parsing {}, {} oxygen(s) have more than two bonds.".format(File,problem_dict[i]))
                    if i == "N": print("WARNING in Table_generator: parsing {}, {} nitrogen(s) have more than four bonds.".format(File,problem_dict[i]))
                    if i == "B": print("WARNING in Table_generator: parsing {}, {} bromine(s) have more than four bonds.".format(File,problem_dict[i]))
        print ("")

    return Adj_mat


# Returns a list with the number of electrons on each atom and a list with the number missing/surplus electrons on the atom
# 
# Inputs:  elements:  a list of element labels indexed to the adj_mat 
#          adj_mat:   np.array of atomic connections
#          bonding_pref: optional list of (index, bond_number) tuples that sets the target bond number of the indexed atoms
#          q_tot:     total charge on the molecule
#          fixed_bonds: optional list of (index_1,index_2,bond_number) tuples that creates fixed bonds between the index_1
#                       and index_2 atoms. No further bonds will be added or subtracted between these atoms.
#
# Optional inputs for ion and radical cases:
#          fc_0:      a list of formal charges on each atom
#          keep_lone: a list of atom index for which contains a radical 
#
# Returns: lone_electrons:
#          bonding_electrons:
#          core_electrons:
#          bond_mat:  an NxN matrix holding the bond orders between all atoms in the adj_mat
#          bonding_pref (optinal): optional list of (index, bond_number) tuples that sets the target bond number of the indexed atoms  
#
def find_lewis(elements,adj_mat_0,bonding_pref=[],q_tot=0,fixed_bonds=[],fc_0=None,keep_lone=[],return_pref=False,verbose=False,b_mat_only=False,return_FC=False,octet_opt=True,check_lewis_flag=False):
    
    # Initialize the preferred lone electron dictionary the first time this function is called
    if not hasattr(find_lewis, "sat_dict"):

        find_lewis.lone_e = {'h':0, 'he':2,\
                             'li':0, 'be':2,                                                                                                                'b':0,     'c':0,     'n':2,     'o':4,     'f':6,    'ne':8,\
                             'na':0, 'mg':2,                                                                                                               'al':0,    'si':0,     'p':2,     's':4,    'cl':6,    'ar':8,\
                             'k':0, 'ca':2, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':None, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':10, 'ge':0,    'as':3,    'se':4,    'br':6,    'kr':None,\
                             'rb':0, 'sr':2,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':None, 'sn':None, 'sb':None, 'te':None,  'i':6,    'xe':None,\
                             'cs':0, 'ba':2, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':None, 'pb':None, 'bi':None, 'po':None, 'at':None, 'rn':None }

        # Initialize periodic table
        find_lewis.periodic = { "h": 1,  "he": 2,\
                                 "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                                 "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                  "k":19, "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                                 "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                                 "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}
        
        # Electronegativity ordering (for determining lewis structure)
        find_lewis.en = { "h" :2.3,  "he":4.16,\
                          "li":0.91, "be":1.58,                                                                                                               "b" :2.05, "c" :2.54, "n" :3.07, "o" :3.61, "f" :4.19, "ne":4.79,\
                          "na":0.87, "mg":1.29,                                                                                                               "al":1.61, "si":1.91, "p" :2.25, "s" :2.59, "cl":2.87, "ar":3.24,\
                          "k" :0.73, "ca":1.03, "sc":1.19, "ti":1.38, "v": 1.53, "cr":1.65, "mn":1.75, "fe":1.80, "co":1.84, "ni":1.88, "cu":1.85, "zn":1.59, "ga":1.76, "ge":1.99, "as":2.21, "se":2.42, "br":2.69, "kr":2.97,\
                          "rb":0.71, "sr":0.96, "y" :1.12, "zr":1.32, "nb":1.41, "mo":1.47, "tc":1.51, "ru":1.54, "rh":1.56, "pd":1.58, "ag":1.87, "cd":1.52, "in":1.66, "sn":1.82, "sb":1.98, "te":2.16, "i" :2.36, "xe":2.58,\
                          "cs":0.66, "ba":0.88, "la":1.09, "hf":1.16, "ta":1.34, "w" :1.47, "re":1.60, "os":1.65, "ir":1.68, "pt":1.72, "au":1.92, "hg":1.76, "tl":1.79, "pb":1.85, "bi":2.01, "po":2.19, "at":2.39, "rn":2.60} 

        # Polarizability ordering (for determining lewis structure)
        find_lewis.pol ={ "h" :4.5,  "he":1.38,\
                          "li":164.0, "be":377,                                                                                                               "b" :20.5, "c" :11.3, "n" :7.4, "o" :5.3,  "f" :3.74, "ne":2.66,\
                          "na":163.0, "mg":71.2,                                                                                                              "al":57.8, "si":37.3, "p" :25.0,"s" :19.4, "cl":14.6, "ar":11.1,\
                          "k" :290.0, "ca":161.0, "sc":97.0, "ti":100.0, "v": 87.0, "cr":83.0, "mn":68.0, "fe":62.0, "co":55, "ni":49, "cu":47.0, "zn":38.7,  "ga":50.0, "ge":40.0, "as":30.0,"se":29.0, "br":21.0, "kr":16.8,\
                          "rb":320.0, "sr":197.0, "y" :162,  "zr":112.0, "nb":98.0, "mo":87.0, "tc":79.0, "ru":72.0, "rh":66, "pd":26.1, "ag":55, "cd":46.0,  "in":65.0, "sn":53.0, "sb":43.0,"te":28.0, "i" :32.9, "xe":27.3,}

        # Bond energy dictionary {}-{}-{} refers to atom1, atom2 additional bonds number (1 refers to double bonds)
        # If energy for multiple bonds is missing, it means it's unusual to form multiple bonds, such value will be -10000.0, if energy for single bonds if missing, directly take multiple bonds energy as the difference 
        # From https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Chemical_Bonding/Fundamentals_of_Chemical_Bonding/Bond_Energies
        #find_lewis.be = { "6-6-1": 267, "6-6-2":492, "6-7-1":310, "6-7-2":586, "6-8-1":387, "6-8-2":714, "7-8-1":406, "7-7-1":258, "7-7-2":781, "8-8-1":349, "8-16-1":523, "16-16-1":152}
        # Or from https://www2.chemistry.msu.edu/faculty/reusch/OrgPage/bndenrgy.htm ("6-15-1" is missing)
        # Remove 6-16-1:73
        find_lewis.be = { "6-6-1": 63, "6-6-2":117, "6-7-1":74, "6-7-2":140, "6-8-1":92.5, "6-8-2":172.5, "7-7-1":70.6, "7-7-2":187.6, "7-8-1":88, "8-8-1":84, "8-15-1":20, "8-16-1":6, "15-15-1":84,"15-15-2": 117, "15-16-1":70}
        
        # Initialize periodic table
        find_lewis.atomic_to_element = { find_lewis.periodic[i]:i for i in find_lewis.periodic.keys() }

    # Consistency check on fc_0 argument, if supplied
    if fc_0 is not None:
        if len(fc_0) != len(elements):
            print("ERROR in find_lewis: the fc_0 and elements lists must have the same dimensions.")
            quit()
        if int(sum(fc_0)) != int(q_tot):
            print("ERROR in find_lewis: the sum of formal charges does not equal q_tot.")
            quit()

    # Initalize elementa and atomic_number lists for use by the function
    atomic_number = [ find_lewis.periodic[i.lower()] for i in elements ]
    adj_mat = deepcopy(adj_mat_0)

    # Initially assign all valence electrons as lone electrons
    lone_electrons    = np.zeros(len(elements),dtype="int")    
    bonding_electrons = np.zeros(len(elements),dtype="int")    
    core_electrons    = np.zeros(len(elements),dtype="int")
    valence           = np.zeros(len(elements),dtype="int")
    bonding_target    = np.zeros(len(elements),dtype="int")
    valence_list      = np.zeros(len(elements),dtype="int")    
    
    for count_i,i in enumerate(elements):

        # Grab the total number of (expected) electrons from the atomic number
        N_tot = atomic_number[count_i]   

        # Determine the number of core/valence electrons based on row in the periodic table
        if N_tot > 54:
            print("ERROR in find_lewis: the algorithm isn't compatible with atomic numbers greater than 54 owing to a lack of rules for treating lanthanides. Exiting...")
            quit()
        elif N_tot > 36:
            N_tot -= 36
            core_electrons[count_i] = 36
            valence[count_i]        = 18
        elif N_tot > 18:
            N_tot -= 18
            core_electrons[count_i] = 18
            valence[count_i]        = 18
        elif N_tot > 10:
            N_tot -= 10
            core_electrons[count_i] = 10
            valence[count_i]        = 8
        elif N_tot > 2:
            N_tot -= 2
            core_electrons[count_i] = 2
            valence[count_i]        = 8
        lone_electrons[count_i] = N_tot
        valence_list[count_i] = N_tot

        # Assign target number of bonds for this atom
        if count_i in [ j[0] for j in bonding_pref ]:
            bonding_target[count_i] = next( j[1] for j in bonding_pref if j[0] == count_i )
        else:
            bonding_target[count_i] = N_tot - find_lewis.lone_e[elements[count_i].lower()]       

    # Loop over the adjmat and assign initial bonded electrons assuming single bonds (and adjust lone electrons accordingly)
    for count_i,i in enumerate(adj_mat_0):
        bonding_electrons[count_i] += sum(i)
        lone_electrons[count_i] -= sum(i)

    # Apply keep_lone: add one electron to such index    
    for count_i in keep_lone:
        lone_electrons[count_i] += 1
        
    # Eliminate all radicals by forming higher order bonds
    change_list = range(len(lone_electrons))
    bonds_made = []    
    loop_list   = [ (atomic_number[i],i) for i in range(len(lone_electrons)) ]
    loop_list   = [ i[1] for i in sorted(loop_list) ]

    # Check for special chemical groups
    for i in range(len(elements)):

        # Handle nitro groups
        if is_nitro(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ]
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],1)]
            bonding_pref += [(O_ind[1],2)]
            bonding_electrons[O_ind[1]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind[1]] -= 1
            lone_electrons[i] -= 2
            lone_electrons[O_ind[0]] += 1
            adj_mat[i,O_ind[1]] += 1
            adj_mat[O_ind[1],i] += 1

        # Handle sulfoxide groups
        if is_sulfoxide(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the thioketone atoms from the bonding_pref list
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],2)]
            bonding_electrons[O_ind[0]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind[0]] -= 1
            lone_electrons[i] -= 1
            bonds_made += [(i,O_ind[0])]
            adj_mat[i,O_ind[0]] += 1
            adj_mat[O_ind[0],i] += 1

        # Handle sulfonyl groups
        if is_sulfonyl(i,adj_mat_0,elements) is True:
            
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the sulfoxide atoms from the bonding_pref list
            bonding_pref += [(i,6)]
            bonding_pref += [(O_ind[0],2)]
            bonding_pref += [(O_ind[1],2)]
            bonding_electrons[O_ind[0]] += 1
            bonding_electrons[O_ind[1]] += 1
            bonding_electrons[i] += 2
            lone_electrons[O_ind[0]] -= 1
            lone_electrons[O_ind[1]] -= 1
            lone_electrons[i] -= 2
            bonds_made += [(i,O_ind[0])]
            bonds_made += [(i,O_ind[1])]
            adj_mat[i,O_ind[0]] += 1
            adj_mat[i,O_ind[1]] += 1
            adj_mat[O_ind[0],i] += 1
            adj_mat[O_ind[1],i] += 1            
        
        # Handle phosphate groups 
        if is_phosphate(i,adj_mat_0,elements) is True:
            O_ind      = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j] in ["o","O"] ] # Index of single bonded O-P oxygens
            O_ind_term = [ j for j in O_ind if sum(adj_mat_0[j]) == 1 ] # Index of double bonded O-P oxygens
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the phosphate atoms from the bonding_pref list
            bonding_pref += [(i,5)]
            bonding_pref += [(O_ind_term[0],2)]  # during testing it ended up being important to only add a bonding_pref tuple for one of the terminal oxygens
            bonding_electrons[O_ind_term[0]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind_term[0]] -= 1
            lone_electrons[i] -= 1
            bonds_made += [(i,O_ind_term[0])]
            adj_mat[i,O_ind_term[0]] += 1
            adj_mat[O_ind_term[0],i] += 1

        # Handle cyano groups
        if is_cyano(i,adj_mat_0,elements) is True:
            C_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j] in  ["c","C"] and sum(adj_mat_0[count_j]) == 2 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in C_ind ] # remove bonds involving the cyano atoms from the bonding_pref list
            bonding_pref += [(i,3)]
            bonding_pref += [(C_ind[0],4)]
            bonding_electrons[C_ind[0]] += 2
            bonding_electrons[i] += 2
            lone_electrons[C_ind[0]] -= 2
            lone_electrons[i] -= 2
            bonds_made += [(i,C_ind[0])]
            bonds_made += [(i,C_ind[0])]
            adj_mat[i,C_ind[0]] += 2
            adj_mat[C_ind[0],i] += 2

        # Handle isocyano groups
        if is_isocyano(i,adj_mat,elements) is True:
            C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in  ["c","C"] and sum(adj_mat[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in C_ind ] # remove bonds involving the cyano atoms from the bonding_pref list
            bonding_pref += [(i,4)]
            bonding_pref += [(C_ind[0],3)]
            bonding_electrons[C_ind[0]] += 2
            bonding_electrons[i] += 2
            lone_electrons[C_ind[0]] -= 2
            lone_electrons[i] -= 2
            bonds_made += [(i,C_ind[0])]
            bonds_made += [(i,C_ind[0])]
            adj_mat[i,C_ind[0]] += 2
            adj_mat[C_ind[0],i] += 2

    # Apply fixed_bonds argument
    off_limits=[]
    for i in fixed_bonds:

        # Initalize intermediate variables
        a = i[0]
        b = i[1]
        N = i[2]
        N_current = len([ j for j in bonds_made if (a,b) == j or (b,a) == j ]) + 1
        # Check that a bond exists between these atoms in the adjacency matrix
        if adj_mat_0[a,b] != 1:
            print("ERROR in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but the adjacency matrix doesn't reflect a bond. Exiting...")
            quit()

        # Check that less than or an equal number of bonds exist between these atoms than is requested
        if N_current > N:
            print("ERROR in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but {} bonds already exist between these atoms. There may be a conflict".format(N_current))
            print("                      between the special groups handling and the requested lewis_structure.")
            quit()

        # Check that enough lone electrons exists on each atom to reach the target bond number
        if lone_electrons[a] < (N - N_current):
            print("Warning in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but atom {} only has {} lone electrons.".format(elements[a],lone_electrons[a]))

        # Check that enough lone electrons exists on each atom to reach the target bond number
        if lone_electrons[b] < (N - N_current):
            print("Warning in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but atom {} only has {} lone electrons.".format(elements[b],lone_electrons[b]))
        

        # Make the bonds between the atoms
        for j in range(N-N_current):
            bonding_electrons[a] += 1
            bonding_electrons[b] += 1
            lone_electrons[a]    -= 1
            lone_electrons[b]    -= 1
            bonds_made += [ (a,b) ]

        # Append bond to off_limits group so that further bond additions/breaks do not occur.
        off_limits += [(a,b),(b,a)]

    # Turn the off_limits list into a set for rapid lookup
    off_limits = set(off_limits)
    
    # Adjust formal charges (if supplied)
    if fc_0 is not None:
        for count_i,i in enumerate(fc_0):
            if i > 0:
                #if lone_electrons[count_i] < i:
                    #print "ERROR in find_lewis: atom ({}, index {}) doesn't have enough lone electrons ({}) to be removed to satisfy the specified formal charge ({}).".format(elements[count_i],count_i,lone_electrons[count_i],i)
                    #quit()
                lone_electrons[count_i] = lone_electrons[count_i] - i
            if i < 0:
                lone_electrons[count_i] = lone_electrons[count_i] + int(abs(i))
        q_tot=0
    
    # diagnostic print            
    if verbose is True:
        print("Starting electronic structure:")
        print("\n{:40s} {:20} {:20} {:20} {:20} {}".format("elements","lone_electrons","bonding_electrons","core_electrons","formal_charge","bonded_atoms"))
        for count_i,i in enumerate(elements):
            print("{:40s} {:<20d} {:<20d} {:<20d} {:<20d} {}".format(elements[count_i],lone_electrons[count_i],bonding_electrons[count_i],core_electrons[count_i],\
                                                                     valence_list[count_i] - bonding_electrons[count_i] - lone_electrons[count_i],\
                                                                     ",".join([ "{}".format(count_j) for count_j,j in enumerate(adj_mat[count_i]) if j == 1 ])))

    # Initialize objects for use in the algorithm
    lewis_total = 1000
    lewis_lone_electrons = []
    lewis_bonding_electrons = []
    lewis_core_electrons = []
    lewis_valence = []
    lewis_bonding_target = []
    lewis_bonds_made = []
    lewis_adj_mat = []
    lewis_identical_mat = []
    
    # Determine the atoms with lone pairs that are unsatisfied as candidates for electron removal/addition to satisfy the total charge condition  
    happy = [ i[0] for i in bonding_pref if i[1] <= bonding_electrons[i[0]]]
    bonding_pref_ind = [i[0] for i in bonding_pref]
        
    # Determine is electrons need to be removed or added
    if q_tot > 0:
        adjust = -1
        octet_violate_e = []
        for count_j,j in enumerate(elements):
            if j.lower() in ["c","n","o","f","si","p","s","cl"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] > 8:
                    octet_violate_e += [count_j]
            elif j.lower() in ["br","i"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] > 18:
                    octet_violate_e += [count_j]
        
        normal_adjust = [ count_i for count_i,i in enumerate(lone_electrons) if i > 0 and count_i not in happy and count_i not in octet_violate_e]
    
    elif q_tot < 0:
        adjust = 1
        octet_violate_e = []
        for count_j,j in enumerate(elements):
            if j.lower() in ["c","n","o","f","si","p","s","cl"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] < 8:
                    octet_violate_e += [count_j]
                    
            elif j.lower() in ["br","i"] and count_j not in bonding_pref_ind:
                if bonding_electrons[count_j]*2 + lone_electrons[count_j] < 18:
                    octet_violate_e += [count_j]

        normal_adjust = [ count_i for count_i,i in enumerate(lone_electrons) if i > 0 and count_i not in happy and count_i not in octet_violate_e]
        
    else:
        adjust = 1
        octet_violate_e = []
        normal_adjust = [ count_i for count_i,i in enumerate(lone_electrons) if i > 0 and count_i not in happy ]
    
    # The outer loop checks each bonding structure produced by the inner loop for consistency with
    # the user specified "pref_bonding" and pref_argument with bonding electrons are
    for dummy_counter in range(lewis_total):
        lewis_loop_list = loop_list
        random.shuffle(lewis_loop_list)
        outer_counter     = 0
        inner_max_cycles  = 1000
        outer_max_cycles  = 1000
        bond_sat = False
        
        lewis_lone_electrons.append(deepcopy(lone_electrons))
        lewis_bonding_electrons.append(deepcopy(bonding_electrons))
        lewis_core_electrons.append(deepcopy(core_electrons))
        lewis_valence.append(deepcopy(valence))
        lewis_bonding_target.append(deepcopy(bonding_target))
        lewis_bonds_made.append(deepcopy(bonds_made))
        lewis_adj_mat.append(deepcopy(adj_mat))
        lewis_counter = len(lewis_lone_electrons) - 1
        
        # Adjust the number of electrons by removing or adding to the available lone pairs
        # The algorithm simply adds/removes from the first N lone pairs that are discovered
        random.shuffle(octet_violate_e)
        random.shuffle(normal_adjust)
        adjust_ind=octet_violate_e+normal_adjust
    
        if len(adjust_ind) >= abs(q_tot): 
            for i in range(abs(q_tot)):
                lewis_lone_electrons[-1][adjust_ind[i]] += adjust
                lewis_bonding_target[-1][adjust_ind[i]] += adjust 
        else:
            for i in range(abs(q_tot)):
                lewis_lone_electrons[-1][0] += adjust
                lewis_bonding_target[-1][0] += adjust

        # Search for an optimal lewis structure
        while bond_sat is False:
        
            # Initialize necessary objects
            change_list   = range(len(lewis_lone_electrons[lewis_counter]))
            inner_counter = 0
            bond_sat = True                
            # Inner loop forms bonds to remove radicals or underbonded atoms until no further
            # changes in the bonding pattern are observed.
            while len(change_list) > 0:
                change_list = []
                for i in lewis_loop_list:

                    # List of atoms that already have a satisfactory binding configuration.
                    happy = [ j[0] for j in bonding_pref if j[1] <= lewis_bonding_electrons[lewis_counter][j[0]]]            
                    
                    # If the current atom already has its target configuration then no further action is taken
                    if i in happy: continue

                    # If there are no lone electrons or too more bond formed then skip
                    if lewis_lone_electrons[lewis_counter][i] == 0: continue
                    
                    # Take action if this atom has a radical or an unsatifisied bonding condition
                    if lewis_lone_electrons[lewis_counter][i] % 2 != 0 or lewis_bonding_electrons[lewis_counter][i] != lewis_bonding_target[lewis_counter][i]:
                        # Try to form a bond with a neighboring radical (valence +1/-1 check ensures that no improper 5-bonded atoms are formed)
                        lewis_bonded_radicals = [ (-find_lewis.en[elements[count_j].lower()],count_j) for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and lewis_lone_electrons[lewis_counter][count_j] % 2 != 0 \
                                                  and 2*(lewis_bonding_electrons[lewis_counter][count_j]+1)+(lewis_lone_electrons[lewis_counter][count_j]-1) <= lewis_valence[lewis_counter][count_j]\
                                                  and lewis_lone_electrons[lewis_counter][count_j]-1 >= 0 and count_j not in happy ]

                        lewis_bonded_lonepairs= [ (-find_lewis.en[elements[count_j].lower()],count_j) for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and lewis_lone_electrons[lewis_counter][count_j] > 0 \
                                                  and 2*(lewis_bonding_electrons[lewis_counter][count_j]+1)+(lewis_lone_electrons[lewis_counter][count_j]-1) <= lewis_valence[lewis_counter][count_j] and lewis_lone_electrons[lewis_counter][count_j]-1 >= 0 \
                                                  and count_j not in happy ]

                        # Sort by atomic number (cheap way of sorting carbon before other atoms, should probably switch over to electronegativities) 
                        lewis_bonded_radicals  = [ j[1] for j in sorted(lewis_bonded_radicals) ]
                        lewis_bonded_lonepairs = [ j[1] for j in sorted(lewis_bonded_lonepairs) ]

                        # Correcting radicals is attempted first
                        if len(lewis_bonded_radicals) > 0:
                            lewis_bonding_electrons[lewis_counter][i] += 1
                            lewis_bonding_electrons[lewis_counter][lewis_bonded_radicals[0]] += 1
                            lewis_adj_mat[lewis_counter][i][lewis_bonded_radicals[0]] += 1
                            lewis_adj_mat[lewis_counter][lewis_bonded_radicals[0]][i] += 1 
                            lewis_lone_electrons[lewis_counter][i] -= 1
                            lewis_lone_electrons[lewis_counter][lewis_bonded_radicals[0]] -= 1
                            change_list += [i,lewis_bonded_radicals[0]]
                            lewis_bonds_made[lewis_counter] += [(i,lewis_bonded_radicals[0])]
                                                        
                        # Else try to form a bond with a neighboring atom with spare lone electrons (valence check ensures that no improper 5-bonded atoms are formed)
                        elif len(lewis_bonded_lonepairs) > 0:
                            lewis_bonding_electrons[lewis_counter][i] += 1
                            lewis_bonding_electrons[lewis_counter][lewis_bonded_lonepairs[0]] += 1
                            lewis_adj_mat[lewis_counter][i][lewis_bonded_lonepairs[0]] += 1
                            lewis_adj_mat[lewis_counter][lewis_bonded_lonepairs[0]][i] += 1
                            lewis_lone_electrons[lewis_counter][i] -= 1
                            lewis_lone_electrons[lewis_counter][lewis_bonded_lonepairs[0]] -= 1
                            change_list += [i,lewis_bonded_lonepairs[0]]
                            lewis_bonds_made[lewis_counter] += [(i,lewis_bonded_lonepairs[0])]
                            #lewis_bonds_en[lewis_counter] += 1.0/find_lewis.en[elements[i].lower()]/find_lewis.en[elements[lewis_bonded_lonepairs[0]].lower()]
                
                # Increment the counter and break if the maximum number of attempts have been made
                inner_counter += 1
                if inner_counter >= inner_max_cycles:
                    print("WARNING: maximum attempts to establish a reasonable lewis-structure exceeded ({}).".format(inner_max_cycles))
            
            # Check if the user specified preferred bond order has been achieved.
            if bonding_pref is not None:
                unhappy = [ i[0] for i in bonding_pref if i[1] != lewis_bonding_electrons[lewis_counter][i[0]]]            
                if len(unhappy) > 0:

                    # Break the first bond involving one of the atoms bonded to the under/over coordinated atoms
                    ind = set([unhappy[0]] + [ count_i for count_i,i in enumerate(adj_mat_0[unhappy[0]]) if i == 1 and (count_i,unhappy[0]) not in off_limits ])
                    
                    # Check if a rearrangment is possible, break if none are available
                    try:
                        break_bond = next( i for i in lewis_bonds_made[lewis_counter] if i[0] in ind or i[1] in ind )
                    except:
                        print("WARNING: no further bond rearrangments are possible and bonding_pref is still not satisfied.")
                        break
                    
                    # Perform bond rearrangment
                    lewis_bonding_electrons[lewis_counter][break_bond[0]] -= 1
                    lewis_lone_electrons[lewis_counter][break_bond[0]] += 1
                    lewis_adj_mat[lewis_counter][break_bond[0]][break_bond[1]] -= 1
                    lewis_adj_mat[lewis_counter][break_bond[1]][break_bond[0]] -= 1
                    lewis_bonding_electrons[lewis_counter][break_bond[1]] -= 1
                    lewis_lone_electrons[lewis_counter][break_bond[1]] += 1

                    # Remove the bond from the list and reorder lewis_loop_list so that the indices involved in the bond are put last                
                    lewis_bonds_made[lewis_counter].remove(break_bond)
                    lewis_loop_list.remove(break_bond[0])
                    lewis_loop_list.remove(break_bond[1])
                    lewis_loop_list += [break_bond[0],break_bond[1]]
                                        
                    # Update the bond_sat flag
                    bond_sat = False
                    
                # Increment the counter and break if the maximum number of attempts have been made
                outer_counter += 1
                    
                # Periodically reorder the list to avoid some cyclical walks
                if outer_counter % 100 == 0:
                    lewis_loop_list = reorder_list(lewis_loop_list,atomic_number)

                # Print diagnostic upon failure
                if outer_counter >= outer_max_cycles:
                    print("WARNING: maximum attempts to establish a lewis-structure consistent")
                    print("         with the user supplied bonding preference has been exceeded ({}).".format(outer_max_cycles))
                    break
        
        # Re-apply keep_lone: remove one electron from such index    
        for count_i in keep_lone:
            lewis_lone_electrons[lewis_counter][count_i] -= 1

        # Special cases, share pair of electrons
        total_electron=np.array(lewis_lone_electrons[lewis_counter])+np.array(lewis_bonding_electrons[lewis_counter])*2
        
        # count for atom which doesn't satisfy
        # Notice: need systematical check for this part !!!
        unsatisfy = [count_t for count_t,te in enumerate(total_electron) if te > 2 and te < 8 and te % 2 ==0]
        for uns in unsatisfy:
            full_connect=[count_i for count_i,i in enumerate(adj_mat_0[uns]) if i == 1 and total_electron[count_i] == 8 and lewis_lone_electrons[lewis_counter][count_i] >= 2]
            if len(full_connect) > 0:                                                                                                                                                                                                                                          
                lewis_lone_electrons[lewis_counter][full_connect[0]]-=2                                                                                                                                                                                                        
                lewis_bonding_electrons[lewis_counter][uns]+=1                                                                                                                                                                                                                 
                lewis_bonding_electrons[lewis_counter][full_connect[0]]+=1                                                                                                                                                                                                     
                lewis_adj_mat[lewis_counter][uns][full_connect[0]]+=1                                                                                                                                                                                                          
                lewis_adj_mat[lewis_counter][full_connect[0]][uns]+=1 

        # Delete last entry in the lewis arrays if the electronic structure is not unique: introduce identical_mat includes both info of bond_mats and formal_charges
        identical_mat=np.vstack([lewis_adj_mat[-1], np.array([ valence_list[k] - lewis_bonding_electrons[-1][k] - lewis_lone_electrons[-1][k] for k in range(len(elements)) ]) ])
        lewis_identical_mat.append(identical_mat)
        
        if array_unique(lewis_identical_mat[-1],lewis_identical_mat[:-1]) is False :
            lewis_lone_electrons    = lewis_lone_electrons[:-1]
            lewis_bonding_electrons = lewis_bonding_electrons[:-1]
            lewis_core_electrons    = lewis_core_electrons[:-1]
            lewis_valence           = lewis_valence[:-1]
            lewis_bonding_target    = lewis_bonding_target[:-1]
            lewis_bonds_made        = lewis_bonds_made[:-1]
            lewis_adj_mat           = lewis_adj_mat[:-1]
            lewis_identical_mat     = lewis_identical_mat[:-1]
            
    # Find the total number of lone electrons in each structure
    lone_electrons_sums = []
    for i in range(len(lewis_lone_electrons)):
        lone_electrons_sums.append(sum(lewis_lone_electrons[i]))
        
    # Find octet violations in each structure
    octet_violations = []
    for i in range(len(lewis_lone_electrons)):
        ov = 0
        if octet_opt is True:
            for count_j,j in enumerate(elements):
                if j.lower() in ["c","n","o","f","si","p","s","cl","br","i"] and count_j not in bonding_pref_ind:
                    if lewis_bonding_electrons[i][count_j]*2 + lewis_lone_electrons[i][count_j] != 8 and lewis_bonding_electrons[i][count_j]*2 + lewis_lone_electrons[i][count_j] != 18:
                        ov += 1
        octet_violations.append(ov)

    ## Calculate bonding energy
    lewis_bonds_energy = []
    for bonds_made in lewis_bonds_made:
        for lb,bond_made in enumerate(bonds_made): bonds_made[lb]=tuple(sorted(bond_made))
        count_bonds_made = ["{}-{}-{}".format(min(atomic_number[bm[0]],atomic_number[bm[1]]),max(atomic_number[bm[0]],atomic_number[bm[1]]),bonds_made.count(bm) ) for bm in set(bonds_made)]
        lewis_bonds_energy += [sum([find_lewis.be[cbm] if cbm in find_lewis.be.keys() else -10000.0 for cbm in count_bonds_made  ]) ]
    # normalize the effect
    lewis_bonds_energy = [-be/max(1,max(lewis_bonds_energy)) for be in lewis_bonds_energy]

    ## Find the total formal charge for each structure
    formal_charges_sums = []
    for i in range(len(lewis_lone_electrons)):
        fc = 0
        for j in range(len(elements)):
            fc += valence_list[j] - lewis_bonding_electrons[i][j] - lewis_lone_electrons[i][j]
        formal_charges_sums.append(fc)
    
    ## Find formal charge eletronegativity contribution
    lewis_formal_charge = [ [ valence_list[i] - lewis_bonding_electrons[_][i] - lewis_lone_electrons[_][i] for i in range(len(elements)) ] for _ in range(len(lewis_lone_electrons)) ]
    lewis_keep_lone     = [ [ count_i for count_i,i in enumerate(lone) if i % 2 != 0] for lone in lewis_lone_electrons]
    lewis_fc_en = []  # Electronegativity for stabling charge/radical
    lewis_fc_pol = [] # Polarizability for stabling charge/radical
    lewis_fc_hc  = [] # Hyper-conjugation contribution
    for i in range(len(lewis_lone_electrons)):
        formal_charge = lewis_formal_charge[i]
        radical_atom = lewis_keep_lone[i]
        fc_ind = [(count_j,j) for count_j,j in enumerate(formal_charge) if j != 0]
        for R_ind in radical_atom:  # assign +0.5 for radical
            fc_ind += [(R_ind,0.5)]
        
        # initialize en,pol and hc
        fc_en,fc_pol,fc_hc = 0,0,0
        
        # Loop over formal charges and radicals
        for count_fc in fc_ind:
            ind = count_fc[0]    
            charge = count_fc[1]
            # Count the self contribution: (-) on the most electronegative atom and (+) on the least electronegative atom
            fc_en += 10 * charge * find_lewis.en[elements[ind].lower()]
             
            # Find the nearest and next-nearest atoms for each formal_charge/radical contained atom
            gs = graph_seps(adj_mat_0)
            nearest_atoms = [count_k for count_k,k in enumerate(lewis_adj_mat[i][ind]) if k >= 1] 
            NN_atoms = list(set([ count_j for count_j,j in enumerate(gs[ind]) if j == 2 ]))
            
            # only count when en > en(C)
            fc_en += charge*(sum([find_lewis.en[elements[count_k].lower()] for count_k in nearest_atoms if find_lewis.en[elements[count_k].lower()] > 2.54] )+\
                             sum([find_lewis.en[elements[count_k].lower()] for count_k in NN_atoms if find_lewis.en[elements[count_k].lower()] > 2.54] ) * 0.1 )

            if charge < 0: # Polarizability only affects negative charge
                fc_pol += charge*sum([find_lewis.pol[elements[count_k].lower()] for count_k in nearest_atoms ])

            # find hyper-conjugation strcuture
            nearby_carbon = [nind for nind in nearest_atoms if elements[nind].lower()=='c']
            for carbon_ind in nearby_carbon:
                carbon_nearby=[nind for nind in NN_atoms if lewis_adj_mat[i][carbon_ind][nind] >= 1 and elements[nind].lower() in ['c','h']]
                if len(carbon_nearby) == 3: fc_hc -= charge*(len([nind for nind in carbon_nearby if elements[nind].lower() == 'c'])*2 + len([nind for nind in carbon_nearby if elements[nind].lower() == 'h']))

        lewis_fc_en.append(fc_en)        
        lewis_fc_pol.append(fc_pol)        
        lewis_fc_hc.append(fc_hc)        

    # normalize the effect
    lewis_fc_en = [lfc/max(1,max(abs(np.array(lewis_fc_en)))) for lfc in lewis_fc_en]
    lewis_fc_pol= [lfp/max(1,max(abs(np.array(lewis_fc_pol)))) for lfp in lewis_fc_pol]
    
    # Add the total number of radicals to the total formal charge to determine the criteria.
    # The radical count is scaled by 0.01 and the lone pair count is scaled by 0.001. This results
    # in the structure with the lowest formal charge always being returned, and the radical count 
    # only being considered if structures with equivalent formal charges are found, and likewise with
    # the lone pair count. The structure(s) with the lowest score will be returned.
    lewis_criteria = []
    for i in range(len(lewis_lone_electrons)):
        #lewis_criteria.append( 10.0*octet_violations[i] + abs(formal_charges_sums[i]) + 0.1*sum([ 1 for j in lewis_lone_electrons[i] if j % 2 != 0 ]) + 0.001*lewis_bonds_energy[i]  + 0.00001*lewis_fc_en[i] + 0.000001*lewis_fc_pol[i] + 0.0000001*lewis_fc_hc[i]) 
        lewis_criteria.append( 10.0*octet_violations[i] + abs(formal_charges_sums[i]) + 0.1*sum([ 1 for j in lewis_lone_electrons[i] if j % 2 != 0 ]) + 0.01*lewis_fc_en[i] + 0.005*lewis_fc_pol[i] + 0.0001*lewis_fc_hc[i] + 0.0001*lewis_bonds_energy[i]) 

    best_lewis = [i[0] for i in sorted(enumerate(lewis_criteria), key=lambda x:x[1])]  # sort from least to most and return a list containing the origial list's indices in the correct order
    best_lewis = [ i for i in best_lewis if lewis_criteria[i] == lewis_criteria[best_lewis[0]] ]    

    # Finally check formal charge to keep those with 
    lewis_re_fc     = [ lewis_formal_charge[_]+lewis_keep_lone[_] for _ in best_lewis]
    appear_times    = [ lewis_re_fc.count(i) for i in lewis_re_fc]
    best_lewis      = [best_lewis[i] for i in range(len(lewis_re_fc)) if appear_times[i] == max(appear_times) ] 
    
    # Apply keep_lone information, remove the electron to form lone electron
    for i in best_lewis:
        for j in keep_lone:
            lewis_lone_electrons[i][j] -= 1

    # Print diagnostics
    if verbose is True:
        for i in best_lewis:
            print("Bonding Matrix  {}".format(i))
            print("Formal_charge:  {}".format(formal_charges_sums[i]))
            print("Lewis_criteria: {}\n".format(lewis_criteria[i]))
            print("{:<40s} {:<40s} {:<15s} {:<15s}".format("Elements","Bond_Mat","Lone_Electrons","FC"))
            for j in range(len(elements)):
                print("{:<40s} {}    {} {}".format(elements[j]," ".join([ str(k) for k in lewis_adj_mat[i][j] ]),lewis_lone_electrons[i][j],valence_list[j] - lewis_bonding_electrons[i][j] - lewis_lone_electrons[i][j]))
            print (" ")

    # If only the bonding matrix is requested, then only that is returned
    if b_mat_only is True:
        if return_FC is False:  
            return [ lewis_adj_mat[_] for _ in best_lewis ]
        else:
            return [ lewis_adj_mat[_] for _ in best_lewis ], [ lewis_formal_charge[_] for _ in best_lewis ]

    # return like check_lewis function
    if check_lewis_flag is True:
        if return_pref is True:
            return lewis_lone_electrons[best_lewis[0]], lewis_bonding_electrons[best_lewis[0]], lewis_core_electrons[best_lewis[0]],bonding_pref
        else:
            return lewis_lone_electrons[best_lewis[0]], lewis_bonding_electrons[best_lewis[0]], lewis_core_electrons[best_lewis[0]]
    
    # Optional bonding pref return to handle cases with special groups
    if return_pref is True:
        if return_FC is False:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ],bonding_pref
        else:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ],[ lewis_formal_charge[_] for _ in best_lewis ],bonding_pref 

    else:
        if return_FC is False:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ]
        else:
            return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
                   [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ],[ lewis_formal_charge[_] for _ in best_lewis ]


# Description: Function to determine whether given atom index in the input sturcture locates on a ring or not
#
# Inputs      adj_mat:   NxN array holding the molecular graph
#             idx:       atom index
#             ring_size: number of atoms in a ring
#
# Returns     Bool value depending on if idx is a ring atom 
#
def ring_atom(adj_mat,idx,start=None,ring_size=10,counter=0,avoid_set=None,in_ring=None):

    # Consistency/Termination checks
    if ring_size < 3:
        print("ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")
    if counter == ring_size:
        return False,[]

    # Automatically assign start to the supplied idx value. For recursive calls this is set manually
    if start is None:
        start = idx
    if avoid_set is None:
        avoid_set = set([])
    if in_ring is None:
        in_ring=set([idx])

    # Trick: The fact that the smallest possible ring has three nodes can be used to simplify
    #        the algorithm by including the origin in avoid_set until after the second step
    if counter >= 2 and start in avoid_set:
        avoid_set.remove(start)    
    elif counter < 2 and start not in avoid_set:
        avoid_set.add(start)

    # Update the avoid_set with the current idx value
    avoid_set.add(idx)    
    
    # Loop over connections and recursively search for idx
    status = 0
    cons = [ count_i for count_i,i in enumerate(adj_mat[idx]) if i == 1 and count_i not in avoid_set ]
    
    if len(cons) == 0:
        return False,[]
    elif start in cons:
        return True,in_ring
    else:
        for i in cons:
            if ring_atom(adj_mat,i,start=start,ring_size=ring_size,counter=counter+1,avoid_set=avoid_set,in_ring=in_ring)[0] == True:
                in_ring.add(i)
                return True,in_ring
        return False,[]


# Description: Canonicalizes the ordering of atoms in a geometry based on a hash function. Atoms that hash to equivalent values retain their relative order from the input geometry.
#
# Inputs:     elements:  a list of element labels indexed to the adj_mat 
#             adj_mat:   np.array of atomic connections
#
# Optional:   geo:       np.array of geometry
#             bond_mat:  np.array of bonding information
#
# Returns     Sorted inputs
#
def canon_geo(elements,adj_mat,geo=None,bond_mat=None,dup=[],change_group_seq=True):
    
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

    # determine the seperate compounds
    gs = graph_seps(adj_mat)
    groups = []
    loop_ind = []
    for i in range(len(gs)):
        if i not in loop_ind:
            new_group = [count_j for count_j,j in enumerate(gs[i,:]) if j >= 0]
            loop_ind += new_group
            groups   += [new_group]

    # sort groups based on the maximum hash value
    if change_group_seq:
        _,group_seq = [list(k) for k in list(zip(*sorted([ (max([hash_list[j] for j in group]),lg) for lg,group in enumerate(groups) ], reverse=True)))]
        groups = [groups[i] for i in group_seq]

    # sort atoms in each group
    atoms = []
    for group in groups:
        _,seq  =  [ list(j) for j in list(zip(*sorted([ (hash_list[i],i) for i in group ],reverse=True)) )]
        atoms += seq
    
    # Update lists/arrays based on atoms
    adj_mat   = adj_mat[atoms]
    adj_mat   = adj_mat[:,atoms]
    elements  = [ elements[i] for i in atoms ]
    hash_list = [ hash_list[j] for j in atoms ]
              
    if geo is not None:
        geo   = geo[atoms]

    if bond_mat is not None:
        if len(bond_mat) == len(elements):
            bond_mat = bond_mat[atoms]
            bond_mat = bond_mat[:,atoms]
        else:
            for i in range(len(bond_mat)):
                bond_mat[i] = bond_mat[i][atoms] 
                bond_mat[i] = bond_mat[i][:,atoms]
    
    # Duplicate the respective lists
    if len(dup) > 0:
        N_dup = {}
        for count_i,i in enumerate(dup):
            N_dup[count_i] = []
            for j in atoms:
                N_dup[count_i] += [i[j]]
        N_dup = [ N_dup[i] for i in range(len(N_dup.keys())) ]

        if bond_mat is not None and geo is not None:
            return elements,adj_mat,hash_list,geo,bond_mat,N_dup
        elif bond_mat is None and geo is not None:
            return elements,adj_mat,hash_list,geo,N_dup
        elif bond_mat is not None and geo is None:
            return elements,adj_mat,hash_list,bond_mat,N_dup
        else:
            return elements,adj_mat,hash_list,N_dup
    else:
        if bond_mat is not None and geo is not None:
            return elements,adj_mat,hash_list,geo,bond_mat
        elif bond_mat is None and geo is not None:
            return elements,adj_mat,hash_list,geo
        elif bond_mat is not None and geo is None:
            return elements,adj_mat,hash_list,bond_mat
        else:
            return elements,adj_mat,hash_list


# Description: Hashing function for canonicalizing geometries on the basis of their adjacency matrices and elements
#
# Inputs      ind  : index of the atom being hashed
#             A    : adjacency matrix
#             M    : masses of the atoms in the molecule
#             gens : depth of the search used for the hash   
#
# Returns     hash value of given atom
#
def atom_hash(ind,A,M,alpha=100.0,beta=0.1,gens=10):    
    if gens <= 0:
        return rec_sum(ind,A,M,beta,gens=0)
    else:
        return alpha * sum(A[ind]) + rec_sum(ind,A,M,beta,gens)

# recursive function for summing up the masses at each generation of connections. 
def rec_sum(ind,A,M,beta,gens,avoid_list=[]):
    if gens != 0:
        tmp = M[ind]*beta
        new = [ count_j for count_j,j in enumerate(A[ind]) if j == 1 and count_j not in avoid_list ]
        if len(new) > 0:
            for i in new:
                tmp += rec_sum(i,A,M,beta*0.1,gens-1,avoid_list=avoid_list+[ind])
            return tmp
        else:
            return tmp
    else:
        return M[ind]*beta

# Return bool depending on if the atom is a nitro nitrogen atom
def is_nitro(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ] 
    if len(O_ind) == 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfoxide sulfur atom
def is_sulfoxide(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ] 
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] ] 
    if len(O_ind) == 1 and len(C_ind) == 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfonyl sulfur atom
def is_sulfonyl(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ] 
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] ] 
    if len(O_ind) == 2 and len(C_ind) == 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a phosphate phosphorus atom
def is_phosphate(i,adj_mat,elements):

    status = False
    if elements[i] not in ["P","p"]:
        return False
    O_ind      = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] ] 
    O_ind_term = [ j for j in O_ind if sum(adj_mat[j]) == 1 ]
    if len(O_ind) == 4 and sum(adj_mat[i]) == 4 and len(O_ind_term) > 0:
        return True
    else:
        return False

# Return bool depending on if the atom is a cyano nitrogen atom
def is_cyano(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"] or sum(adj_mat[i]) > 1:
        return False
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] and sum(adj_mat[count_j]) == 2 ]
    if len(C_ind) == 1:
        return True
    else:
        return False

# Return bool depending on if the atom is a cyano nitrogen atom
def is_isocyano(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"] or sum(adj_mat[i]) > 1:
        return False
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] and sum(adj_mat[count_j]) == 1 ]
    if len(C_ind) == 1:
        return True
    else:
        return False

# Returns a matrix of graphical separations for all nodes in a graph defined by the inputted adjacency matrix 
def graph_seps(adj_mat_0):

    # Create a new name for the object holding A**(N), initialized with A**(1)
    adj_mat = deepcopy(adj_mat_0)
    
    # Initialize an array to hold the graphical separations with -1 for all unassigned elements and 0 for the diagonal.
    seps = np.ones([len(adj_mat),len(adj_mat)])*-1
    np.fill_diagonal(seps,0)

    # Perform searches out to len(adj_mat) bonds (maximum distance for a graph with len(adj_mat) nodes
    for i in np.arange(len(adj_mat)):        

        # All perform assignments to unassigned elements (seps==-1) 
        # and all perform an assignment if the value in the adj_mat is > 0        
        seps[np.where((seps==-1)&(adj_mat>0))] = i+1

        # Since we only care about the leading edge of the search and not the actual number of paths at higher orders, we can 
        # set the larger than 1 values to 1. This ensures numerical stability for larger adjacency matrices.
        adj_mat[np.where(adj_mat>1)] = 1
        
        # Break once all of the elements have been assigned
        if -1 not in seps:
            break

        # Take the inner product of the A**(i+1) with A**(1)
        adj_mat = np.dot(adj_mat,adj_mat_0)

    return seps

# Description: This function calls obminimize (open babel geometry optimizer function) to optimize the current geometry
#
# Inputs:      geo:      Nx3 array of atomic coordinates
#              adj_mat:  NxN array of connections
#              elements: N list of element labels
#              ff:       force-field specification passed to obminimize (uff, gaff)
#               q:       total charge on the molecule   
#
# Returns:     geo:      Nx3 array of optimized atomic coordinates
# 
def opt_geo(geo,adj_mat,elements,q=0,ff='uff',step=100):

    # Write a temporary molfile for obminimize to use
    tmp_filename = '.tmp.mol'
    count = 0
    while os.path.isfile(tmp_filename):
        count += 1
        if count == 10:
            print("ERROR in opt_geo: could not find a suitable filename for the tmp geometry. Exiting...")
            return geo
        else:
            tmp_filename = ".tmp" + tmp_filename            

    # Use the mol_write function imported from the write_functions.py 
    # to write the current geometry and topology to file
    mol_write(tmp_filename,elements,geo,adj_mat,q=q,append_opt=False)

    substring = 'obabel {} -O result.xyz --sd --minimize --steps {} --ff {}'.format(tmp_filename,step,ff)
    output = subprocess.Popen(substring, shell=True, stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.PIPE,bufsize=-1).communicate()[0]
    element,geo = xyz_parse("result.xyz") 

    # Remove the tmp file that was read by obminimize
    try:
        os.remove(tmp_filename)
        os.remove("result.xyz")

    except:
        pass

    return geo

# Description: Simple wrapper function for writing xyz file
#
# Inputs      name:     string holding the filename of the output
#             elements: list of element types (list of strings)
#             geo:      Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#
# Returns     None
#
def xyz_write(name,elements,geo,append_opt=False,comment=''):

    if append_opt == True:
        open_cond = 'a'
    else:
        open_cond = 'w'
        
    with open(name,open_cond) as f:
        f.write('{}\n'.format(len(elements)))
        f.write('{}\n'.format(comment))
        for count_i,i in enumerate(elements):
            f.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(i,geo[count_i][0],geo[count_i][1],geo[count_i][2]))
    return 

# Description: Simple wrapper function for writing a mol (V2000) file
#
# Inputs      name:     string holding the filename of the output
#             elements: list of element types (list of strings)
#             geo:      Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#             adj_mat:  NxN array holding the molecular graph
#
# Returns     None
#
def mol_write(name,elements,geo,adj_mat,q=0,append_opt=False):

    # Consistency check
    if len(elements) >= 1000:
        print("ERROR in mol_write: the V2000 format can only accomodate up to 1000 atoms per molecule.")
        return 

    # Check for append vs overwrite condition
    if append_opt == True:
        open_cond = 'a'
    else:
        open_cond = 'w'

    if q == 0:
        # Get the bond orders
        bond_mat = find_lewis(elements,adj_mat, q_tot=q,b_mat_only=True,verbose=False)
        
    else:
        # Get the bond orders
        bond_mat,FC = find_lewis(elements,adj_mat, q_tot=q,b_mat_only=True,verbose=False,return_FC=True)
    
    # Parse the basename for the mol header
    base_name = name.split(".")
    if len(base_name) > 1:
        base_name = ".".join(base_name[:-1])
    else:
        base_name = base_name[0]

    # Write the file
    with open(name,open_cond) as f:

        # Write the header
        f.write('{}\nGenerated by mol_write.py\n\n'.format(base_name))

        # Write the number of atoms and bonds
        f.write("{:>3d}{:>3d}  0  0  0  0  0  0  0  0  1 V2000\n".format(len(elements),int(sum(sum(adj_mat/2.0)))))

        # Write the geometry
        for count_i,i in enumerate(elements):
            f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n".format(geo[count_i][0],geo[count_i][1],geo[count_i][2],i))

        # Write the bonds
        bonds = [ (count_i,count_j) for count_i,i in enumerate(adj_mat) for count_j,j in enumerate(i) if j == 1 and count_j > count_i ] 
        for i in bonds:

            # Calculate bond order from the bond_mat
            bond_order = int(bond_mat[0][i[0],i[1]])

            f.write("{:>3d}{:>3d}{:>3d}  0  0  0  0\n".format(i[0]+1,i[1]+1,bond_order))
        f.write("M  END\n$$$$\n")

    return 

# Description: Simple wrapper function for grabbing the coordinates and
#              elements from an xyz file
#
# Inputs      input: string holding the filename of the xyz
# Returns     Elements: list of element types (list of strings)
#             Geometry: Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#
def xyz_parse(input,read_types=False):

    # Commands for reading only the coordinates and the elements
    if read_types is False:
        
        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print("ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count]=fields[0]
                            Geometry[count,:]=np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))

        return Elements,Geometry

    # Commands for reading the atomtypes from the fourth column
    if read_types is True:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        Atom_types = [None]*N_atoms
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print("ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count]=fields[0]
                            Geometry[count,:]=np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                            if len(fields) > 4:
                                Atom_types[count] = fields[4]
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))

        return Elements,Geometry,Atom_types

# Description: Parses the molecular charge from the comment line of the xyz file if present
#
# Inputs       input: string holding the filename of the xyz file. 
# Returns      q:     int or None
#
def parse_q(xyz):

    with open(xyz,'r') as f:
        for lc,lines in enumerate(f):
            if lc == 1:
                fields = lines.split()
                if "q" in fields:
                    q = int(float(fields[fields.index("q")+1]))
                else:
                    q = 0
                break
    return q

# Description: Checks if an array "a" is unique compared with a list of arrays "a_list"
#              at the first match False is returned.
def array_unique(a,a_list):
    for ind,i in enumerate(a_list):
        if np.array_equal(a,i):
            return False,ind
    return True,0
