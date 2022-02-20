import sys,argparse,os,subprocess,shutil,fnmatch
import pickle,json
from itertools import combinations
# import taffi related functions
from taffi_functions import * 
from xtb_functions import *
from ase import io
from ase.build import minimize_rotation_and_translation

# Function to Return smiles string 
def return_smi(E,G,adj_mat=None,namespace='obabel'):
    if adj_mat is None:
        xyz_write("{}_input.xyz".format(namespace),E,G)
        substring = "obabel -ixyz {}_input.xyz -ocan".format(namespace)
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        smile  = output.split()[0]
        os.system("rm {}_input.xyz".format(namespace))
    
    else:
        mol_write("{}_input.mol".format(namespace),E,G,adj_mat)
        substring = "obabel -imol {}_input.mol -ocan".format(namespace)
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        smile  = output.split()[0]
        os.system("rm {}_input.mol".format(namespace))

    return smile

# Function to Return smiles string 
def return_inchikey(E,G,adj_mat=None,separate=False,namespace='obabel'):

    if adj_mat is None:
        adj_mat = Table_generator(E, G)

    if separate:

        # Seperate reactant(s)
        gs      = graph_seps(adj_mat)
        groups  = []
        loop_ind= []
        for i in range(len(gs)):
            if i not in loop_ind:
                new_group =[count_j for count_j,j in enumerate(gs[i,:]) if j >= 0]
                loop_ind += new_group
                groups   +=[new_group]
            
        # Determine the inchikey of all components in the reactant
        inchi_list = []
        for group in groups:
            N_atom = len(group)
            frag_E = [E[ind] for ind in group]
            frag_adj = adj_mat[group][:,group]
            frag_G = np.zeros([N_atom,3])
            for count_i,i in enumerate(group):
                frag_G[count_i,:] = G[i,:]

            # generate inchikeu
            mol_write("{}_input.mol".format(namespace),frag_E,frag_G,frag_adj)
            substring = "obabel -imol {}_input.mol -oinchikey".format(namespace)
            output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
            inchi  = output.split()[0]
            inchi_list+= [inchi]
        
        os.system("rm {}_input.mol".format(namespace))
        if len(groups) == 1:
            return inchi_list[0]
        else:
            return '-'.join(sorted([i[:14] for i in inchi_list]))

    else:
        mol_write("{}_input.mol".format(namespace),E,G,adj_mat)
        substring = "obabel -imol {}_input.mol -oinchikey".format(namespace)
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        inchi  = output.split()[0]
        os.system("rm {}_input.mol".format(namespace))

    return inchi

# Function to Return atom index mapped smiles string 
def return_atommaped_smi(E,G,adj_mat=None,namespace='obabel'):
    
    from rdkit import Chem
    # generate adj_mat if is not provided
    if adj_mat is None: adj_mat = Table_generator(E, G)
    # write mol file
    mol_write("{}_input.mol".format(namespace),E,G,adj_mat)
    # convert mol file into rdkit mol onject
    mol=Chem.rdmolfiles.MolFromMolFile('{}_input.mol'.format(namespace),removeHs=False)
    # assign atom index
    mol=mol_with_atom_index(mol)

    return Chem.MolToSmiles(mol)

# function to assign atom index
def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms): mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol

# Function that take smile string and return element and geometry
def parse_smiles(smiles,ff='mmff94',steps=100):
    
    # load in rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        # construct rdkir object
        m = Chem.MolFromSmiles(smiles)
        m2= Chem.AddHs(m)
        AllChem.EmbedMolecule(m2)
        q = 0

        # parse mol file and obtain E & G
        lines = Chem.MolToMolBlock(m2)

        # create a temporary molfile
        tmp_filename = '.tmp.mol'
        with open(tmp_filename,'w') as g: g.write(lines)

        # apply force-field optimization
        try:
            command = 'obabel {} -O result.xyz --sd --minimize --steps {} --ff {}'.format(tmp_filename,steps,ff)
            output = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,bufsize=-1).communicate()[1].decode('utf8')
            E,G = xyz_parse("result.xyz")

        except:
            command = 'obabel {} -O result.xyz --sd --minimize --steps {} --ff uff'.format(tmp_filename,steps)
            output = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,bufsize=-1).communicate()[1].decode('utf8')
            E,G = xyz_parse("result.xyz")

        # Remove the tmp file that was read by obminimize
        try:
            os.remove(tmp_filename)
            os.remove("result.xyz")

        except:
            pass

        return True,E,G,q

    except: 

        return False,[],[],0

# Function to calcualte RMSD (Root-mean-square-displacement)
def return_RMSD(E,G1,G2,rotate=True,mass_weighted=False,namespace='node'):

    # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
    mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                 'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                 'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                 'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                 'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                 'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                 'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                 'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    if rotate:

        from ase import io
        from ase.build import minimize_rotation_and_translation

        # write two xyz file 
        xyz_write('{}1.xyz'.format(namespace),E,G1)
        xyz_write('{}2.xyz'.format(namespace),E,G2)
        node1 = io.read('{}1.xyz'.format(namespace))
        node2 = io.read('{}2.xyz'.format(namespace))
        minimize_rotation_and_translation(node1,node2)
        io.write('{}2.xyz'.format(namespace),node2)
    
        # reload node 2 geometry and compute RMSD
        _,G2  = xyz_parse('{}2.xyz'.format(namespace))
    
        try:
            os.remove('{}1.xyz'.format(namespace))
            os.remove('{}2.xyz'.format(namespace))
        except:
            pass

    # compute RMSD
    DG = G1 - G2
    RMSD = 0
    if mass_weighted:
        for i in range(len(E)):
            RMSD += sum(DG[i]**2)*mass_dict[E[i]]

        return np.sqrt(RMSD / sum([mass_dict[Ei] for Ei in E]))

    else:
        for i in range(len(E)):
            RMSD += sum(DG[i]**2)

        return np.sqrt(RMSD / len(E))

# Function to calculate change of distance between interested atoms
def return_Dis(E,G1,G2,involve,normal_bonds): 
   
    # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
    mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                 'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                 'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                 'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                 'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                 'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                 'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                 'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    # initialize dis, weighted dis and bond_dis
    Ei = [E[ind] for ind in involve]
    
    G1i= G1[involve,:]
    G2i= G2[involve,:]
    RMSDi = return_RMSD(Ei,G1i,G2i,rotate=True,mass_weighted=True)

    BDis = 0
    for bind in normal_bonds:
        Dis1  = np.sqrt(sum((G1[bind[0]]-G1[bind[1]])**2))
        Dis2  = np.sqrt(sum((G2[bind[0]]-G2[bind[1]])**2))
        if abs(Dis1-Dis2) > 0.2:
            BDis += np.abs(Dis1-Dis2)

    return RMSDi,BDis

# Function to analyze frequency mode
def analyze_TS(E,Radj_mat,Padj_mat,TS_out,package='Gaussian'):

    # Initialize UFF bond radii (Rappe et al. JACS 1992)
    Radii = {  'H':0.39, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.15,  'Si':1.050,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.473, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    # Initialze mass dict
    mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                 'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                 'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                 'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                 'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                 'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                 'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                 'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    # calculate bond changes and the atoms involved in the reaction
    bond_break,bond_form,TS_G,count_xyz = [],[],[],[]
    adj_change = Padj_mat - Radj_mat
    for i in range(len(E)):
        for j in range(i+1,len(E)):
            if adj_change[i][j] == -1:
                bond_break += [(i,j)]
                
            if adj_change[i][j] == 1:
                bond_form += [(i,j)]

    center_atoms = list(set(sum(bond_break, ())+sum(bond_form, ())))
    
    # analyze TS optimization output files and imag frequence
    imag_flag = False
    imag_mode = []
    if package.lower() == 'gaussian':
        with open(TS_out,'r') as f:
            for lc,lines in enumerate(f):
                fields = lines.split()
                # identify whether there are imaginary frequency in the out put file.
                if 'imaginary' in fields and 'frequencies' in fields and '(negative' in fields:
                    imag_flag = True
                    image_number = int(fields[1])

                if len(fields) == 2 and fields[0] == "Standard" and fields[1] == "orientation:": 
                    count_xyz += [lc]

                if imag_flag is True and len(fields)==5 and fields[0] == 'Frequencies' and float(fields[2]) < 0:
                    imag_lc = lc

        if imag_flag is False or image_number > 1:
            print("Check IRC and TS output files for {}, shouldn't contain 0/more than 1 imaginary frequency.".format(TS_out))
            quit()

        with open(TS_out,'r') as f:
            for lc,lines in enumerate(f):
                if lc >= count_xyz[-1] + 5 and lc < count_xyz[-1]+5 + len(E):
                    fields = lines.split()
                    TS_G  += [[float(fields[3]),float(fields[4]),float(fields[5])]]

                if lc >= imag_lc+5 and lc < imag_lc+5+len(E):
                    fields    = lines.split()
                    imag_mode+= [[float(fields[2]),float(fields[3]),float(fields[4])]]
        imag_mode = np.array(imag_mode)

    elif package.lower() == 'orca':

        # determine the frequence and normal mode lines
        geo_lines,freq_lines, mode_lines, imag_ind = [],[],[],[]
        with open(TS_out,'r') as f: TSlines = f.readlines()
        for lc,lines in enumerate(TSlines):
            if 'VIBRATIONAL FREQUENCIES' in lines: freq_lines += [lc]
            if 'NORMAL MODES' in lines: mode_lines += [lc]
            if 'CARTESIAN COORDINATES (ANGSTROEM)' in lines: geo_lines += [lc]
        geo_line,freq_line,mode_line = geo_lines[-1],freq_lines[-1],mode_lines[-1]
        
        # find imaginary frequency 
        for lc in range(freq_line,freq_line+len(E)*3+5):
            if 'cm**-1' not in TSlines[lc]: continue
            fields = TSlines[lc].split()
            if '-' in fields[1]: imag_ind += [int(fields[0].split(':')[0])]
            if len(imag_ind) > 2: break
        
        if len(imag_ind) != 1:
            print("Check TS output filrs for {}, shouldn't contain 0/more than 1 imaginary frequency.".format(TS_out))
            quit()

        # parse geometry
        TS_G = []
        for lc in range(geo_line+1,geo_line+5+len(E)):
            fields = TSlines[lc].split()
            if len(fields) != 4: continue
            TS_G  += [[float(fields[1]),float(fields[2]),float(fields[3])]]

        # parse imaginary mode
        imag_mode = []
        for lc in range(mode_line+6,len(TSlines)):
            if str(imag_ind[0]) not in TSlines[lc]: continue
            fields = TSlines[lc].split()
            if len(fields) == 6:
                start_line = lc+1
                position = fields.index(str(imag_ind[0]))+1
                break

        for lc in range(start_line,start_line+len(E)*3):
            fields = TSlines[lc].split()
            imag_mode += [float(fields[position])]
            
        # reshape and normalize imag_mode
        # first time massi**0.5 to convert to normal displacement
        imag_mode = np.array(imag_mode)
        imag_mode = imag_mode.reshape((len(E),3))
        
    # initialize
    TS_G = np.array(TS_G)
    TS_f = TS_G + imag_mode * 0.5
    TS_b = TS_G - imag_mode * 0.5

    # compute the maximum distance between the breaking and forming bonds
    bonds = bond_break + bond_form
    dist = 0
    for bond in bonds:
        dist = max(dist,np.linalg.norm(TS_G[bond[0]]-TS_G[bond[1]]))

    # analyze possible "bonds" in TS geometry
    P_bonds = []

    # Generate distance matrix holding atom-atom separations (only save upper right)
    Dist_Mat = np.triu(cdist(TS_G,TS_G))

    # Find plausible connections
    x_ind,y_ind = np.where( (Dist_Mat > 0.0) & (Dist_Mat < max([ Radii[i]**2.0 for i in Radii.keys() ])) )

    # Iterate over plausible connections and determine actual connections
    for count,i in enumerate(x_ind):
        if Dist_Mat[i,y_ind[count]] < (Radii[E[i]]+Radii[E[y_ind[count]]])*1.8:
            P_bonds += [(i,y_ind[count])]

    # find the most activate atom
    movement = [(i,np.linalg.norm(imag_mode[i])) for i in range(len(E))] 
    dis_change = []

    # loop over each pair to generate movement element
    for bond in P_bonds:
        dis = abs(np.linalg.norm(TS_f[bond[0]]-TS_f[bond[1]]) - np.linalg.norm(TS_b[bond[0]]-TS_b[bond[1]]))
        dis_change += [((bond[0],bond[1]),dis)]

    dis_change.sort(key = lambda x: -x[-1])

    dis_ind= [ind[0] for ind in dis_change]
    index  = []
    target = []
    for pair in bond_break+bond_form:
        if tuple(sorted(pair)) in dis_ind:
            index += [dis_ind.index(tuple(sorted(pair)))]
            target+= [dis_change[index[-1]][-1]]
        else:
            index += [len(dis_ind)]
            target+= [0]

    # return sum of unexpect bond changes
    freq_unexpect = sum([ind[-1] for count_i,ind in enumerate(dis_change[:max(index)]) if count_i not in index])

    return np.mean(target),min(target),freq_unexpect,dist

# Function to parse pyGSM input files
def parse_input(input_xyz,return_adj=False):

    name = input_xyz.split('/')[-1].split('xyz')[0]
    xyz  = ['','']
    count= 0

    # read in pairs of xyz file
    with open(input_xyz,"r") as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if lc == 0: 
                N = int(fields[0])
                xyz[0] += lines
                continue

            if len(fields) == 1 and float(fields[0]) == float(N):
                count+=1

            xyz[count]+=lines

    with open('{}_reactant.xyz'.format(name),"w") as f:
        f.write(xyz[0])

    with open('{}_product.xyz'.format(name),"w") as f:
        f.write(xyz[1])

    # parse reactant info
    E,RG   = xyz_parse('{}_reactant.xyz'.format(name))

    # parse product info
    _,PG   = xyz_parse('{}_product.xyz'.format(name))
                
    try:
        os.remove('{}_reactant.xyz'.format(name))
        os.remove('{}_product.xyz'.format(name))
    except:
        pass

    if return_adj:
        # generate adj_mat if is needed
        Radj_mat = Table_generator(E, RG)
        Padj_mat = Table_generator(E, PG)
        return E,RG,PG,Radj_mat,Padj_mat

    else:
        return E,RG,PG

# Function to identify whether two reactants far away from each other in a bi-molecular cases
# Also works for more than two fragments, remove the cases where all pairs of fragments are far away from each other
def check_bimolecule(adj,geo,factor='auto'):

    # Seperate molecules(s)
    gs      = graph_seps(adj)
    groups  = []
    loop_ind= []
    for i in range(len(gs)):
        if i not in loop_ind:
            new_group = [count_j for count_j,j in enumerate(gs[i,:]) if j >= 0]
            loop_ind += new_group
            groups   += [new_group]
        
    # if only one fragment, return True 
    if len(groups) == 1: return True

    # compute center of mass
    centers = []
    radius  = []
    for group in groups:
        center = np.array([0.0, 0.0, 0.0])
        for i in group:
            center += geo[i,:]/float(len(group))

        centers += [center]
        radius  += [max([ np.linalg.norm(geo[i,:]-center) for i in group])]

    # iterate over all paris of centers
    combs = combinations(range(len(centers)), 2)
    max_dis = 0
    satisfy = []
    
    if factor == 'auto':
        if len(adj) > 12: factor = 1.5
        else: factor = 2.0

        if min([len(j) for j in groups]) < 3: factor = 4.0
        elif min([len(j) for j in groups]) < 5: factor = 3.0

    for comb in combs:
        dis = np.linalg.norm(centers[comb[0]]-centers[comb[1]])
        if dis > factor * (radius[comb[0]]+radius[comb[1]]):
            satisfy += [False]
        else:
            satisfy += [True]

    return (False not in satisfy)

# Function that take smile string and return element and geometry
def count_radicals(smiles,ff='mmff94',steps=100):
    
    # load in rdkik
    from rdkit import Chem
    from rdkit.Chem.Descriptors import NumRadicalElectrons

    # count radicals
    Nr = 0
    for i in smiles.split('.'):
        try:
            Nr+=NumRadicalElectrons(Chem.MolFromSmiles(i))
        except:
            Nr+=1

    return Nr

# Function to take in prepared pyGSM pairs
def parse_pyGSM_inputs(final_products,E_dict,adj_dict={},hash_dict={},F_dict={},N_dict={}):

    # loop over all pairs of reactant & product
    for count_i,i in enumerate(sorted(final_products)):

        # get the name of xyz_file
        name = i.split('/')[-1].split('.xyz')[0]

        # create dictionary of adj_mat and hash list
        adj_dict[name] ={}
        hash_dict[name]={}

        # initialize list for reactant/product geometry
        xyz = ['','']
        count=0

        # read in pairs of xyz file
        with open(i,"r") as f:
            for lc,lines in enumerate(f):
                fields = lines.split()
                if lc == 0: 
                    N = int(fields[0])
                    xyz[0] += lines
                    continue
            
                if len(fields) == 1 and float(fields[0]) == float(N):
                    count+=1

                xyz[count]+=lines

        with open('reactant.xyz',"w") as f:
            f.write(xyz[0])

        with open('product.xyz',"w") as f:
            f.write(xyz[1])

        # parse reactant info
        RE,RG   = xyz_parse('reactant.xyz')
        Radj_mat= Table_generator(RE,RG)
        _,_,Rhash_list=canon_geo(RE,Radj_mat)

        # parse product info
        E,PG   = xyz_parse('product.xyz')
        Padj_mat= Table_generator(E,PG)
        _,_,Phash_list=canon_geo(E,Padj_mat)
                
        # Seperate reactant(s)
        R_gs    = graph_seps(Radj_mat)
        Rgroups = []
        loop_ind= []
        for i in range(len(R_gs)):
            if i not in loop_ind:
                new_group = [count_j for count_j,j in enumerate(R_gs[i,:]) if j >= 0]
                loop_ind += new_group
                Rgroups   += [new_group]
            
        # Determine the inchikey of all components in the reactant
        Rinchi_list = []

        for group in Rgroups:
            N_atom = len(group)
            frag_E = [RE[ind] for ind in group]
            frag_G = np.zeros([N_atom,3])
            for count_i,i in enumerate(group):
                frag_G[count_i,:] = RG[i,:]
                
            inchikey = return_inchikey(frag_E,frag_G)
            Rinchi_list+= [inchikey]

        # write info into dictionaies
        N_dict[name] = len(RE)
        try:
            F_dict[name] = sum([E_dict[inchi[:14]]['F'] for inchi in Rinchi_list])
        except:
            F_dict[name] = 0

        adj_dict[name]["reactant"]= Radj_mat                
        adj_dict[name]["product"] = Padj_mat
        hash_dict[name]["reactant"]= Rhash_list              
        hash_dict[name]["product"] = Phash_list

    # clean up xyz files
    try:
        os.remove('reactant.xyz')
        os.remove('product.xyz')
    except:
        pass

    return N_dict,F_dict,adj_dict,hash_dict

# Function to parse Gaussian DFT geo-opt output
def read_Gaussian_output(file_name):
    
    if not os.path.isfile(file_name): return False,False,0.0,0.0,0.0,0.0,0

    imag_flag = False
    finish_flag = False
    ZPE_corr,zero_E,H_298,F_298=0,0,0,0
    grad_lines = []
    # parse Gaussian DFT output file
    with open(file_name,'r') as f: out_lines = f.readlines()

    for lc,lines in enumerate(out_lines):
        fields = lines.split()

        if 'imaginary' in fields and 'frequencies' in fields and '(negative' in fields and int(fields[1]) == 1: imag_flag,imag_line=True,lc
        if 'Normal' in fields and 'termination' in fields and 'of' in fields and 'Gaussian' in fields: finish_flag=True
        if 'Forces' in fields and '(Hartrees/Bohr)' in fields: grad_lines += [lc]
        if len(fields) == 4 and fields[0] == 'Zero-point' and fields[1] == 'correction=' and fields[3] == '(Hartree/Particle)': ZPE_corr = float(fields[-2])
        if len(fields) == 7 and fields[0] == 'Sum' and fields[2] == 'electronic' and fields[4] == 'zero-point': zero_E = float(fields[-1])
        if len(fields) == 7 and fields[0] == 'Sum' and fields[2] == 'electronic' and fields[5] == 'Enthalpies=': H_298 = float(fields[-1])
        if len(fields) == 8 and fields[0] == 'Sum' and fields[2] == 'electronic' and fields[5] == 'Free' and fields[6] == 'Energies=': F_298 = float(fields[-1])

    # if the imaginary frequencies is less than 100 cm**-1, turn imag_flag to False
    if imag_flag:
        imag_line = out_lines[imag_line+9]
        if abs(float(imag_line.split()[2])) < 100:
            print("Small imaginary frequency identified in {} which refers to a tiny vibration rather than a reaction...".format(file_name))
            imag_flag = False

    return finish_flag,imag_flag,zero_E-ZPE_corr,zero_E,H_298,F_298,len(grad_lines)

# Function to parse Orca DFT geo-opt output
def read_Orca_output(file_name):
    
    finish_flag = False
    freq_lines, ZPE_lines, SPE_lines, H_lines, F_lines, grad_lines = [],[],[],[],[],[]
    try:
        with open(file_name,'r') as f: Tlines = f.readlines()
    except:
        return False,False,0.0,0.0,0.0,0.0,0

    for lc,lines in enumerate(Tlines):
        if 'ORCA TERMINATED NORMALLY' in lines: finish_flag = True
        if 'VIBRATIONAL FREQUENCIES' in lines: freq_lines += [lc]
        if 'FINAL SINGLE POINT ENERGY' in lines: SPE_lines += [lc]
        if 'Zero point energy' in lines: ZPE_lines += [lc]
        if 'Total Enthalpy' in lines: H_lines += [lc]
        if 'Final Gibbs free energy' in lines: F_lines += [lc]
        if 'ORCA SCF GRADIENT CALCULATION' in lines: grad_lines += [lc]
        if 'Number of atoms' in lines: N_atoms = int(lines.split()[-1])

    if finish_flag is False: return finish_flag,False,0.0,0.0,0.0,0.0,0
        
    # check imag freq
    freq_line, ZPE_line, SPE_line, H_line, F_line = freq_lines[-1], ZPE_lines[-1], SPE_lines[-1], H_lines[-1], F_lines[-1]
        
    # find imaginary frequency 
    imag_ind = []
    for lc in range(freq_line,freq_line+N_atoms*3+5):
        if 'cm**-1' not in Tlines[lc]: continue
        fields = Tlines[lc].split()
        if '-' in fields[1]: imag_ind += [int(fields[0].split(':')[0])]
        if len(imag_ind) > 2: break
        
    if len(imag_ind) == 1: imag_flag = True
    else: imag_flag = False

    # parse energies
    SPE = float(Tlines[SPE_line].split()[-1])
    ZPE = float(Tlines[ZPE_line].split()[4])
    H_298 = float(Tlines[H_line].split()[-2])
    F_298 = float(Tlines[F_line].split()[-2])
    
    return finish_flag,imag_flag,SPE,SPE+ZPE,H_298,F_298,len(grad_lines)

# Function to read pyGSM inputs and prepare for DFT level TS geo-opt
def read_pyGSM_output(output_folder,TS_folder,N_dict):

    # load in GSM output log file
    with open(output_folder+'/log','r') as f: lines = f.readlines()

    # check success
    if 'Finished GSM!\n' in lines:

        pname = output_folder.split('/')[-1]
        print("pyGSM-xTB for {} finished".format(pname))

        # obtain the location of TS
        for line in lines:
            fields = line.split()
            if len(fields) == 12 and fields[0]=='min' and fields[2]=='node:' and fields[8]=='TS':
                N_TS = int(fields[-1])
                    
            # obatin GSM-xTB barrier height
            if len(fields) == 3 and fields[0]=='TS' and fields[1]=='energy:':
                E_TS = float(fields[2])

        # if TS Energy are not obtained, this is a failure channel
        try: 
            E_TS
        except:
            return False

        # if TS energy is so high, we assert GSM locates to a wrong TS
        if E_TS < 1000:

            # find the geometry of TS
            if os.path.isfile(output_folder+'/scratch/opt_converged_000_000.xyz'):
                xyz_file = output_folder+'/scratch/opt_converged_000_000.xyz'
            else:
                xyz_file = output_folder+'/scratch/grown_string_000_001.xyz'
            with open(xyz_file,'r') as g: lines = g.readlines()
            count = 0
            write_lines = []
            for lc,line in enumerate(lines):
                fields = line.split()
                if len(fields)==1 and fields[0] == str(N_dict[pname]): count += 1
                if count == N_TS + 1: write_lines += [line]
                        
            # write the TS initial guess xyz file 
            output_xyz = TS_folder+'/{}-TS.xyz'.format(pname)
            with open(output_xyz,'w') as g:
                for line in write_lines:
                    g.write(line)

            return pname
            
        else:
            return False

    else:
        return False
    
# function to take in reactant and product geometry and return 1. inchi-index 2. smiles index 3. full smile index
def return_Rindex(E,RG,PG=None,Radj_mat=None,Padj_mat=None):

    # if both PG and Padj_mat are not given, exit
    if Padj_mat is None and PG is None:
        print("At least provide PG or Padj_mat, quit...")
        quit()

    # calculate adj_mat is not given
    if Radj_mat is None: Radj_mat=Table_generator(E,RG)
    if Padj_mat is None: Padj_mat=Table_generator(E,PG)

    # calculate product geometry if PG is not given
    if PG is None: PG = opt_geo(RG,Padj_mat,E,q=0,ff='ghemical',step=100)

    # Seperate compounds
    R_gs    = graph_seps(Radj_mat)
    Rgroups = []
    loop_ind= []
    for i in range(len(R_gs)):
        if i not in loop_ind:
            new_group = [count_j for count_j,j in enumerate(R_gs[i,:]) if j >= 0]
            loop_ind += new_group
            Rgroups   += [new_group]

    P_gs    = graph_seps(Padj_mat)
    Pgroups = []
    loop_ind= []
    for i in range(len(P_gs)):
        if i not in loop_ind:
            new_group = [count_j for count_j,j in enumerate(P_gs[i,:]) if j >= 0]
            loop_ind += new_group
            Pgroups   += [new_group]

    # obtain smiles and inchikey for each component in reactants
    Rinchi_list = []
    Rsmile_list = []
    for group in Rgroups:
        N_atom = len(group)
        frag_E = [E[ind] for ind in group]
        frag_G = np.zeros([N_atom,3])
        for count_i,i in enumerate(group):
            frag_G[count_i,:] = RG[i,:]
                
        Rinchi_list += [return_inchikey(frag_E,frag_G)[:14]]
        Rsmile_list += [return_smi(frag_E,frag_G)]

    # obtain smiles and inchikey for each component in products
    Pinchi_list = []
    Psmile_list = []
    for group in Pgroups:
        N_atom = len(group)
        frag_E = [E[ind] for ind in group]
        frag_G = np.zeros([N_atom,3])
        for count_i,i in enumerate(group):
            frag_G[count_i,:] = PG[i,:]
            
        Pinchi_list += [return_inchikey(frag_E,frag_G)[:14]]
        Psmile_list += [return_smi(frag_E,frag_G)]

    # find duplicate inchikeys and then check whether the bond changes
    duplicates = [inchi for inchi in Pinchi_list if inchi in Rinchi_list]
    for dup in duplicates:
        Rgroup = Rgroups[Rinchi_list.index(dup)]
        Pgroup = Pgroups[Pinchi_list.index(dup)]
        R_frag_adj = Radj_mat[Rgroup][:,Rgroup]
        P_frag_adj = Padj_mat[Pgroup][:,Pgroup]
        try:
            if np.abs(R_frag_adj-P_frag_adj).sum() != 0: duplicates.remove(dup)
        except:
            duplicates.remove(dup)

    if len(duplicates) > 0:
        
        # find spectater index in R and P
        dup_inds = [(Rinchi_list.index(dup),Pinchi_list.index(dup)) for dup in duplicates]
        Rdup,Pdup= map(list, zip(*dup_inds))  

        # determine reaction index
        Rsmile_ind_full = '.'.join(sorted(Rsmile_list))
        Psmile_ind_full = '.'.join(sorted(Psmile_list))
        Rsmile_ind = '.'.join(sorted([smi for count,smi in enumerate(Rsmile_list) if count not in Rdup]))
        Psmile_ind = '.'.join(sorted([smi for count,smi in enumerate(Psmile_list) if count not in Pdup]))
        Rinchi_ind = '.'.join(sorted([inc for count,inc in enumerate(Rinchi_list) if count not in Rdup]))
        Pinchi_ind = '.'.join(sorted([inc for count,inc in enumerate(Pinchi_list) if count not in Pdup]))

    else:

        # determine inchi ind and smile ind
        Rsmile_ind = '.'.join(sorted(Rsmile_list))
        Psmile_ind = '.'.join(sorted(Psmile_list))
        Rinchi_ind = '-'.join(sorted(Rinchi_list))
        Pinchi_ind = '-'.join(sorted(Pinchi_list))
        Rsmile_ind_full,Psmile_ind_full = Rsmile_ind,Psmile_ind

    # determine the final index
    reaction_inchi_ind = (sorted([Rinchi_ind,Pinchi_ind])[0],sorted([Rinchi_ind,Pinchi_ind])[1])
    reaction_smile_ind = (sorted([Rsmile_ind,Psmile_ind])[0],sorted([Rsmile_ind,Psmile_ind])[1])
    reaction_smile_ind_full = (sorted([Rsmile_ind_full,Psmile_ind_full])[0],sorted([Rsmile_ind_full,Psmile_ind_full])[1])

    return reaction_inchi_ind,reaction_smile_ind,reaction_smile_ind_full

# Function to analyze TS output and classify reactions
def TS_characterize(input_reaction,TS_output,model='IRC_model.json',thresh=0.5,package='Gaussian'):
    
    # load in xgboost model
    from xgboost import XGBClassifier
    IRCmodel = XGBClassifier()
    IRCmodel.load_model(model)

    # obtain intended reaction info from the input xyz file
    E,RG,PG  = parse_input(input_reaction)
    Radj_mat = Table_generator(E,RG)
    Padj_mat = Table_generator(E,PG)

    # determine bond changes and unchanged bonds
    lone,bond,core,Rbond_mat,fc = find_lewis(E,Radj_mat,return_pref=False,return_FC=True)
    lone,bond,core,Pbond_mat,fc = find_lewis(E,Padj_mat,return_pref=False,return_FC=True)
    BE_change   = Pbond_mat[0] - Rbond_mat[0]
    bond_break  = []
    bond_form   = []
    normal_bonds= []
    for i in range(len(E)):
        for j in range(i+1,len(E)):
            if BE_change[i][j] == -1:
                bond_break += [(i,j)]
                
            if BE_change[i][j] == 1:
                bond_form += [(i,j)]

            if BE_change[i][j] == 0 and Radj_mat[i][j] == 1:
                normal_bonds += [(i,j)]

    involve = [set(sorted(list(sum(bond_break, ())))),set(sorted(list(sum(bond_form, ()))))]

    # determine "reactive atom list"
    if involve[0]==involve[1]:
        involve_atoms = involve[0]
    else:
        if len(involve[0]) == 4: involve_atoms =involve[0]
        else: involve_atoms =involve[1]

    # compute imaginary frequency mode 
    TS_Dis_mean,TS_Dis_min,TS_unexpect,dist = analyze_TS(E,Radj_mat,Padj_mat,TS_output,package=package)

    # use trained xgboost model to predict the intended rate proba
    intended_proba = IRCmodel.predict_proba(np.array([[TS_Dis_mean,TS_Dis_min,TS_unexpect,dist]]))[0][1]

    if intended_proba > thresh: return True
    else: return False
                

# Function to analyze Gaussian TS output and classify reactions
def channel_classification(output_path,initial_xyz,inp_list,low_IRC_dict,N_dict,adj_dict,F_dict,reaction_dict,TS_folder=None,append=True,model='IRC_model.json',thresh=0.55,package='Gaussian',dg_thresh=None):

    # load in xgboost model
    from xgboost import XGBClassifier
    IRCmodel = XGBClassifier()
    IRCmodel.load_model(model)

    # find Gaussian TS jobs folder
    if TS_folder is None:
        TS_folder = os.path.join(output_path,'TS-folder')

    # define keep_list for the jobs classified as unintended 
    keep_list = []

    # count Number of gradient calls
    N_grad = 0
    
    # TS gibbs free energy dictionary
    TSEne_dict={}

    # create report file 
    with open(output_path+'/report.txt','w') as g:
        g.write('{:<40s} {:<60s} {:<15s}\n'.format('channel','product',"barrier"))

    # check job status, if TS successfully found, put into IRC list
    for TSinp in inp_list:

        # return file name
        pname = TSinp.split('/')[-1].split('-TS')[0]

        # change .gjf/.in to .out will return TS geo-opt output file
        if package.lower() == 'gaussian': 
            TSout = TSinp.replace('.gjf','.out')
            finish_flag,imag_flag,SPE,zero_E,H_298,F_298,N_g = read_Gaussian_output(TSout)

        if package.lower() == 'orca': 
            TSout = TSinp.replace('.in','.out')
            finish_flag,imag_flag,SPE,zero_E,H_298,F_298,N_g = read_Orca_output(TSout)

        # for the success tasks, generate optimized TS geometry
        if imag_flag and finish_flag:

            if package.lower() == 'gaussian': 
                # apply read_Gaussian_output.py to obatin optimized TS geometry
                command='python {}/utilities/read_Gaussian_output.py -t geo-opt -i {} -o {}/{}-TS.xyz -n {} --count'
                os.system(command.format('/'.join(os.getcwd().split('/')[:-1]),TSout,TS_folder,pname,N_dict[pname]))

            elif package.lower() == 'orca':
                if TSinp.replace('.in','.xyz') != '{}/{}-TS.xyz'.format(TS_folder,pname):
                    os.system("cp {} {}/{}-TS.xyz".format(TSinp.replace('.in','.xyz'),TS_folder,pname))
                    
            # load in free energy
            TSEne_dict[pname]=F_298

            # add gradient calls to Total gradient calls 
            N_grad += N_g

            # obtain intended reaction info from the input xyz file
            xyz_input= '{}/{}.xyz'.format(initial_xyz,pname)
            E,RG,PG  = parse_input(xyz_input)
            Radj_mat = adj_dict[pname]['reactant']
            Padj_mat = adj_dict[pname]['product']
            _,_,Rhash_list=canon_geo(E,Radj_mat)
            _,_,Phash_list=canon_geo(E,Padj_mat)

            # get the smiles string of intended product
            Psmiles  = return_smi(E,PG,Padj_mat)

            # determine bond changes and unchanged bonds
            lone,bond,core,Rbond_mat,fc = find_lewis(E,Radj_mat,return_pref=False,return_FC=True)
            lone,bond,core,Pbond_mat,fc = find_lewis(E,Padj_mat,return_pref=False,return_FC=True)
            BE_change   = Pbond_mat[0] - Rbond_mat[0]
            bond_break  = []
            bond_form   = []
            normal_bonds= []
            for i in range(len(E)):
                for j in range(i+1,len(E)):
                    if BE_change[i][j] == -1:
                        bond_break += [(i,j)]
                
                    if BE_change[i][j] == 1:
                        bond_form += [(i,j)]

                    if BE_change[i][j] == 0 and Radj_mat[i][j] == 1:
                        normal_bonds += [(i,j)]

            involve = [set(sorted(list(sum(bond_break, ())))),set(sorted(list(sum(bond_form, ()))))]

            # determine "reactive atom list"
            if involve[0]==involve[1]:
                involve_atoms = involve[0]
            else:
                if len(involve[0]) == 4:
                    involve_atoms =involve[0]
                else:
                    involve_atoms =involve[1]

            # load in DFT level and low level TS geometry
            _,low_G = xyz_parse('{}/{}/{}-TS.xyz'.format(output_path,pname,pname))
            _,DFT_G = xyz_parse('{}/{}-TS.xyz'.format(TS_folder,pname))

            # compute displacement indicators
            try:
                Dis,bond_dis = return_Dis(E,low_G,DFT_G,sorted(list(involve_atoms)),normal_bonds)
                RMSD   = return_RMSD(E,low_G,DFT_G,rotate=True,mass_weighted=True)        
                continue_flag = True
            except:
                continue_flag = False
            
            if not continue_flag:
                keep_list += [TSinp]
                continue

            # compute imaginary frequency mode 
            TS_Dis_mean,TS_Dis_min,TS_unexpect,dist = analyze_TS(E,Radj_mat,Padj_mat,'{}/{}-TS.out'.format(TS_folder,pname),package=package)

            # use trained xgboost model to predict the intended rate proba
            try:
                intended_proba = IRCmodel.predict_proba(np.array([[Dis,bond_dis,RMSD,low_IRC_dict[pname],TS_Dis_mean,TS_Dis_min,TS_unexpect,dist]]))[0][1]
            except:
                intended_proba = IRCmodel.predict_proba(np.array([[TS_Dis_mean,TS_Dis_min,TS_unexpect,dist]]))[0][1]
                
            if intended_proba < thresh:
                if dg_thresh is not None and (F_298-F_dict[pname])*627.5 > dg_thresh: 
                    print("TS for reaction payway {} is likely to be unintended and with high barrier ({}, threshold is {} kcal/mol), discard it...".format(TSinp,(F_298-F_dict[pname])*627.5,dg_thresh))
                else: 
                    keep_list += [TSinp]

            else:
                print("TS for reaction payway to {} is found with Energy barrier {}".format(Psmiles,(F_298-F_dict[pname])*627.5))
                # write intended channel into the report
                with open(output_path+'/report.txt','a') as g:
                    g.write('{:<40s} {:<60s} {:<15s} {:<15.4f}\n'.format(pname,Psmiles,'Intended',(F_298-F_dict[pname])*627.5))
 
                # write into reaction database if is needed
                if append:
                    G1,G2,adj_1,adj_2,rtype=RG,PG,Radj_mat,Padj_mat,'Intended'
                    reaction_index,smile_ind,full_smile_ind = return_Rindex(E,G1,G2,adj_1,adj_2)
                
                    # if this reaction pathway has not been included in the database, create this item
                    if reaction_index not in reaction_dict.keys():
                        reaction_dict[reaction_index]={}
                        reaction_dict[reaction_index]['smiles_index'] = smile_ind
                        reaction_dict[reaction_index]['type']=rtype
                        reaction_dict[reaction_index]['TS']={}
                        
                        # append TS into dict
                        TS_ind = 0
                        reaction_dict[reaction_index]['TS'][TS_ind] = {}
                        reaction_dict[reaction_index]['TS'][TS_ind]['smiles_index(full)'] = full_smile_ind
                        reaction_dict[reaction_index]['TS'][TS_ind]['source']="{}/{}".format(output_path,pname)
                        reaction_dict[reaction_index]['TS'][TS_ind]['E'],reaction_dict[reaction_index]['TS'][TS_ind]['G']=xyz_parse('{}/TS-folder/{}-TS.xyz'.format(output_path,pname))
                        reaction_dict[reaction_index]['TS'][TS_ind]['RG'],reaction_dict[reaction_index]['TS'][TS_ind]['PG']=G1,G2
                        reaction_dict[reaction_index]['TS'][TS_ind]['TS_DG'] = TSEne_dict[pname]
                        reaction_dict[reaction_index]['TS'][TS_ind]['IRC_type'] = rtype
                        reaction_dict[reaction_index]['TS_Energy']=[TSEne_dict[pname]]
                
                    # if this reaction pathway already exist in the database, check TS energy to see whether this is another TS structure 
                    else:
                        if min([abs(Ene - TSEne_dict[pname]) for Ene in reaction_dict[reaction_index]['TS_Energy']]) > 5.0 / 627.5:

                            # update reaction type if intended channel is given     
                            if reaction_dict[reaction_index]['type'] == 'unintended' and rtype == 'intended':
                                reaction_dict[reaction_index]['type'] = 'intended'

                            # append TS into dict 
                            TS_ind = len(reaction_dict[reaction_index]['TS'].keys())
                            reaction_dict[reaction_index]['TS'][TS_ind] = {}
                            reaction_dict[reaction_index]['TS'][TS_ind]['smiles_index(full)'] = full_smile_ind
                            reaction_dict[reaction_index]['TS'][TS_ind]['source']="{}/{}".format(output_path,pname)
                            reaction_dict[reaction_index]['TS'][TS_ind]['E'],reaction_dict[reaction_index]['TS'][TS_ind]['G']=xyz_parse('{}/TS-folder/{}-TS.xyz'.format(output_path,pname))
                            reaction_dict[reaction_index]['TS'][TS_ind]['RG'],reaction_dict[reaction_index]['TS'][TS_ind]['PG']=G1,G2
                            reaction_dict[reaction_index]['TS'][TS_ind]['TS_DG'] = TSEne_dict[pname]
                            reaction_dict[reaction_index]['TS'][TS_ind]['IRC_type'] = rtype
                            reaction_dict[reaction_index]['TS_Energy'] += [TSEne_dict[pname]]

        else:
            print("TS for reaction payway to {} fails (either no imag freq or geo-opt fails)".format(pname))
            continue

    return N_grad,reaction_dict,keep_list

# Function to analyze Gaussian IRC outputs and update reaction database
def read_IRC_outputs(output_path,IRCinp_list,adj_dict,hash_dict,N_dict,reaction_dict,F_dict=None,TSEne_dict=None,append=False,select=False,dg_thresh=None,folder_name='IRC-result',package='Gaussian'):
    
    # make folder for DFT 
    IRC_result= '/'.join([output_path,folder_name])
    IRC_type  = {}
    keep_list = []

    # make a IRC result folder
    if os.path.isdir(IRC_result) is False: os.mkdir(IRC_result)

    # create record.txt to write the IRC result
    with open(IRC_result+'/IRC-record.txt','w') as g:
        g.write('{:<40s} {:<60s} {:<60s} {:<15s} {:<15s}\n'.format('channel','Node1','Node2','type','barrier'))

    # loop over IRC output files
    for IRCinp in IRCinp_list:

        # obtain file name
        pname = IRCinp.split('/')[-1].split('-IRC')[0]

        if package.lower() == 'gaussian':
            
            # change .gjf to .out will return IRC output file  
            IRCout = IRCinp.replace('.gjf','.out')
            # create finish_flag to check whether IRC task is normally finished
            finish_flag=False

            # read the IRC output file, chech whether it is finished and what is final image number
            if os.path.isfile(IRCout) is False:
                print("Missing IRC calculation job {}, skip...".format(IRCout))
                continue

            with open(IRCout,'r') as f:
                for lc,lines in enumerate(f):
                    fields = lines.split()
                    if 'Normal' in fields and 'termination' in fields and 'of' in fields and 'Gaussian' in fields:
                        finish_flag=True
                
                    if len(fields)== 5 and fields[0]=='Total' and fields[1]=='number' and fields[2]=='of' and fields[3]=='points:':
                        N_image = int(fields[4]) + 1

            # If IRC task finished, parse IRC output file
            if finish_flag:
                print("IRC for reaction payway to {} is finished".format(pname))
                # apply read_Gaussian_output.py to generate IRC pathway file
                IRC_xyz = '{}/{}-IRC.xyz'.format(IRC_result,pname)
                command='python {}/utilities/read_Gaussian_output.py -t IRC -i {} -o {} -n {}'
                os.system(command.format('/'.join(os.getcwd().split('/')[:-1]),IRCout,IRC_xyz,N_dict[pname]))

            else:
                print("IRC calculation for reaction payway to {} fails".format(pname))
                continue

        elif package.lower() == 'orca':
            
            # change .in to .out will return IRC output file  
            IRCout = IRCinp.replace('.in','.out')
            # create finish_flag to check whether IRC task is normally finished
            finish_flag=False

            # read the IRC output file, chech whether it is finished and what is final image number
            with open(IRCout,'r') as f: lines = f.readlines()
            for line in lines:
                if 'ORCA TERMINATED NORMALLY' in line: 
                    finish_flag=True
                    break

            # If IRC task finished, parse IRC output file
            if finish_flag:
                print("IRC for reaction payway to {} is finished".format(pname))
                IRC_xyz = '{}/{}-IRC.xyz'.format(IRC_result,pname)
                os.system('cp {} {}'.format(IRCinp.replace('.in','_IRC_Full_trj.xyz'),IRC_xyz))

            else:
                print("IRC calculation for reaction payway to {} fails".format(pname))
                continue

        # find the geometry of reactant & product
        with open(IRC_xyz,'r') as g: lines = g.readlines()
        count = 0
        write_reactant= []
        write_product = []
        Energy_list   = []
        N_image = int(len(lines)/(N_dict[pname]+2))
        for lc,line in enumerate(lines):
            fields = line.split()
            if len(fields)==1 and fields[0] == str(N_dict[pname]): count += 1
            if len(fields) == 3 and 'energy' in fields[2]: Energy_list += [float(fields[2].split(':')[-1])]
            if 'Coordinates' in fields and 'E' in fields: Energy_list += [float(fields[-1])]
            if count == 1: write_reactant+= [line]
            if count == N_image: write_product += [line]

        Energy_list = np.array(Energy_list)

        # write the reactant and product
        with open(IRC_result+'/{}-start.xyz'.format(pname),'w') as g:
            for line in write_reactant: g.write(line)

        # parse IRC start point xyz file
        NE,NG1  = xyz_parse('{}/{}-start.xyz'.format(IRC_result,pname))
        N_adj_1 = Table_generator(NE,NG1)
        smiles1 = return_smi(NE,NG1,N_adj_1)
        _,_,Nhash_list1=canon_geo(NE,N_adj_1)

        # generate end point of IRC
        with open(IRC_result+'/{}-end.xyz'.format(pname),'w') as g:
            for line in write_product: g.write(line)
            
        # parse IRC start point xyz file
        NE,NG2  = xyz_parse('{}/{}-end.xyz'.format(IRC_result,pname))
        N_adj_2 = Table_generator(NE,NG2)
        smiles2 = return_smi(NE,NG2,N_adj_2)
        _,_,Nhash_list2=canon_geo(NE,N_adj_2)

        # add info into dictionary
        o_adj_1  = adj_dict[pname]['reactant']
        o_adj_2  = adj_dict[pname]['product']
        o_hash1  = np.array(hash_dict[pname]['reactant'])
        o_hash2  = np.array(hash_dict[pname]['product'])
        adj_diff = np.abs((N_adj_1+N_adj_2) - (o_adj_1+o_adj_2))
        hash_diff= np.abs((np.array(Nhash_list1)+np.array(Nhash_list2)) - (o_hash1+o_hash2))
            
        if adj_diff.sum() == 0 or hash_diff.sum() == 0:
            words = "Intended"
            rtype = 'intended'
            keep_list += [IRCinp.replace('IRC.','TS.')]
            IRC_type[pname] = 1

        else:
            if adj_diff.sum() == 2:
                words = "Intended (1 bond diff)"
                rtype = 'intended'
                IRC_type[pname] = 0
            else:
                words = "Unintended"
                rtype = 'unintended'
                IRC_type[pname] = 0

            # check whether one of the node matches with reactant
            # case one: only if reactant exist in the IRC pathway will we keep this candidate 
            if np.abs(np.array(Nhash_list1)-o_hash1).sum() == 0 or np.abs(np.array(Nhash_list2)-o_hash1).sum() == 0:
            # case two: if one of the node matches with reactant/product will we keep this candidate
            #if np.abs(np.array(Nhash_list1)-o_hash1).sum() == 0 or np.abs(np.array(Nhash_list2)-o_hash1).sum() == 0 or np.abs(np.array(Nhash_list1)-o_hash2).sum() == 0 or np.abs(np.array(Nhash_list2)-o_hash2).sum() == 0:
                keep_list += [IRCinp.replace('IRC.','TS.')]

        # parse barrier from IRC output
        barrier1 = 627.5*(max(Energy_list)-Energy_list[0])
        barrier2 = 627.5*(max(Energy_list)-Energy_list[1])

        if select:
            # if one node is reactant, print out barrier
            if np.abs(np.array(Nhash_list1)-o_hash1).sum() == 0: barrier = '{:<5.2f}'.format(barrier1) 
            elif np.abs(np.array(Nhash_list2)-o_hash1).sum() == 0: barrier = '{:<5.2f}'.format(barrier2)
            else: barrier = 'None'

            # further select based on DG if df_thresh is applied
            if dg_thresh is not None and barrier != 'None':
                if float(barrier) > dg_thresh + 5 and IRCinp.replace('IRC.','TS.') in keep_list: 
                    print("Exclude {} due to high avtivation barrier ({}, threshold is {} kcal/mol)".format(IRCinp,barrier,dg_thresh))
                    keep_list.remove(IRCinp.replace('IRC.','TS.'))

            with open(IRC_result+'/IRC-record.txt','a') as g:
                g.write('{:<40s} {:<60s} {:<60s} {:15s} {:10s}\n'.format(pname,'{} ({:<5.2f})'.format(smiles1,barrier1),'{} ({:<5.2f})'.format(smiles2,barrier2),words,barrier))

        else:
            # compute Gibbs free energy of activation
            barrier = (TSEne_dict[pname]-F_dict[pname])*627.5
            # exclude same smiles string, they serve as spectator
            reaction_index,smile_ind,full_smile_ind = return_Rindex(NE,NG1,NG2,N_adj_1,N_adj_2)
            with open(IRC_result+'/IRC-record.txt','a') as g:
                g.write('{:<40s} {:<60s} {:<60s} {:15s} {:<15.4f}\n'.format(pname,full_smile_ind[0],full_smile_ind[1],words,barrier))
    
        if append:

            # if this reaction pathway has not been included in the database, create this item
            if reaction_index not in reaction_dict.keys():
                reaction_dict[reaction_index]={}
                reaction_dict[reaction_index]['smiles_index'] = smile_ind
                reaction_dict[reaction_index]['type']=rtype
                reaction_dict[reaction_index]['TS']={}

                # append TS into dict
                TS_ind = 0
                reaction_dict[reaction_index]['TS'][TS_ind] = {}
                reaction_dict[reaction_index]['TS'][TS_ind]['smiles_index(full)'] = full_smile_ind
                reaction_dict[reaction_index]['TS'][TS_ind]['source']="{}/{}-IRC.xyz".format(IRC_result,pname)
                reaction_dict[reaction_index]['TS'][TS_ind]['E'],reaction_dict[reaction_index]['TS'][TS_ind]['G']=xyz_parse('{}/TS-folder/{}-TS.xyz'.format(output_path,pname))
                reaction_dict[reaction_index]['TS'][TS_ind]['RG'],reaction_dict[reaction_index]['TS'][TS_ind]['PG']=NG1,NG2
                reaction_dict[reaction_index]['TS'][TS_ind]['TS_DG'] = TSEne_dict[pname]
                reaction_dict[reaction_index]['TS'][TS_ind]['IRC_type'] = rtype
                reaction_dict[reaction_index]['TS_Energy']=[TSEne_dict[pname]]
                
            # if this reaction pathway already exist in the database, check TS energy to see whether this is another TS structure 
            else:
                if min([abs(Ene - TSEne_dict[pname]) for Ene in reaction_dict[reaction_index]['TS_Energy']]) > 5.0 / 627.5:

                    # update reaction type if intended channel is given     
                    if reaction_dict[reaction_index]['type'] == 'unintended' and rtype == 'intended': reaction_dict[reaction_index]['type'] = 'intended'

                    # append TS into dict 
                    TS_ind = len(reaction_dict[reaction_index]['TS'].keys())
                    reaction_dict[reaction_index]['TS'][TS_ind] = {}
                    reaction_dict[reaction_index]['TS'][TS_ind]['smiles_index(full)'] = full_smile_ind
                    reaction_dict[reaction_index]['TS'][TS_ind]['source']="{}/{}-IRC.xyz".format(IRC_result,pname)
                    reaction_dict[reaction_index]['TS'][TS_ind]['E'],reaction_dict[reaction_index]['TS'][TS_ind]['G']=xyz_parse('{}/TS-folder/{}-TS.xyz'.format(output_path,pname))
                    reaction_dict[reaction_index]['TS'][TS_ind]['RG'],reaction_dict[reaction_index]['TS'][TS_ind]['PG']=NG1,NG2
                    reaction_dict[reaction_index]['TS'][TS_ind]['TS_DG'] = TSEne_dict[pname]
                    reaction_dict[reaction_index]['TS'][TS_ind]['IRC_type'] = rtype
                    reaction_dict[reaction_index]['TS_Energy'] += [TSEne_dict[pname]]
                    
    if select: return keep_list,IRC_type
    else: return reaction_dict

# Function to parse the energy dictionary
def parse_Energy(db_files,E_dict={}):
    with open(db_files,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if lc == 0: continue
            if len(fields) ==0: continue
            if len(fields) >= 4:
                inchi = fields[0][:14]
                if fields[0] not in E_dict.keys():
                    E_dict[inchi] = {}
                    E_dict[inchi]["E_0"]= float(fields[1])
                    E_dict[inchi]["H"]  = float(fields[2])
                    E_dict[inchi]["F"]  = float(fields[3])
                    E_dict[inchi]["SPE"]= float(fields[4])

    return E_dict

# return hash value function
def return_hash(elements,geo,adj_mat):
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
    hash_list = [ atom_hash(i,adj_mat,masses) for i in range(len(geo)) ]

    return hash_list

# Function that sleeps the script until jobids are no longer in a running or pending state in the queue
def monitor_jobs(jobids):
    
    current_jobs = check_queue()
    while True in [ i in current_jobs for i in jobids ]:
        time.sleep(60)
        current_jobs = check_queue()  
    return

# Returns the pending and running jobids for the user as a list
def check_queue():

    # The first time this function is executed, find the user name and scheduler being used. 
    if not hasattr(check_queue, "user"):

        # Get user name
        check_queue.user = subprocess.check_output("echo ${USER}", shell=True).decode('utf-8').strip("\r\n")

        # Get batch system being used
        squeue_tmp = subprocess.Popen(['which', 'squeue'], stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').strip("\r\n")
        qstat_tmp = subprocess.Popen(['which', 'qstat'], stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').strip("\r\n")
        check_queue.sched =  None
        if "no squeue in" not in squeue_tmp:
            check_queue.sched = "slurm"
        elif "no qstat in" not in qstat_tmp:
            check_queue.sched = "pbs"
        else:
            print("ERROR in check_queue: neither slurm or pbs schedulers are being used.")
            quit()

    # Get running and pending jobs using the slurm scheduler
    if check_queue.sched == "slurm":

        # redirect a squeue call into output
        output = subprocess.check_output("squeue -l", shell=True).decode('utf-8')

        # Initialize job information dictionary
        jobs = []
        id_ind = None
        for count_i,i in enumerate(output.split('\n')):            
            fields = i.split()
            if len(fields) == 0: continue                
            if id_ind is None and "JOBID" in fields:
                id_ind = fields.index("JOBID")
                if "STATE" not in fields:
                    print("ERROR in check_queue: Could not identify STATE column in squeue -l output.")
                    quit()
                else:
                    state_ind = fields.index("STATE")
                if "USER" not in fields:
                    print("ERROR in check_queue: Could not identify USER column in squeue -l output.")
                    quit()
                else:
                    user_ind = fields.index("USER")
                continue

            # If this job belongs to the user and it is pending or running, then add it to the list of active jobs
            if id_ind is not None and fields[user_ind] == check_queue.user and fields[state_ind] in ["PENDING","RUNNING"]:
                jobs += [fields[id_ind]]

    # Get running and pending jobs using the pbs scheduler
    elif check_queue.sched == "pbs":

        # redirect a qstat call into output
        output = subprocess.check_output("qstat -f", shell=True).decode('utf-8')

        # Initialize job information dictionary
        jobs = []
        job_dict = {}
        current_key = None
        for count_i,i in enumerate(output.split('\n')):
            fields = i.split()
            if len(fields) == 0: continue
            if "Job Id" in i:

                # Check if the previous job belongs to the user and needs to be added to the pending or running list. 
                if current_key is not None:
                    if job_dict[current_key]["State"] in ["R","Q"] and job_dict[current_key]["User"] == check_queue.user:
                        jobs += [current_key]
                current_key = i.split()[2]
                job_dict[current_key] = { "State":"NA" , "Name":"NA", "Walltime":"NA", "Queue":"NA", "User":"NA"}
                continue
            if "Job_Name" == fields[0]:
                job_dict[current_key]["Name"] = fields[2]
            if "job_state" == fields[0]:
                job_dict[current_key]["State"] = fields[2]
            if "queue" == fields[0]:
                job_dict[current_key]["Queue"] = fields[2]
            if "Resource_List.walltime" == fields[0]:
                job_dict[current_key]["Walltime"] = fields[2]        
            if "Job_Owner" == fields[0]:
                job_dict[current_key]["User"] = fields[2].split("@")[0]

        # Check if the last job belongs to the user and needs to be added to the pending or running list. 
        if current_key is not None:
            if job_dict[current_key]["State"] in ["R","Q"] and job_dict[current_key]["User"] == check_queue.user:
                jobs += [current_key]

    return jobs

# Function to check and append DFT energies for reactants
def check_DFT_energy(reactant_smiles,E_dict,conf_path,functional,basis,config,dispersion=None,solvation=None,temperature=None,package='Gaussian'):

    # initialize input input file path and crest inputs
    inp_path,R_list = [],[]    

    # loop over eact reactant to check whether DFT energy is given
    for smi in reactant_smiles:

        # parse the smiles string and generate elements and geometry
        readin,E,G,q = parse_smiles(smi,ff='mmff94') 
        if readin is False:
            readin,E,G,q = parse_smiles(smi,ff='uff') 
            if readin is False:
                print("Check {}, fail to parse it...".format(smi))
                continue
            
        adj_mat = Table_generator(E,G)
        inchikey= return_inchikey(E,G,adj_mat)
        inchi_s = inchikey[:14]
        if inchi_s not in E_dict.keys():
            print("{} doesn't has DFT energy at given level of theory, put it into computing list...".format(smi))
            lone_electrons,bonding_electrons,core_electrons,bond_mat,fc = find_lewis(E,adj_mat,q_tot=q,keep_lone=[],return_pref=False,return_FC=True)
            keep_lone = [ [ count_j for count_j,j in enumerate(lone_electron) if j%2 != 0] for lone_electron in lone_electrons][0]
            multiplicity = 1+len(keep_lone)

            if os.path.isdir(conf_path+'/{}'.format(inchi_s)) is False and os.path.isdir(conf_path+'/{}'.format(inchikey)) is False:
                xyz_write('xyz_files/{}.xyz'.format(inchikey),E,G)
                R_list += [os.getcwd()+'/xyz_files/{}.xyz'.format(inchikey)]
                continue
            elif os.path.isdir(conf_path+'/{}'.format(inchi_s)):
                geo_file = conf_path+'/{}/crest_best.xyz'.format(inchi_s)
            else:
                geo_file = conf_path+'/{}/crest_best.xyz'.format(inchikey)

            # generate gjf files from the most stable crest conformer
            if package.lower() == 'gaussian':
                level = "{}/{}".format(functional,basis)
                if dispersion is not None: level += ' EmpiricalDispersion=G{}'.format(dispersion)
                if solvation is not None: level += ' SCRF=({},solvent={})'.format(solvation.split('/')[0],solvation.split('/')[1])
                substring = "python {}/utilities/xyz_to_Gaussian.py {} -o DFT/{}.gjf -q {} -m {} -c \"{}\" -ty \"{} Opt=(maxcycles=100) Int=UltraFine SCF=QC Freq\" -t \"{} DFT\" "
                substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),geo_file,inchikey,q,multiplicity,"False",level,inchikey)
                output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0]            
                #print(inchikey,substring)
                # Check success
                if os.path.isfile("DFT/{}.gjf".format(inchikey)) is False:
                    print("ERROR: generation of the input file {}.gjf failed.".format(inchikey))
                    quit()
                inp_path += ["DFT/{}.gjf".format(inchikey)]

            elif package.lower() == 'orca':
                level = '-f {}'.format(functional)
                if dispersion is not None: level += ' -d {}'.format(dispersion)
                if basis is not None: level +=' -b {}'.format(basis)
                if temperature is not None:
                    substring = "python {}/utilities/xyz_to_Orca.py {} -p {} -o DFT/{} -q {} -m {} -T {} {} -ty \"Opt Freq\"  "
                    substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),geo_file,config["procs"],inchikey,q,multiplicity,temperature,level)
                else:
                    substring = "python {}/utilities/xyz_to_Orca.py {} -p {} -o DFT/{} -q {} -m {} {} -ty \"Opt Freq\"  "
                    substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),geo_file,config["procs"],inchikey,q,multiplicity,level)

                output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0]            

                # Check success
                if os.path.isfile("DFT/{}.in".format(inchikey)) is False:
                    print("ERROR: generation of the input file {}.in failed.".format(inchikey))
                    quit()
                inp_path += ["DFT/{}.in".format(inchikey)]

    if len(R_list) > 0:
        print("Missing CREST results, first run CREST jobs and then run DFT calculations...")
        return False,R_list

    # if has gif files, run Gaussian geo-opt
    if len(inp_path) > 0:    
        
        # Remove all .submit files in this folder
        submit_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames if fnmatch.fnmatch(f,'G_opt.*') ]
        if len(submit_files) > 0: os.system("rm G_opt.*")

        # Generate geo_opt job and wait for the result
        if package.lower() == 'gaussian':
            substring="python {}/utilities/Gaussian_submit.py -f '*.gjf' -ff \"{}\" -d DFT -para {} -p {} -n {} -ppn {} -q {} -mem {} -sched {} -t {} -o G_opt --silent"
            substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),inp_path,config["parallel"],config["procs"],config["njobs"],config["ppn"],config["queue"],config["memory"],config["sched"],config["wt"])   
            subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0] 

        elif package.lower() == 'orca':
            substring="python {}/utilities/Orca_submit.py -f '*.in' -ff \"{}\" -d DFT -para {} -p {} -n {} -ppn {} -q {} -mem {} -sched {} -t {} -o G_opt --silent"
            substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),inp_path,config["parallel"],config["procs"],config["njobs"],config["ppn"],config["queue"],config["memory"],config["sched"],config["wt"])   
            subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0] 
            
        # submit all the jobs
        substring="python {}/utilities/job_submit.py -f 'G_opt.*.submit' -sched {}".format('/'.join(os.getcwd().split('/')[:-1]),config["sched"])
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0] 
        output = output.decode('utf-8')
        if config["batch"] == 'pbs': print("\t running {} DFT jobs...".format(len(output.split())))
        elif config["batch"] == 'slurm': print("\t running {} DFT jobs...".format(int(len(output.split())/4)))
        monitor_jobs(output.split())

    # read in Gaussian outputs to get DFT energies
    for inp in inp_path:

        if package.lower() == 'gaussian':
            DFTout = inp.replace('.gjf','.out')
            inchikey = inp.split('/')[-1].split('.gjf')[0]
            finish_flag,imag_flag,SPE,zero_E,H_298,F_298,_ = read_Gaussian_output(DFTout)

        elif package.lower() == 'orca':
            DFTout = inp.replace('.in','.out')
            inchikey = inp.split('/')[-1].split('.in')[0]
            finish_flag,imag_flag,SPE,zero_E,H_298,F_298,_ = read_Orca_output(DFTout)

        if imag_flag:
            print("Warning, DFT optimized structure for {} contains an imaginary frequence, please update it later...".format(inchikey))

        if finish_flag:
            E_dict[inchikey[:14]]={}
            E_dict[inchikey[:14]]["E_0"]= zero_E
            E_dict[inchikey[:14]]["H"]  = H_298
            E_dict[inchikey[:14]]["F"]  = F_298
            E_dict[inchikey[:14]]["SPE"]= SPE
            with open(config["e_dict"],'a') as g:
                g.write("{:<30s} {:< 20.8f} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(inchikey,zero_E,H_298,F_298,SPE))

        else:
            print("Error, DFT jobs fails for {}, solve this issue first!".format(inchikey))
            #quit()

    return True,E_dict

# Function to check duplicate TSs 
def select_conf(TSinp_list,package='Gaussian'):
    
    # Initialize conf dict
    conf_dict = {}
    keep_list = []

    # check job status, if TS successfully found, put into IRC list
    for TSinp in TSinp_list:

        # return file name
        pname = TSinp.split('/')[-1].split('-TS')[0]
        product_ind = '_'.join(pname.split('_')[:-1])

        if package.lower() == 'gaussian':
            # change .gjf to .out will return TS geo-opt output file
            TSout = TSinp.replace('.gjf','.out')

            # imag_flag refers whether there is an imaginary frequency in TS output file; finish_flag refers to whether TS geo-opt normally finished
            if os.path.isfile(TSout) is False:
                print("Missing low level Berny optimization {}, skip...".format(TSout))
                continue

            finish_flag,imag_flag,SPE,zero_E,H_298,F_298,_ = read_Gaussian_output(TSout)

        elif package.lower() == 'orca':
            # change .in to .out will return TS geo-opt output file
            TSout = TSinp.replace('.in','.out')
            # imag_flag refers whether there is an imaginary frequency in TS output file; finish_flag refers to whether TS geo-opt normally finished
            finish_flag,imag_flag,SPE,zero_E,H_298,F_298,_ = read_Orca_output(TSout)
            
        if imag_flag and finish_flag:
            if product_ind not in conf_dict.keys():
                conf_dict[product_ind] = [F_298]
                keep_list += [TSinp]
            else:
                min_diff = [abs(i-F_298) for i in conf_dict[product_ind]]
                if min(min_diff) > 1e-5: 
                    conf_dict[product_ind] += [F_298]
                    keep_list += [TSinp]

    return keep_list

# Function to take in reactant and product geometry and perform a pre joint-optimization for preparing pyGSM inputs
def pre_align(E,qt,unpair,RG,PG,Radj_mat=None,Padj_mat=None,working_folder='.',ff='mmff94',Rname=None,Pname=None,model=None):

    # calculate adj_mat if not given
    if Radj_mat is None:
        Radj_mat = Table_generator(E,RG)

    if Padj_mat is None:
        Padj_mat = Table_generator(E,PG)

    # create opt and xTB folder if is not exist
    if os.path.isdir(working_folder+'/opt-folder') is False:
        os.mkdir(working_folder+'/opt-folder')

    if os.path.isdir(working_folder+'/xTB-folder') is False:
        os.mkdir(working_folder+'/xTB-folder')

    if os.path.isdir(working_folder+'/input_files_conf') is False:
        os.mkdir(working_folder+'/input_files_conf')

    # Determine the inchikey before xTB geo-opt
    oinchi = return_inchikey(E,PG)
    if Pname is None:
        Pname = oinchi[:14]
    if Rname is None:
        Rname = Pname+'_R'

    # Apply xtb geo-opt on the product
    product_opt = working_folder+'/opt-folder/{}-opt.xyz'.format(Pname)
    xyz_write(product_opt,E,PG)
    Energy,opted_geo,finish = xtb_geo_opt(product_opt,charge=qt,unpair=unpair,namespace=Pname,workdir=working_folder+'/xTB-folder',level='normal',output_xyz=product_opt,cleanup=False)

    # If geo-opt fails to converge, skip this product...
    if not finish: 
        return False
                                
    # Determine the inchikey of the product after xTB geo-opt
    E,PG  = xyz_parse(opted_geo)
    ninchi= return_inchikey(E,PG)

    # Check whether geo-opt changes the product. If so, geo-opt fails
    if oinchi[:14] != ninchi[:14]:
        return False
    
    # Determine the hash list of this product
    _,_,Phash_list=canon_geo(E,Padj_mat)
    
    # take the product geometry and apply ff-opt to regenerate reactant geo
    new_G = opt_geo(PG,Radj_mat,E,ff=ff,step=500)
    reactant_opt = working_folder+'/xTB-folder/{}-opt.xyz'.format(Rname)
    xyz_write(reactant_opt,E,new_G)

    # Apply ase minimize_rotation_and_translation to optinize the reaction pathway
    reactant= io.read(reactant_opt)
    product = io.read(opted_geo)
    minimize_rotation_and_translation(reactant,product)
    io.write(opted_geo,product)

    # check the intended score of this alignment
    if model is not None:
        RCSmodel = pickle.load(open(model, 'rb'))
        E,RG  = xyz_parse(reactant_opt)
        _,PG  = xyz_parse(opted_geo)
        indicators = return_indicator(E,RG,PG,namespace=Rname)
        if RCSmodel.predict_proba([indicators])[0][1] > 0.4:
            # cat reactant and product together
            command_line = "cd {}/input_files_conf; mv {} {}_0.xyz; cat {} >> {}_0.xyz".format(working_folder,reactant_opt,Pname,opted_geo,Pname)
            os.system(command_line)
    else:
        # cat reactant and product together
        command_line = "cd {}/input_files_conf; mv {} {}_0.xyz; cat {} >> {}_0.xyz".format(working_folder,reactant_opt,Pname,opted_geo,Pname)
        os.system(command_line)
    
    return True

# Function that take smile string and return inchikey
def smiles2inchikey(smiles,ff='mmff94',steps=100):
    
    # load in rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        # construct rdkir object
        m = Chem.MolFromSmiles(smiles)
        m2= Chem.AddHs(m)
        AllChem.EmbedMolecule(m2)
        q = 0

        # parse mol file and obtain E & G
        lines = Chem.MolToMolBlock(m2)

        # create a temporary molfile
        tmp_filename = '.tmp.mol'
        with open(tmp_filename,'w') as g: g.write(lines)

        # apply force-field optimization
        try:
            command = 'obabel {} -O result.xyz --sd --minimize --steps {} --ff {}'.format(tmp_filename,steps,ff)
            output = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,bufsize=-1).communicate()[1].decode('utf8')
            E,G = xyz_parse("result.xyz")

        except:
            command = 'obabel {} -O result.xyz --sd --minimize --steps {} --ff uff'.format(tmp_filename,steps)
            output = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,bufsize=-1).communicate()[1].decode('utf8')
            E,G = xyz_parse("result.xyz")

        # return inchikey
        inchikey = return_inchikey(E,G,separate=True)

        # Remove the tmp file that was read by obminimize
        try:
            os.remove(tmp_filename)
            os.remove("result.xyz")

        except:
            pass

        return inchikey

    except: 

        return smiles

# function to find indicators for reactant-product alignments
# Input: 
#         E:   elements
#         RG:  reactant geometry 
#         RG:  reactant geometry 
# Output:
#     RMSD: mass-weighted RMSD between reactant and product, threshold < 1.6
#     max_dis:  maximum bond length change between non-H atoms, threshold < 4.0
#     min_cross_dis: shorted distance between atoms' path (non-H atoms) to original bonds, threshold > 0.6
#     path_cross: if all atoms involved in bond changes are non-H, path_cross refers to the distance between two bond changes, threshold > 0.6 
#     max_Hdis: maximum bond length change if contains H, threshold < 4.5 (* optional)
#     min_Hcross_dis: shorted distance between atoms' path (H atoms involves) to original bonds, threshold > 0.4 (* optional)
#     h = RMSD/1.6 + max_dis/4.0 + 0.6/min_cross_dis + 0.6/path_cross + 0.5 * max_Hdis/4.5 + 0.1/min_cross_dis
#
def return_indicator(E,RG,PG,namespace='node'):

    # calculate adj_mat
    Radj=Table_generator(E, RG)
    Padj=Table_generator(E, PG)
    
    # determine bond changes
    bond_break, bond_form=[], []
    del_adj = Padj - Radj
    for i in range(len(E)):
        for j in range(i+1, len(E)):
            if del_adj[i][j]==-1: bond_break+=[(i, j)]
            if del_adj[i][j]==1: bond_form+=[(i, j)]

    # identify hydrogen atoms, atoms involved in the reactions
    H_index=[i for i, e in enumerate(E) if e=='H']
    involve=list(set(list(sum(bond_break+bond_form, ()))))

    # create observed segments
    bond_seg={i:[] for i in bond_break+bond_form}
    for bond in bond_break:
        bond_seg[bond]=(PG[bond[1]]-PG[bond[0]], np.linalg.norm(PG[bond[1]]-PG[bond[0]]))
    for bond in bond_form:
        bond_seg[bond]=(RG[bond[1]]-RG[bond[0]], np.linalg.norm(RG[bond[1]]-RG[bond[0]]))

    # create bond list to check cross
    bond_dict={i: [] for i in bond_break+bond_form}
    for i in range(len(E)):
        for j in range(i+1, len(E)):
            for bond in bond_break:
                if Padj[i][j]>0 and i not in bond and j not in bond: bond_dict[bond]+=[(i, j)]
            for bond in bond_form:
                if Radj[i][j]>0 and i not in bond and j not in bond: bond_dict[bond]+=[(i, j)]

    # Compute indicator
    rmsd = return_RMSD(E,RG,PG,rotate=False,mass_weighted=True,namespace=namespace)
    Hbond_dis = np.array([i[1] for bond,i in bond_seg.items() if (bond[0] in H_index or bond[1] in H_index)])
    bond_dis  = np.array([i[1] for bond,i in bond_seg.items() if (bond[0] not in H_index and bond[1] not in H_index)])
    if len(Hbond_dis)>0: 
        max_Hdis=max(Hbond_dis)
    else: 
        max_Hdis=2.0
    if len(bond_dis)>0: 
        max_dis=max(bond_dis)
    else: 
        max_dis=2.0

    # Compute "cross" behaviour
    min_cross, min_Hcross=[], []
    for bond in bond_break:
        cross_dis=[]
        for ibond in bond_dict[bond]:
            _,_,dis=closestDistanceBetweenLines(PG[bond[0]], PG[bond[1]], PG[ibond[0]], PG[ibond[1]])
            cross_dis+=[dis]
        if len(cross_dis)>0: 
            min_dis=min(cross_dis)
        else: 
            min_dis=2.0

        if bond[0] in H_index or bond[1] in H_index: 
            min_Hcross+=[min_dis]
        else: 
            min_cross+=[min_dis]

    for bond in bond_form:
        cross_dis=[]
        for ibond in bond_dict[bond]:
            _,_,dis=closestDistanceBetweenLines(RG[bond[0]], RG[bond[1]], RG[ibond[0]], RG[ibond[1]])
            cross_dis+=[dis]
        if len(cross_dis) > 0: 
            min_dis=min(cross_dis)
        else: 
            min_dis=2.0
        if bond[0] in H_index or bond[1] in H_index: 
            min_Hcross+=[min_dis]
        else: 
            min_cross+=[min_dis]

    # Find the smallest bonds distance for each bond, if None, return 2.0
    if len(min_cross) > 0:
        min_cross_dis = min(min_cross)
    else:
        min_cross_dis = 2.0

    if len(min_Hcross) > 0:
        min_Hcross_dis = min(min_Hcross)
    else:
        min_Hcross_dis = 2.0

    # Find the cross distanc ebetween two bond changes
    if len([ind for ind in involve if ind in H_index]) ==0:

        if len(bond_break) == 2:
            _,_,dis = closestDistanceBetweenLines(PG[bond_break[0][0]],PG[bond_break[0][1]],PG[bond_break[1][0]],PG[bond_break[1][1]],clampAll=True)
        else:
            dis = 2.0
        path_cross = dis

        if len(bond_form) == 2:
            _,_,dis = closestDistanceBetweenLines(RG[bond_form[0][0]],RG[bond_form[0][1]],RG[bond_form[1][0]],RG[bond_form[1][1]],clampAll=True)
        else:
            dis = 2.0
        path_cross = min(dis,path_cross)

    else:
        path_cross = 2.0

    return [rmsd, max_dis, max_Hdis, min_cross_dis, min_Hcross_dis, path_cross]

# Function to calculate spatial distance between two segments
def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=True,clampA0=False,clampA1=False,clampB0=False,clampB1=False):

    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    '''

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
   
    _A = A / magA
    _B = B / magB
   
    cross = np.cross(_A, _B);
    denom = np.linalg.norm(cross)**2
   
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A,(b1-a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0,b0,np.linalg.norm(a0-b0)
                    return a0,b1,np.linalg.norm(a0-b1)


            # Is segment B after A?
        elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1)


        # Segments overlap, return distance between parallel segments
        return None,None,np.linalg.norm(((d0*_A)+a0)-b0)

    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0);
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom;
    t1 = detB/denom;

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA,pB,np.linalg.norm(pA-pB)

