import sys,os
from copy import deepcopy
import pickle,json

def main(argv):
    
    smiles_list = []

    # load in reaction db
    with open("reactants_b2f2.p",'rb') as f:
        R=pickle.load(f)

    with open("/depot/bsavoie/data/TCIT/TCIT_result.json","r") as f:
        TCIT_dict = json.load(f)

    for inchi_ind,Ri in R.items():
        R_Hf_298 = sum([TCIT_dict[smi]["Hf_298"] for smi in Ri['prop']['smiles'].split('.')])
        if abs(R_Hf_298-Ri['prop']['TCIT_Hf']) > 0.1: print("TCIT prediction for reactant {} is wrong".format(inchi_ind))
        for pind,pp in Ri['possible_products'].items():
            try:
                P_Hf_298 = sum([TCIT_dict[smi]["Hf_298"] for smi in pp["name"].split('.')])
                if abs(P_Hf_298-pp["TCIT_Hf"]) > 0.1:
                    print(P_Hf_298,R_Hf_298,inchi_ind,pind,pp["name"])   
                    pp["TCIT_Hf"] = deepcopy(P_Hf_298)
                    #if P_Hf_298-R_Hf_298 < 80:print(P_Hf_298,R_Hf_298,inchi_ind,pind,pp["name"])
            except:
                print("Missing TCIT predictions for {}".format(pp["name"]))
    with open("reactants_b2f2.p","wb") as f:
        pickle.dump(R, f, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    for inchi_ind,Ri in R.items():
        smiles_list += [Ri['prop']['smiles']]
        for _,pp in Ri['possible_products'].items():
            smiles_list += pp["name"].split('.')
            
    with open('inp_smiles.txt','w') as f:
        for i in list(set(smiles_list)):
            f.write('{}\n'.format(i))
    '''
    #print(list(set(smiles_list)))
    return

# Function to load in energy database
def parse_Energy(db_files,E_dict={}):
    with open(db_files,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if lc == 0: continue
            if len(fields) ==0: continue
            if len(fields) == 4:
                if fields[0] not in E_dict.keys():
                    E_dict[fields[0]] = {}
                    E_dict[fields[0]]["E_0"]= float(fields[1])
                    E_dict[fields[0]]["H"]  = float(fields[2])
                    E_dict[fields[0]]["F"]  = float(fields[3])

    return E_dict
 
# Function to determine energy barrier height
def calculate_BH(energy_db,target,reaction,level='6-31+G*'):

    from rdkit import Chem 

    # create similarity match dictionary
    similar_match = {}
    for inchi in energy_db.keys():
        similar_match[inchi[:14]]=inchi

    # initialize barrier list
    barrier_list = []

    for _,TS in reaction['TS'].items():
    
        # first check level
        if TS['level'] != level:
            continue

        G_r = 0
        inchi_list = [[smiles2inchikey(smi)[:14] for smi in smiles.split('.')] for smiles in TS['smiles_index(full)']]
        if target in inchi_list[0]:
            R_inchi = inchi_list[0]
        elif target in inchi_list[1]:
            R_inchi = inchi_list[1]
        else:
            return 0,False

        for inchi in R_inchi:
            if inchi not in energy_db.keys() and inchi[:14] in similar_match.keys():
                inchi = similar_match[inchi[:14]]
    
            if inchi in energy_db.keys():
                G_r += energy_db[inchi]["F"]

            else:
                print("DFT energy for {} is missing".format(inchi))
                return 0,False

        # get free energy of TS
        BH  = (TS['G_TS'] - G_r) * 627.5
        barrier_list += [BH]

    return barrier_list,True

def smiles2inchikey(smiles):

    # load in rdkit 
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdinchi import MolToInchiKey

    return MolToInchiKey(MolFromSmiles(smiles))

if __name__ == "__main__":
    main(sys.argv[1:])

