import sys,os
import pickle

def main(argv):
    
    # load in reaction db
    with open("/depot/bsavoie/data/YARP/reaction/M052xD3_def2SVP.p",'rb') as f:
        reaction=pickle.load(f)
    print("Currently the reaction database contain {} reactions".format(len(reaction.keys())))

    # load in DFT db
    E_dict=parse_Energy('/depot/bsavoie/data/YARP/DFT-db/M052xD3_def2SVP.db')

    # find target involved reactions 
    target   = argv[0]
    inchi    = smiles2inchikey(target)[:14]
    involved = [pathway for pathway in reaction.keys() if (pathway[0].split('-')+pathway[1].split('-')).count(inchi) == 1]  

    # return energy barrier
    for i in involved:

        TS,report = calculate_BH(E_dict,inchi,reaction[i])
        smiles_index = reaction[i]['smiles_index']
        if report:
            print("Reaction pathway {} with energy barrier {} kcal/mol".format(smiles_index,TS))
        
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
def calculate_BH(energy_db,target,reaction):

    from rdkit import Chem 

    # create similarity match dictionary
    similar_match = {}
    for inchi in energy_db.keys():
        similar_match[inchi[:14]]=inchi

    # initialize barrier list
    barrier_list = []

    for _,TS in reaction['TS'].items():
    
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
        BH  = (TS['TS_DG'] - G_r) * 627.5
        barrier_list += [BH]

    return barrier_list,True

def smiles2inchikey(smiles):

    # load in rdkit 
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdinchi import MolToInchiKey

    return MolToInchiKey(MolFromSmiles(smiles))

if __name__ == "__main__":
    main(sys.argv[1:])

