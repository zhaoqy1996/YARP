from ase.calculators.calculator import ReadError
import os.path as op
import numpy as np
from ase.units import Hartree

def parse_xray(filename):
    #filename = self.label + '/deMon.xry'
    if op.isfile(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        mode = lines[0].split()[0]
        ntrans = int(lines[0].split()[1])

        E_trans = []
        osc_strength = []
        trans_dip = []
        for i in range(1, ntrans + 1):
            tokens = lines[i].split()            

            E_trans.append(float(tokens[0]))
            osc_strength.append(
                float(tokens[1].replace('D', 'e')))
            
            dip1 = float(tokens[3].replace('D', 'e'))
            dip2 = float(tokens[4].replace('D', 'e'))
            dip3 = float(tokens[5].replace('D', 'e'))
            trans_dip.append([dip1, dip2, dip3])

        return mode, ntrans, np.array(E_trans) * Hartree, np.array(osc_strength), np.array(trans_dip)
    
    else:
        raise ReadError('The file {0} does not exist'
                        .format(filename))
