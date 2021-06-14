from __future__ import print_function
import numpy as np
from ase.units import Bohr
from ase.utils import basestring

def attach_charges(atoms, fileobj='ACF.dat', displacement=1e-4):
    """Attach the charges from the fileobj to the Atoms."""
    if isinstance(fileobj, basestring):
        fileobj = open(fileobj)

    sep = '---------------'
    i = 0 # Counter for the lines
    k = 0 # Counter of sep
    assume6columns = False
    for line in fileobj:
        if line[0] == '\n': # check if there is an empty line in the 
            i -= 1          # head of ACF.dat file
        if i == 0:
            headings = line
            if 'BADER' in headings.split():
                j = headings.split().index('BADER')
            elif 'CHARGE' in headings.split():
                j = headings.split().index('CHARGE')
            else:
                print('Can\'t find keyword "BADER" or "CHARGE".' \
                +' Assuming the ACF.dat file has 6 columns.')
                j = 4
                assume6columns = True
        if sep in line: # Stop at last seperator line
            if k == 1:
                break
            k += 1
        if not i > 1:
            pass
        else:
            words = line.split()
            if assume6columns is True:
                if len(words) != 6:
                    raise IOError('Number of columns in ACF file incorrect!\n'
                                  'Check that Bader program version >= 0.25')
                
            atom = atoms[int(words[0]) - 1]
            atom.charge = float(words[j])
            if displacement is not None: # check if the atom positions match
                xyz = np.array([float(w) for w in words[1:4]])
                # ACF.dat units could be Bohr or Angstrom
                norm1 = np.linalg.norm(atom.position - xyz)
                norm2 = np.linalg.norm(atom.position - xyz*Bohr)
                assert norm1 < displacement or norm2 < displacement
        i += 1

