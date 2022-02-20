import openbabel as ob
import numpy as np


# Standard openbabel molecule load
conv = ob.OBConversion()
conv.SetInAndOutFormats('xyz','xyz')
mol = ob.OBMol()

''' test on ff-opt with constraint '''
conv.ReadFile(mol,'../ERS_enumeration/Reactant/Ga_ethyl_ethene.xyz')
#conv.ReadFile(mol,'sixth_template/pp_4_0-end.xyz')

# Define constraints
constraints= ob.OBFFConstraints()
fixed_list = np.arange(1,22)
#for atom in fixed_list:
#    constraints.AddAtomConstraint(atom)

notfixed_list = [7,9,13,14,15,16,17,18,19,20,21,22]
for atom in fixed_list:
    if atom not in notfixed_list:
        constraints.AddAtomConstraint(int(atom))

# Setup the force field with the constraints
forcefield = ob.OBForceField.FindForceField("uff")
forcefield.Setup(mol, constraints)
forcefield.SetConstraints(constraints)

# Do a 500 steps conjugate gradient minimiazation
# and save the coordinates to mol.
forcefield.ConjugateGradients(50)
forcefield.GetCoordinates(mol)

# Write the mol to a file
conv.WriteFile(mol,'TS_opt.xyz')
#'''

