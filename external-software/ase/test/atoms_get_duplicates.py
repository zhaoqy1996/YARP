from ase import Atoms
from ase.geometry import get_duplicate_atoms

at = Atoms('H5', positions=[[0., 0., 0.],
                            [1., 0., 0.],
                            [1.01, 0, 0],
                            [3, 2.2, 5.2],
                            [0.1, -0.01, 0.1]])

dups = get_duplicate_atoms(at)
assert all((dups == [[1, 2]]).tolist()) is True

dups = get_duplicate_atoms(at, cutoff=0.2)
assert all((dups == [[0, 4], [1, 2]]).tolist()) is True

get_duplicate_atoms(at, delete=True)
assert len(at) == 4

at = Atoms('H3', positions=[[0., 0., 0.],
                            [1., 0., 0.],
                            [3, 2.2, 5.2]])

# test if it works if no duplicates are detected.
get_duplicate_atoms(at, delete=True)
dups = get_duplicate_atoms(at)

assert dups.size == 0
