from __future__ import division
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
from random import randint
import numpy as np

heavy_test = False


def get_atoms_with_mixed_elements(crystalstructure="fcc"):
    atoms = bulk("Al", crystalstructure=crystalstructure, a=3.2)
    atoms = atoms * (2, 2, 2)
    symbs = ["Al", "Cu", "Zn"]
    symbols = [symbs[randint(0, len(symbs) - 1)] for _ in range(len(atoms))]
    for i in range(len(atoms)):
        atoms[i].symbol = symbols[i]
    return atoms


def test_compare(comparator):
    s1 = bulk("Al")
    s1 = s1 * (2, 2, 2)
    s2 = bulk("Al")
    s2 = s2 * (2, 2, 2)
    assert comparator.compare(s1, s2)


def test_fcc_bcc(comparator):
    s1 = bulk("Al", crystalstructure="fcc")
    s2 = bulk("Al", crystalstructure="bcc", a=4.05)
    s1 = s1 * (2, 2, 2)
    s2 = s2 * (2, 2, 2)
    assert not comparator.compare(s1, s2)


def test_single_impurity(comparator):
    s1 = bulk("Al")
    s1 = s1 * (2, 2, 2)
    s1[0].symbol = "Mg"
    s2 = bulk("Al")
    s2 = s2 * (2, 2, 2)
    s2[3].symbol = "Mg"
    assert comparator.compare(s1, s2)


def test_translations(comparator):
    s1 = get_atoms_with_mixed_elements()
    s2 = s1.copy()

    xmax = 2.0 * np.max(s1.get_cell().T)
    N = 3
    dx = xmax / N
    pos_ref = s2.get_positions()
    structures = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                displacement = np.array([dx * i, dx * j, dx * k])
                new_pos = pos_ref + displacement
                s2.set_positions(new_pos)
                structures.append(s2)
    assert comparator.compare(s1, structures)


def test_rot_60_deg(comparator):
    s1 = get_atoms_with_mixed_elements()
    s2 = s1.copy()
    ca = np.cos(np.pi / 3.0)
    sa = np.sin(np.pi / 3.0)
    matrix = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
    s2.set_positions(matrix.dot(s2.get_positions().T).T)
    s2.set_cell(matrix.dot(s2.get_cell().T).T)
    assert comparator.compare(s1, s2)


def test_rot_120_deg(comparator):
    s1 = get_atoms_with_mixed_elements()
    s2 = s1.copy()
    ca = np.cos(2.0 * np.pi / 3.0)
    sa = np.sin(2.0 * np.pi / 3.0)
    matrix = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
    s2.set_positions(matrix.dot(s2.get_positions().T).T)
    s2.set_cell(matrix.dot(s2.get_cell().T).T)
    assert comparator.compare(s1, s2)


def test_rotations_to_standard(comparator):
    s1 = Atoms("Al")
    tol = 1E-6
    num_tests = 4
    if heavy_test:
        num_tests = 20
    for _ in range(num_tests):
        cell = np.random.rand(3, 3) * 4.0 - 4.0
        s1.set_cell(cell)
        new_cell = comparator._standarize_cell(s1).get_cell().T
        assert abs(new_cell[1, 0]) < tol
        assert abs(new_cell[2, 0]) < tol
        assert abs(new_cell[2, 1]) < tol


def test_point_inversion(comparator):
    s1 = get_atoms_with_mixed_elements()
    s2 = s1.copy()
    s2.set_positions(-s2.get_positions())
    assert comparator.compare(s1, s2)


def test_mirror_plane(comparator):
    s1 = get_atoms_with_mixed_elements(crystalstructure="hcp")
    s2 = s1.copy()
    mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    s2.set_positions(mat.dot(s2.get_positions().T).T)
    assert comparator.compare(s1, s2)

    mat = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    s2.set_positions(mat.dot(s1.get_positions().T).T)
    assert comparator.compare(s1, s2)

    mat = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    s2.set_positions(mat.dot(s1.get_positions().T).T)
    assert comparator.compare(s1, s2)


def test_hcp_symmetry_ops(comparator):
    s1 = get_atoms_with_mixed_elements(crystalstructure="hcp")
    s2 = s1.copy()
    sg = spacegroup.Spacegroup(194)
    cell = s2.get_cell().T
    inv_cell = np.linalg.inv(cell)
    operations = sg.get_rotations()
    if not heavy_test:
        operations = operations[::int(np.ceil(len(operations) / 4))]
    for op in operations:
        s1 = get_atoms_with_mixed_elements(crystalstructure="hcp")
        s2 = s1.copy()
        transformed_op = cell.dot(op).dot(inv_cell)
        s2.set_positions(transformed_op.dot(s1.get_positions().T).T)
        assert comparator.compare(s1, s2)


def test_fcc_symmetry_ops(comparator):
    s1 = get_atoms_with_mixed_elements()
    s2 = s1.copy()
    sg = spacegroup.Spacegroup(225)
    cell = s2.get_cell().T
    inv_cell = np.linalg.inv(cell)
    operations = sg.get_rotations()
    if not heavy_test:
        operations = operations[::int(np.ceil(len(operations) / 4))]
    for op in operations:
        s1 = get_atoms_with_mixed_elements()
        s2 = s1.copy()
        transformed_op = cell.dot(op).dot(inv_cell)
        s2.set_positions(transformed_op.dot(s1.get_positions().T).T)
        assert comparator.compare(s1, s2)


def test_bcc_symmetry_ops(comparator):
    s1 = get_atoms_with_mixed_elements(crystalstructure="bcc")
    s2 = s1.copy()
    sg = spacegroup.Spacegroup(229)
    cell = s2.get_cell().T
    inv_cell = np.linalg.inv(cell)
    operations = sg.get_rotations()
    if not heavy_test:
        operations = operations[::int(np.ceil(len(operations) / 4))]
    for op in operations:
        s1 = get_atoms_with_mixed_elements(crystalstructure="bcc")
        s2 = s1.copy()
        transformed_op = cell.dot(op).dot(inv_cell)
        s2.set_positions(transformed_op.dot(s1.get_positions().T).T)
        assert comparator.compare(s1, s2)


def test_bcc_translation(comparator):
    s1 = get_atoms_with_mixed_elements(crystalstructure="bcc")
    s2 = s1.copy()
    s2.set_positions(s2.get_positions() + np.array([6.0, -2.0, 1.0]))
    assert comparator.compare(s1, s2)


def test_one_atom_out_of_pos(comparator):
    s1 = get_atoms_with_mixed_elements()
    s2 = s1.copy()
    pos = s1.get_positions()
    pos[0, :] += 0.2
    s2.set_positions(pos)
    assert not comparator.compare(s1, s2)


def test_reduce_to_primitive(comparator):
    atoms1 = crystal(symbols=['V', 'Li', 'O'],
                     basis=[(0.000000, 0.000000, 0.000000),
                            (0.333333, 0.666667, 0.000000),
                            (0.333333, 0.000000, 0.250000)],
                     spacegroup=167,
                     cellpar=[5.123, 5.123, 13.005, 90., 90., 120.],
                     size=[1, 1, 1], primitive_cell=False)

    atoms2 = crystal(symbols=['V', 'Li', 'O'],
                     basis=[(0.000000, 0.000000, 0.000000),
                            (0.333333, 0.666667, 0.000000),
                            (0.333333, 0.000000, 0.250000)],
                     spacegroup=167,
                     cellpar=[5.123, 5.123, 13.005, 90., 90., 120.],
                     size=[1, 1, 1], primitive_cell=True)
    try:
        # Tell the comparator to reduce to primitive cell
        comparator.to_primitive = True

        assert comparator.compare(atoms1, atoms2)
    except SpgLibNotFoundError:
        pass

    # Reset the comparator to its original state
    comparator.to_primitive = False


def test_order_of_candidates(comparator):
    s1 = bulk("Al", crystalstructure='fcc', a=3.2)
    s1 = s1 * (2, 2, 2)
    s2 = s1.copy()
    s1.positions[0, :] += .2

    assert comparator.compare(s2, s1) == comparator.compare(s1, s2)


def test_original_paper_structures():
    # Structures from the original paper:
    # Comput. Phys. Commun. 183, 690-697 (2012)
    # They should evaluate equal (within a certain tolerance)
    syms = ['O', 'O', 'Mg', 'F']
    cell1 = [(3.16, 0.00, 0.00), (-0.95, 4.14, 0.00), (-0.95, -0.22, 4.13)]
    p1 = [(0.44, 0.40, 0.30), (0.94, 0.40, 0.79),
          (0.45, 0.90, 0.79), (0.94, 0.40, 0.29)]
    s1 = Atoms(syms, cell=cell1, scaled_positions=p1, pbc=True)

    cell2 = [(6.00, 0.00, 0.00), (1.00, 3.00, 0.00), (2.00, -3.00, 3.00)]
    p2 = [(0.00, 0.00, 0.00), (0.00, 0.00, 0.50),
          (0.50, 0.00, 0.00), (0.00, 0.50, 0.00)]
    s2 = Atoms(syms, cell=cell2, scaled_positions=p2, pbc=True)

    comp = SymmetryEquivalenceCheck()

    assert comp.compare(s1, s2)
    assert comp.compare(s2, s1) == comp.compare(s1, s2)


def test_symmetrical_one_element_out(comparator):
    s1 = get_atoms_with_mixed_elements()
    s1.set_chemical_symbols(['Zn', 'Zn', 'Al', 'Zn', 'Zn', 'Al', 'Zn', 'Zn'])
    s2 = s1.copy()
    s2.positions[0, :] += 0.2
    assert not comparator.compare(s1, s2)
    assert not comparator.compare(s2, s1)


def test_one_vs_many():
    s1 = Atoms('H3', positions=[[0.5, 0.5, 0], [0.5, 1.5, 0], [1.5, 1.5, 0]],
               cell=[2, 2, 2], pbc=True)
    # Get the unit used for position comparison
    u = (s1.get_volume() / len(s1))**(1 / 3)
    comp = SymmetryEquivalenceCheck(stol=.095 / u, scale_volume=True)
    s2 = s1.copy()
    assert comp.compare(s1, s2)
    s2_list = []
    s3 = Atoms('H3', positions=[[0.5, 0.5, 0], [0.5, 1.5, 0], [1.5, 1.5, 0]],
               cell=[3, 3, 3], pbc=True)
    s2_list.append(s3)
    for d in np.linspace(0.1, 1.0, 5):
        s2 = s1.copy()
        s2.positions[0] += [d, 0, 0]
        s2_list.append(s2)
    assert not comp.compare(s1, s2_list[:-1])
    assert comp.compare(s1, s2_list)


def run_all_tests(comparator):
    test_compare(comparator)
    test_fcc_bcc(comparator)
    test_single_impurity(comparator)
    test_translations(comparator)
    test_rot_60_deg(comparator)
    test_rot_120_deg(comparator)
    test_rotations_to_standard(comparator)
    test_point_inversion(comparator)
    test_mirror_plane(comparator)
    test_hcp_symmetry_ops(comparator)
    test_fcc_symmetry_ops(comparator)
    test_bcc_symmetry_ops(comparator)
    test_bcc_translation(comparator)
    test_one_atom_out_of_pos(comparator)
    test_reduce_to_primitive(comparator)
    test_order_of_candidates(comparator)
    test_one_vs_many()
    test_original_paper_structures()


comparator = SymmetryEquivalenceCheck()
run_all_tests(comparator)
