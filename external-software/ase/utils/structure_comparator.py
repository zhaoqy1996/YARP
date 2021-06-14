"""Determine symmetry equivalence of two structures.
Based on the recipe from Comput. Phys. Commun. 183, 690-697 (2012)."""
from collections import Counter
from itertools import combinations, product
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce

def normalize(cell):
    for i in range(3):
        cell[i] /= np.linalg.norm(cell[i])


try:
    from itertools import filterfalse
except ImportError:  # python2.7
    from itertools import ifilterfalse as filterfalse


class SpgLibNotFoundError(Exception):
    """Raised if SPG lib is not found when needed."""

    def __init__(self, msg):
        super(SpgLibNotFoundError, self).__init__(msg)


class SymmetryEquivalenceCheck(object):
    """Compare two structures to determine if they are symmetry equivalent.

    Based on the recipe from Comput. Phys. Commun. 183, 690-697 (2012).

    Parameters:

    angle_tol: float
        angle tolerance for the lattice vectors in degrees

    ltol: float
        relative tolerance for the length of the lattice vectors (per atom)

    stol: float
        position tolerance for the site comparison in units of
        (V/N)^(1/3) (average length between atoms)

    vol_tol: float
        volume tolerance in angstrom cubed to compare the volumes of
        the two structures

    scale_volume: bool
        if True the volumes of the two structures are scaled to be equal

    to_primitive: bool
        if True the structures are reduced to their primitive cells
        note that this feature requires spglib to installed

    Examples:

    >>> from ase.build import bulk
    >>> from ase.utils.structure_comparator import SymmetryEquivalenceCheck
    >>> comp = SymmetryEquivalenceCheck()

    Compare a cell with a rotated version

    >>> a = bulk('Al', orthorhombic=True)
    >>> b = a.copy()
    >>> b.rotate(60, 'x', rotate_cell=True)
    >>> comp.compare(a, b)
    True

    Transform to the primitive cell and then compare

    >>> pa = bulk('Al')
    >>> comp.compare(a, pa)
    False
    >>> comp = SymmetryEquivalenceCheck(to_primitive=True)
    >>> comp.compare(a, pa)
    True

    Compare one structure with a list of other structures

    >>> import numpy as np
    >>> from ase import Atoms
    >>> s1 = Atoms('H3', positions=[[0.5, 0.5, 0],
    ...                             [0.5, 1.5, 0],
    ...                             [1.5, 1.5, 0]],
    ...            cell=[2, 2, 2], pbc=True)
    >>> comp = SymmetryEquivalenceCheck(stol=0.068)
    >>> s2_list = []
    >>> for d in np.linspace(0.1, 1.0, 5):
    ...     s2 = s1.copy()
    ...     s2.positions[0] += [d, 0, 0]
    ...     s2_list.append(s2)
    >>> comp.compare(s1, s2_list[:-1])
    False
    >>> comp.compare(s1, s2_list)
    True

    """

    def __init__(self, angle_tol=1.0, ltol=0.05, stol=0.05, vol_tol=0.1,
                 scale_volume=False, to_primitive=False):
        self.angle_tol = angle_tol * np.pi / 180.0  # convert to radians
        self.scale_volume = scale_volume
        self.stol = stol
        self.ltol = ltol
        self.vol_tol = vol_tol
        self.position_tolerance = 0.0
        self.to_primitive = to_primitive

        # Variables to be used in the compare function
        self.s1 = None
        self.s2 = None
        self.expanded_s1 = None
        self.expanded_s2 = None
        self.least_freq_element = None

    def _niggli_reduce(self, atoms):
        """Reduce to niggli cells.

        Reduce the atoms to niggli cells, then rotates the niggli cells to
        the so called "standard" orientation with one lattice vector along the
        x-axis and a second vector in the xy plane.
        """
        niggli_reduce(atoms)
        self._standarize_cell(atoms)

    def _standarize_cell(self, atoms):
        """Rotate the first vector such that it points along the x-axis.
        Then rotate around the first vector so the second vector is in the
        xy plane.
        """
        # Rotate first vector to x axis
        cell = atoms.get_cell().T
        total_rot_mat = np.eye(3)
        v1 = cell[:, 0]
        l1 = np.sqrt(v1[0]**2 + v1[2]**2)
        angle = np.abs(np.arcsin(v1[2] / l1))
        if (v1[0] < 0.0 and v1[2] > 0.0):
            angle = np.pi - angle
        elif (v1[0] < 0.0 and v1[2] < 0.0):
            angle = np.pi + angle
        elif (v1[0] > 0.0 and v1[2] < 0.0):
            angle = -angle
        ca = np.cos(angle)
        sa = np.sin(angle)
        rotmat = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
        total_rot_mat = rotmat.dot(total_rot_mat)
        cell = rotmat.dot(cell)

        v1 = cell[:, 0]
        l1 = np.sqrt(v1[0]**2 + v1[1]**2)
        angle = np.abs(np.arcsin(v1[1] / l1))
        if (v1[0] < 0.0 and v1[1] > 0.0):
            angle = np.pi - angle
        elif (v1[0] < 0.0 and v1[1] < 0.0):
            angle = np.pi + angle
        elif (v1[0] > 0.0 and v1[1] < 0.0):
            angle = -angle
        ca = np.cos(angle)
        sa = np.sin(angle)
        rotmat = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
        total_rot_mat = rotmat.dot(total_rot_mat)
        cell = rotmat.dot(cell)

        # Rotate around x axis such that the second vector is in the xy plane
        v2 = cell[:, 1]
        l2 = np.sqrt(v2[1]**2 + v2[2]**2)
        angle = np.abs(np.arcsin(v2[2] / l2))
        if (v2[1] < 0.0 and v2[2] > 0.0):
            angle = np.pi - angle
        elif (v2[1] < 0.0 and v2[2] < 0.0):
            angle = np.pi + angle
        elif (v2[1] > 0.0 and v2[2] < 0.0):
            angle = -angle
        ca = np.cos(angle)
        sa = np.sin(angle)
        rotmat = np.array([[1.0, 0.0, 0.0], [0.0, ca, sa], [0.0, -sa, ca]])
        total_rot_mat = rotmat.dot(total_rot_mat)
        cell = rotmat.dot(cell)

        atoms.set_cell(cell.T)
        atoms.set_positions(total_rot_mat.dot(atoms.get_positions().T).T)
        atoms.wrap(pbc=[1, 1, 1])
        return atoms

    def _get_element_count(self, struct):
        """Count the number of elements in each of the structures."""
        return Counter(struct.numbers)

    def _get_angles(self, cell):
        """Get the internal angles of the unit cell."""
        cell = cell.copy()

        normalize(cell)

        dot = cell.dot(cell.T)

        # Extract only the relevant dot products
        dot = [dot[0, 1], dot[0, 2], dot[1, 2]]

        # Return angles
        return np.arccos(dot)

    def _has_same_elements(self):
        """Check if two structures have same elements."""
        elem1 = self._get_element_count(self.s1)
        return elem1 == self._get_element_count(self.s2)

    def _has_same_angles(self):
        """Check that the Niggli unit vectors has the same internal angles."""
        ang1 = np.sort(self._get_angles(self.s1.get_cell()))
        ang2 = np.sort(self._get_angles(self.s2.get_cell()))

        return np.allclose(ang1, ang2, rtol=0, atol=self.angle_tol)

    def _has_same_volume(self):
        vol1 = self.s1.get_volume()
        vol2 = self.s2.get_volume()
        return np.abs(vol1 - vol2) < self.vol_tol

    def _scale_volumes(self):
        """Scale the cell of s2 to have the same volume as s1."""
        cell2 = self.s2.get_cell()
        # Get the volumes
        v2 = np.linalg.det(cell2)
        v1 = np.linalg.det(self.s1.get_cell())

        # Scale the cells
        coordinate_scaling = (v1 / v2)**(1.0 / 3.0)
        cell2 *= coordinate_scaling
        self.s2.set_cell(cell2, scale_atoms=True)

    def compare(self, s1, s2):
        """Compare the two structures.

        Return *True* if the two structures are equivalent, *False* otherwise.

        Parameters:

        s1: Atoms object.
            Transformation matrices are calculated based on this structure.

        s2: Atoms or list
            s1 can be compared to one structure or many structures supplied in
            a list. If s2 is a list it returns True if any structure in s2
            matches s1, False otherwise.
        """
        if self.to_primitive:
            s1 = self._reduce_to_primitive(s1)
        self._set_least_frequent_element(s1)
        self._least_frequent_element_to_origin(s1)
        self.s1 = s1.copy()
        vol = self.s1.get_volume()
        self.expanded_s1 = None
        s1_niggli_reduced = False

        if isinstance(s2, Atoms):
            # Just make it a list of length 1
            s2 = [s2]

        matrices = None
        translations = None
        for struct in s2:
            self.s2 = struct.copy()
            self.expanded_s2 = None

            if self.to_primitive:
                self.s2 = self._reduce_to_primitive(self.s2)

            # Compare number of elements in structures
            if len(self.s1) != len(self.s2):
                continue

            # Compare chemical formulae
            if not self._has_same_elements():
                continue

            # Compare angles
            if not s1_niggli_reduced:
                self._niggli_reduce(self.s1)
            self._niggli_reduce(self.s2)
            if not self._has_same_angles():
                continue

            # Compare volumes
            if self.scale_volume:
                self._scale_volumes()
            if not self._has_same_volume():
                continue

            if matrices is None:
                matrices, translations = \
                    self._get_rotation_reflection_matrices()

            # After the candidate translation based on s1 has been computed
            # we need potentially to swap s1 and s2 for robust comparison
            self._least_frequent_element_to_origin(self.s2)
            switch = self._switch_reference_struct()

            # Calculate tolerance on positions
            self.position_tolerance = \
                self.stol * (vol / len(self.s2))**(1.0 / 3.0)

            if self._positions_match(matrices, translations):
                return True

            # Set the reference structure back to its original
            self.s1 = s1.copy()
            if switch:
                self.expanded_s1 = self.expanded_s2
        return False

    def _set_least_frequent_element(self, atoms):
        """Save the atomic number of the least frequent element."""
        elem1 = self._get_element_count(atoms)
        self.least_freq_element = elem1.most_common()[-1][0]

    def _get_only_least_frequent_of(self, struct):
        """Get the atoms object with all other elements than the least frequent
        one removed. Wrap the positions to get everything in the cell."""
        pos = struct.get_positions(wrap=True)

        indices = struct.numbers == self.least_freq_element
        least_freq_struct = struct[indices]
        least_freq_struct.set_positions(pos[indices])

        return least_freq_struct

    def _switch_reference_struct(self):
        """There is an intrinsic assymetry in the system because
        one of the atoms are being expanded, while the other is not.
        This can cause the algorithm to return different result
        depending on which structure is passed first.
        We adopt the convention of using the atoms object
        having the fewest atoms in its expanded cell as the
        reference object.
        We return True if a switch of structures has been performed."""

        # First expand the cells
        if self.expanded_s1 is None:
            self.expanded_s1 = self._expand(self.s1)
        if self.expanded_s2 is None:
            self.expanded_s2 = self._expand(self.s2)

        exp1 = self.expanded_s1
        exp2 = self.expanded_s2
        if len(exp1) < len(exp2):
            # s1 should be the reference structure
            # We have to swap s1 and s2
            s1_temp = self.s1.copy()
            self.s1 = self.s2
            self.s2 = s1_temp
            exp1_temp = self.expanded_s1.copy()
            self.expanded_s1 = self.expanded_s2
            self.expanded_s2 = exp1_temp
            return True
        return False

    def _positions_match(self, rotation_reflection_matrices, translations):
        """Check if the position and elements match.

        Note that this function changes self.s1 and self.s2 to the rotation and
        translation that matches best. Hence, it is crucial that this function
        calls the element comparison, not the other way around.
        """
        pos1_ref = self.s1.get_positions(wrap=True)

        # Get the expanded reference object
        exp2 = self.expanded_s2
        # Build a KD tree to enable fast look-up of nearest neighbours
        tree = KDTree(exp2.get_positions())
        for i in range(translations.shape[0]):
            # Translate
            pos1_trans = pos1_ref - translations[i]
            for matrix in rotation_reflection_matrices:
                # Rotate
                pos1 = matrix.dot(pos1_trans.T).T

                # Update the atoms positions
                self.s1.set_positions(pos1)
                self.s1.wrap(pbc=[1, 1, 1])
                if self._elements_match(self.s1, exp2, tree):
                    return True
        return False

    def _expand(self, ref_atoms, tol=0.0001):
        """If an atom is closer to a boundary than tol it is repeated at the
        opposite boundaries.

        This ensures that atoms having crossed the cell boundaries due to
        numerical noise are properly detected.

        The distance between a position and cell boundary is calculated as:
        dot(position, (b_vec x c_vec) / (|b_vec| |c_vec|) ), where x is the
        cross product.
        """
        syms = ref_atoms.get_chemical_symbols()
        cell = ref_atoms.get_cell()
        positions = ref_atoms.get_positions(wrap=True)
        expanded_atoms = ref_atoms.copy()

        # Calculate normal vectors to the unit cell faces
        normal_vectors = np.array([np.cross(cell[1, :], cell[2, :]),
                                   np.cross(cell[0, :], cell[2, :]),
                                   np.cross(cell[0, :], cell[1, :])])
        normalize(normal_vectors)

        # Get the distance to the unit cell faces from each atomic position
        pos2faces = np.abs(positions.dot(normal_vectors.T))

        # And the opposite faces
        pos2oppofaces = np.abs(np.dot(positions - np.sum(cell, axis=0),
                                      normal_vectors.T))

        for i, i2face in enumerate(pos2faces):
            # Append indices for positions close to the other faces
            # and convert to boolean array signifying if the position at
            # index i is close to the faces bordering origo (0, 1, 2) or
            # the opposite faces (3, 4, 5)
            i_close2face = np.append(i2face, pos2oppofaces[i]) < tol
            # For each position i.e. row it holds that
            # 1 x True -> close to face -> 1 extra atom at opposite face
            # 2 x True -> close to edge -> 3 extra atoms at opposite edges
            # 3 x True -> close to corner -> 7 extra atoms opposite corners
            # E.g. to add atoms at all corners we need to use the cell
            # vectors: (a, b, c, a + b, a + c, b + c, a + b + c), we use
            # itertools.combinations to get them all
            for j in range(sum(i_close2face)):
                for c in combinations(np.nonzero(i_close2face)[0], j + 1):
                    # Get the displacement vectors by adding the corresponding
                    # cell vectors, if the atom is close to an opposite face
                    # i.e. k > 2 subtract the cell vector
                    disp_vec = np.zeros(3)
                    for k in c:
                        disp_vec += cell[k % 3] * (int(k < 3) * 2 - 1)
                    pos = positions[i] + disp_vec
                    expanded_atoms.append(Atom(syms[i], position=pos))
        return expanded_atoms

    def _equal_elements_in_array(self, arr):
        s = np.sort(arr)
        return np.any(s[1:] == s[:-1])

    def _elements_match(self, s1, s2, kdtree):
        """Check if all the elements in s1 match the corresponding position in s2

        NOTE: The unit cells may be in different octants
        Hence, try all cyclic permutations of x,y and z
        """
        pos1 = s1.get_positions()
        for order in range(1):  # Is the order still needed?
            pos_order = [order, (order + 1) % 3, (order + 2) % 3]
            pos = pos1[:, np.argsort(pos_order)]
            dists, closest_in_s2 = kdtree.query(pos)

            # Check if the elements are the same
            if not np.all(s2.numbers[closest_in_s2] == s1.numbers):
                return False

            # Check if any distance is too large
            if np.any(dists > self.position_tolerance):
                return False

            # Check for duplicates in what atom is closest
            if self._equal_elements_in_array(closest_in_s2):
                return False

        return True

    def _least_frequent_element_to_origin(self, atoms):
        """Put one of the least frequent elements at the origin."""
        least_freq = self._get_only_least_frequent_of(atoms)
        cell_diag = np.sum(atoms.get_cell(), axis=0)
        d = least_freq.get_positions()[0] - 1e-6 * cell_diag
        atoms.positions -= d
        atoms.wrap(pbc=[1, 1, 1])

    def _get_rotation_reflection_matrices(self):
        """Compute candidates for the transformation matrix."""
        atoms1_ref = self._get_only_least_frequent_of(self.s1)
        cell = self.s1.get_cell().T
        cell_diag = np.sum(cell, axis=1)
        angle_tol = self.angle_tol

        # Additional vector that is added to make sure that
        # there always is an atom at the origin
        delta_vec = 1E-6 * cell_diag

        # Store three reference vectors and their lengths
        ref_vec = self.s2.get_cell()
        ref_vec_lengths = np.linalg.norm(ref_vec, axis=1)

        # Compute ref vec angles
        # ref_angles are arranged as [angle12, angle13, angle23]
        ref_angles = np.array(self._get_angles(ref_vec))
        large_angles = ref_angles > np.pi / 2.0
        ref_angles[large_angles] = np.pi - ref_angles[large_angles]

        # Translate by one cell diagonal so that a central cell is
        # surrounded by cells in all directions
        sc_atom_search = atoms1_ref * (3, 3, 3)
        new_sc_pos = sc_atom_search.get_positions()
        new_sc_pos -= new_sc_pos[0] + cell_diag - delta_vec

        lengths = np.linalg.norm(new_sc_pos, axis=1)

        candidate_indices = []
        rtol = self.ltol / len(self.s1)
        for k in range(3):
            correct_lengths_mask = np.isclose(lengths,
                                              ref_vec_lengths[k],
                                              rtol=rtol, atol=0)
            # The first vector is not interesting
            correct_lengths_mask[0] = False
            candidate_indices.append(np.nonzero(correct_lengths_mask)[0])
        # Now we calculate all relevant angles in one step. The relevant angles
        # are the ones made by the current candidates. We will have to keep
        # track of the indices in the angles matrix and the indices in the
        # position and length arrays.

        # Get all candidate indices (aci), only unique values
        aci = np.sort(list(set().union(*candidate_indices)))

        # Make a dictionary from original positions and lengths index to
        # index in angle matrix
        i2ang = dict(zip(aci, range(len(aci))))

        # Calculate the dot product divided by the lengths:
        # cos(angle) = dot(vec1, vec2) / |vec1| |vec2|
        cosa = np.inner(new_sc_pos[aci],
                        new_sc_pos[aci]) / np.outer(lengths[aci],
                                                    lengths[aci])
        # Make sure the inverse cosine will work
        cosa[cosa > 1] = 1
        cosa[cosa < -1] = -1
        angles = np.arccos(cosa)
        # Do trick for enantiomorphic structures
        angles[angles > np.pi / 2] = np.pi - angles[angles > np.pi / 2]

        # Check which angles match the reference angles
        # Test for all combinations on candidates. filterfalse makes sure
        # that there are no duplicate candidates. product is the same as
        # nested for loops.
        refined_candidate_list = []
        for p in filterfalse(self._equal_elements_in_array,
                             product(*candidate_indices)):
            a = np.array([angles[i2ang[p[0]], i2ang[p[1]]],
                          angles[i2ang[p[0]], i2ang[p[2]]],
                          angles[i2ang[p[1]], i2ang[p[2]]]])

            if np.allclose(a, ref_angles, atol=angle_tol, rtol=0):
                refined_candidate_list.append(new_sc_pos[np.array(p)].T)

        # Get the rotation/reflection matrix [R] by:
        # [R] = [V][T]^-1, where [V] is the reference vectors and
        # [T] is the trial vectors
        # XXX What do we know about the length/shape of refined_candidate_list?
        if len(refined_candidate_list) == 1:
            inverted_trial = 1.0 / refined_candidate_list
        else:
            inverted_trial = np.linalg.inv(refined_candidate_list)

        # Equivalent to np.matmul(ref_vec.T, inverted_trial)
        candidate_trans_mat = np.dot(ref_vec.T, inverted_trial.T).T
        return candidate_trans_mat, atoms1_ref.get_positions()

    def _reduce_to_primitive(self, structure):
        """Reduce the two structure to their primitive type"""
        try:
            import spglib
        except ImportError:
            raise SpgLibNotFoundError(
                "SpgLib is required if to_primitive=True")
        cell = (structure.get_cell()).tolist()
        pos = structure.get_scaled_positions().tolist()
        numbers = structure.get_atomic_numbers()

        cell, scaled_pos, numbers = spglib.standardize_cell(
            (cell, pos, numbers), to_primitive=True)

        atoms = Atoms(
            scaled_positions=scaled_pos,
            numbers=numbers,
            cell=cell,
            pbc=True)
        return atoms
