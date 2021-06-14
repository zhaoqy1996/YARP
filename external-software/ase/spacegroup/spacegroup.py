from __future__ import print_function, division
# Copyright (C) 2010, Jesper Friis
# (see accompanying license files for details).

"""Definition of the Spacegroup class.

This module only depends on NumPy and the space group database.
"""

import os
import warnings
from functools import total_ordering

import numpy as np
from ase.utils import basestring

__all__ = ['Spacegroup']


class SpacegroupError(Exception):
    """Base exception for the spacegroup module."""
    pass


class SpacegroupNotFoundError(SpacegroupError):
    """Raised when given space group cannot be found in data base."""
    pass


class SpacegroupValueError(SpacegroupError):
    """Raised when arguments have invalid value."""
    pass


@total_ordering
class Spacegroup(object):
    """A space group class.

    The instances of Spacegroup describes the symmetry operations for
    the given space group.

    Example:

    >>> from ase.spacegroup import Spacegroup
    >>>
    >>> sg = Spacegroup(225)
    >>> print('Space group', sg.no, sg.symbol)
    Space group 225 F m -3 m
    >>> sg.scaled_primitive_cell
    array([[ 0. ,  0.5,  0.5],
           [ 0.5,  0. ,  0.5],
           [ 0.5,  0.5,  0. ]])
    >>> sites, kinds = sg.equivalent_sites([[0,0,0]])
    >>> sites
    array([[ 0. ,  0. ,  0. ],
           [ 0. ,  0.5,  0.5],
           [ 0.5,  0. ,  0.5],
           [ 0.5,  0.5,  0. ]])
    """
    no = property(
        lambda self: self._no,
        doc='Space group number in International Tables of Crystallography.')
    symbol = property(
        lambda self: self._symbol,
        doc='Hermann-Mauguin (or international) symbol for the space group.')
    setting = property(
        lambda self: self._setting,
        doc='Space group setting. Either one or two.')
    lattice = property(
        lambda self: self._symbol[0],
        doc="""Lattice type:

    P     primitive
    I     body centering, h+k+l=2n
    F     face centering, h,k,l all odd or even
    A,B,C single face centering, k+l=2n, h+l=2n, h+k=2n
    R     rhombohedral centering, -h+k+l=3n (obverse); h-k+l=3n (reverse)
            """)
    centrosymmetric = property(
        lambda self: self._centrosymmetric,
        doc='Whether a center of symmetry exists.')
    scaled_primitive_cell = property(
        lambda self: self._scaled_primitive_cell,
        doc='Primitive cell in scaled coordinates as a matrix with the '
        'primitive vectors along the rows.')
    reciprocal_cell = property(
        lambda self: self._reciprocal_cell,
        doc='Tree Miller indices that span all kinematically non-forbidden '
        'reflections as a matrix with the Miller indices along the rows.')
    nsubtrans = property(
        lambda self: len(self._subtrans),
        doc='Number of cell-subtranslation vectors.')

    def _get_nsymop(self):
        """Returns total number of symmetry operations."""
        if self.centrosymmetric:
            return 2 * len(self._rotations) * len(self._subtrans)
        else:
            return len(self._rotations) * len(self._subtrans)
    nsymop = property(_get_nsymop, doc='Total number of symmetry operations.')
    subtrans = property(
        lambda self: self._subtrans,
        doc='Translations vectors belonging to cell-sub-translations.')
    rotations = property(
        lambda self: self._rotations,
        doc='Symmetry rotation matrices. The invertions are not included '
        'for centrosymmetrical crystals.')
    translations = property(
        lambda self: self._translations,
        doc='Symmetry translations. The invertions are not included '
        'for centrosymmetrical crystals.')

    def __init__(self, spacegroup, setting=1, datafile=None):
        """Returns a new Spacegroup instance.

        Parameters:

        spacegroup : int | string | Spacegroup instance
            The space group number in International Tables of
            Crystallography or its Hermann-Mauguin symbol. E.g.
            spacegroup=225 and spacegroup='F m -3 m' are equivalent.
        setting : 1 | 2
            Some space groups have more than one setting. `setting`
            determines Which of these should be used.
        datafile : None | string
            Path to database file. If `None`, the the default database
            will be used.
        """
        if isinstance(spacegroup, Spacegroup):
            for k, v in spacegroup.__dict__.items():
                setattr(self, k, v)
            return
        if not datafile:
            datafile = get_datafile()
        f = open(datafile, 'r')
        try:
            _read_datafile(self, spacegroup, setting, f)
        finally:
            f.close()

    def __repr__(self):
        return 'Spacegroup(%d, setting=%d)' % (self.no, self.setting)

    def todict(self):
        return {'number': self.no, 'setting': self.setting}

    def __str__(self):
        """Return a string representation of the space group data in
        the same format as found the database."""
        retval = []
        # no, symbol
        retval.append('%-3d   %s\n' % (self.no, self.symbol))
        # setting
        retval.append('  setting %d\n' % (self.setting))
        # centrosymmetric
        retval.append('  centrosymmetric %d\n' % (self.centrosymmetric))
        # primitive vectors
        retval.append('  primitive vectors\n')
        for i in range(3):
            retval.append('   ')
            for j in range(3):
                retval.append(' %13.10f' % (self.scaled_primitive_cell[i, j]))
            retval.append('\n')
        # primitive reciprocal vectors
        retval.append('  reciprocal vectors\n')
        for i in range(3):
            retval.append('   ')
            for j in range(3):
                retval.append(' %3d' % (self.reciprocal_cell[i, j]))
            retval.append('\n')
        # sublattice
        retval.append('  %d subtranslations\n' % self.nsubtrans)
        for i in range(self.nsubtrans):
            retval.append('   ')
            for j in range(3):
                retval.append(' %13.10f' % (self.subtrans[i, j]))
            retval.append('\n')
        # symmetry operations
        nrot = len(self.rotations)
        retval.append('  %d symmetry operations (rot+trans)\n' % nrot)
        for i in range(nrot):
            retval.append(' ')
            for j in range(3):
                retval.append(' ')
                for k in range(3):
                    retval.append(' %2d' % (self.rotations[i, j, k]))
                retval.append('  ')
            for j in range(3):
                retval.append(' %13.10f' % self.translations[i, j])
            retval.append('\n')
        retval.append('\n')
        return ''.join(retval)

    def __eq__(self, other):
        return self.no == other.no and self.setting == other.setting

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.no < other.no or (
            self.no == other.no and self.setting < other.setting)

    def __index__(self):
        return self.no

    __int__ = __index__

    def get_symop(self):
        """Returns all symmetry operations (including inversions and
        subtranslations) as a sequence of (rotation, translation)
        tuples."""
        symop = []
        parities = [1]
        if self.centrosymmetric:
            parities.append(-1)
        for parity in parities:
            for subtrans in self.subtrans:
                for rot, trans in zip(self.rotations, self.translations):
                    newtrans = np.mod(trans + subtrans, 1)
                    symop.append((parity * rot, newtrans))
        return symop

    def get_op(self):
        """Returns all symmetry operations (including inversions and
        subtranslations), but unlike get_symop(), they are returned as
        two ndarrays."""
        if self.centrosymmetric:
            rot = np.tile(np.vstack((self.rotations, -self.rotations)),
                          (self.nsubtrans, 1, 1))
            trans = np.tile(np.vstack((self.translations, -self.translations)),
                            (self.nsubtrans, 1))
            trans += np.repeat(self.subtrans, 2 * len(self.rotations), axis=0)
            trans = np.mod(trans, 1)
        else:
            rot = np.tile(self.rotations, (self.nsubtrans, 1, 1))
            trans = np.tile(self.translations, (self.nsubtrans, 1))
            trans += np.repeat(self.subtrans, len(self.rotations), axis=0)
            trans = np.mod(trans, 1)
        return rot, trans

    def get_rotations(self):
        """Return all rotations, including inversions for
        centrosymmetric crystals."""
        if self.centrosymmetric:
            return np.vstack((self.rotations, -self.rotations))
        else:
            return self.rotations

    def equivalent_reflections(self, hkl):
        """Return all equivalent reflections to the list of Miller indices
        in hkl.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.equivalent_reflections([[0, 0, 2]])
        array([[ 0,  0, -2],
               [ 0, -2,  0],
               [-2,  0,  0],
               [ 2,  0,  0],
               [ 0,  2,  0],
               [ 0,  0,  2]])
        """
        hkl = np.array(hkl, dtype='int', ndmin=2)
        rot = self.get_rotations()
        n, nrot = len(hkl), len(rot)
        R = rot.transpose(0, 2, 1).reshape((3 * nrot, 3)).T
        refl = np.dot(hkl, R).reshape((n * nrot, 3))
        ind = np.lexsort(refl.T)
        refl = refl[ind]
        diff = np.diff(refl, axis=0)
        mask = np.any(diff, axis=1)
        return np.vstack((refl[:-1][mask], refl[-1, :]))

    def equivalent_lattice_points(self, uvw):
        """Return all lattice points equivalent to any of the lattice points
        in `uvw` with respect to rotations only.

        Only equivalent lattice points that conserves the distance to
        origo are included in the output (making this a kind of real
        space version of the equivalent_reflections() method).

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.equivalent_lattice_points([[0, 0, 2]])
        array([[ 0,  0, -2],
               [ 0, -2,  0],
               [-2,  0,  0],
               [ 2,  0,  0],
               [ 0,  2,  0],
               [ 0,  0,  2]])

        """
        uvw = np.array(uvw, ndmin=2)
        rot = self.get_rotations()
        n, nrot = len(uvw), len(rot)
        directions = np.dot(uvw, rot).reshape((n * nrot, 3))
        ind = np.lexsort(directions.T)
        directions = directions[ind]
        diff = np.diff(directions, axis=0)
        mask = np.any(diff, axis=1)
        return np.vstack((directions[:-1][mask], directions[-1:]))

    def symmetry_normalised_reflections(self, hkl):
        """Returns an array of same size as *hkl*, containing the
        corresponding symmetry-equivalent reflections of lowest
        indices.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.symmetry_normalised_reflections([[2, 0, 0], [0, 2, 0]])
        array([[ 0,  0, -2],
               [ 0,  0, -2]])
        """
        hkl = np.array(hkl, dtype=int, ndmin=2)
        normalised = np.empty(hkl.shape, int)
        R = self.get_rotations().transpose(0, 2, 1)
        for i, g in enumerate(hkl):
            gsym = np.dot(R, g)
            j = np.lexsort(gsym.T)[0]
            normalised[i, :] = gsym[j]
        return normalised

    def unique_reflections(self, hkl):
        """Returns a subset *hkl* containing only the symmetry-unique
        reflections.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.unique_reflections([[ 2,  0,  0],
        ...                        [ 0, -2,  0],
        ...                        [ 2,  2,  0],
        ...                        [ 0, -2, -2]])
        array([[2, 0, 0],
               [2, 2, 0]])
        """
        hkl = np.array(hkl, dtype=int, ndmin=2)
        hklnorm = self.symmetry_normalised_reflections(hkl)
        perm = np.lexsort(hklnorm.T)
        iperm = perm.argsort()
        xmask = np.abs(np.diff(hklnorm[perm], axis=0)).any(axis=1)
        mask = np.concatenate(([True], xmask))
        imask = mask[iperm]
        return hkl[imask]

    def equivalent_sites(self, scaled_positions, onduplicates='error',
                         symprec=1e-3, occupancies=None):
        """Returns the scaled positions and all their equivalent sites.

        Parameters:

        scaled_positions: list | array
            List of non-equivalent sites given in unit cell coordinates.

        occupancies: list | array, optional (default=None)
            List of occupancies corresponding to the respective sites.

        onduplicates : 'keep' | 'replace' | 'warn' | 'error'
            Action if `scaled_positions` contain symmetry-equivalent
            positions of full occupancy:

            'keep'
               ignore additional symmetry-equivalent positions
            'replace'
                replace
            'warn'
                like 'keep', but issue an UserWarning
            'error'
                raises a SpacegroupValueError

        symprec: float
            Minimum "distance" betweed two sites in scaled coordinates
            before they are counted as the same site.

        Returns:

        sites: array
            A NumPy array of equivalent sites.
        kinds: list
            A list of integer indices specifying which input site is
            equivalent to the corresponding returned site.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sites, kinds = sg.equivalent_sites([[0, 0, 0], [0.5, 0.0, 0.0]])
        >>> sites
        array([[ 0. ,  0. ,  0. ],
               [ 0. ,  0.5,  0.5],
               [ 0.5,  0. ,  0.5],
               [ 0.5,  0.5,  0. ],
               [ 0.5,  0. ,  0. ],
               [ 0. ,  0.5,  0. ],
               [ 0. ,  0. ,  0.5],
               [ 0.5,  0.5,  0.5]])
        >>> kinds
        [0, 0, 0, 0, 1, 1, 1, 1]
        """
        kinds = []
        sites = []

        scaled = np.array(scaled_positions, ndmin=2)

        for kind, pos in enumerate(scaled):
            for rot, trans in self.get_symop():
                site = np.mod(np.dot(rot, pos) + trans, 1.)
                if not sites:
                    sites.append(site)
                    kinds.append(kind)
                    continue
                t = site - sites
                mask = np.all((abs(t) < symprec) |
                              (abs(abs(t) - 1.0) < symprec), axis=1)
                if np.any(mask):
                    inds = np.argwhere(mask).flatten()
                    for ind in inds:
                        # then we would just add the same thing again -> skip
                        if kinds[ind] == kind:
                            pass
                        elif onduplicates == 'keep':
                            pass
                        elif onduplicates == 'replace':
                            kinds[ind] = kind
                        elif onduplicates == 'warn':
                            warnings.warn('scaled_positions %d and %d '
                                          'are equivalent' % (kinds[ind], kind))
                        elif onduplicates == 'error':
                            raise SpacegroupValueError(
                                'scaled_positions %d and %d are equivalent' % (
                                    kinds[ind], kind))
                        else:
                            raise SpacegroupValueError(
                                'Argument "onduplicates" must be one of: '
                                '"keep", "replace", "warn" or "error".')
                else:
                    sites.append(site)
                    kinds.append(kind)

        return np.array(sites), kinds

    def symmetry_normalised_sites(self, scaled_positions,
                                  map_to_unitcell=True):
        """Returns an array of same size as *scaled_positions*,
        containing the corresponding symmetry-equivalent sites of
        lowest indices.

        If *map_to_unitcell* is true, the returned positions are all
        mapped into the unit cell, i.e. lattice translations are
        included as symmetry operator.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.symmetry_normalised_sites([[0.0, 0.5, 0.5], [1.0, 1.0, 0.0]])
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])
        """
        scaled = np.array(scaled_positions, ndmin=2)
        normalised = np.empty(scaled.shape, np.float)
        rot, trans = self.get_op()
        for i, pos in enumerate(scaled):
            sympos = np.dot(rot, pos) + trans
            if map_to_unitcell:
                # Must be done twice, see the scaled_positions.py test
                sympos %= 1.0
                sympos %= 1.0
            j = np.lexsort(sympos.T)[0]
            normalised[i, :] = sympos[j]
        return normalised

    def unique_sites(self, scaled_positions, symprec=1e-3, output_mask=False,
                     map_to_unitcell=True):
        """Returns a subset of *scaled_positions* containing only the
        symmetry-unique positions.  If *output_mask* is True, a boolean
        array masking the subset is also returned.

        If *map_to_unitcell* is true, all sites are first mapped into
        the unit cell making e.g. [0, 0, 0] and [1, 0, 0] equivalent.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.unique_sites([[0.0, 0.0, 0.0],
        ...                  [0.5, 0.5, 0.0],
        ...                  [1.0, 0.0, 0.0],
        ...                  [0.5, 0.0, 0.0]])
        array([[ 0. ,  0. ,  0. ],
               [ 0.5,  0. ,  0. ]])
        """
        scaled = np.array(scaled_positions, ndmin=2)
        symnorm = self.symmetry_normalised_sites(scaled, map_to_unitcell)
        perm = np.lexsort(symnorm.T)
        iperm = perm.argsort()
        xmask = np.abs(np.diff(symnorm[perm], axis=0)).max(axis=1) > symprec
        mask = np.concatenate(([True], xmask))
        imask = mask[iperm]
        if output_mask:
            return scaled[imask], imask
        else:
            return scaled[imask]

    def tag_sites(self, scaled_positions, symprec=1e-3):
        """Returns an integer array of the same length as *scaled_positions*,
        tagging all equivalent atoms with the same index.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.tag_sites([[0.0, 0.0, 0.0],
        ...               [0.5, 0.5, 0.0],
        ...               [1.0, 0.0, 0.0],
        ...               [0.5, 0.0, 0.0]])
        array([0, 0, 0, 1])
        """
        scaled = np.array(scaled_positions, ndmin=2)
        scaled %= 1.0
        scaled %= 1.0
        tags = -np.ones((len(scaled), ), dtype=int)
        mask = np.ones((len(scaled), ), dtype=np.bool)
        rot, trans = self.get_op()
        i = 0
        while mask.any():
            pos = scaled[mask][0]
            sympos = np.dot(rot, pos) + trans
            # Must be done twice, see the scaled_positions.py test
            sympos %= 1.0
            sympos %= 1.0
            m = ~np.all(np.any(np.abs(scaled[np.newaxis, :, :] -
                                      sympos[:, np.newaxis, :]) > symprec,
                               axis=2), axis=0)
            assert not np.any((~mask) & m)
            tags[m] = i
            mask &= ~m
            i += 1
        return tags


def get_datafile():
    """Return default path to datafile."""
    return os.path.join(os.path.dirname(__file__), 'spacegroup.dat')


def format_symbol(symbol):
    """Returns well formatted Hermann-Mauguin symbol as extected by
    the database, by correcting the case and adding missing or
    removing dublicated spaces."""
    fixed = []
    s = symbol.strip()
    s = s[0].upper() + s[1:].lower()
    for c in s:
        if c.isalpha():
            if len(fixed) and fixed[-1] == '/':
                fixed.append(c)
            else:
                fixed.append(' ' + c + ' ')
        elif c.isspace():
            fixed.append(' ')
        elif c.isdigit():
            fixed.append(c)
        elif c == '-':
            fixed.append(' ' + c)
        elif c == '/':
            fixed.append(c)
    s = ''.join(fixed).strip()
    return ' '.join(s.split())


# Functions for parsing the database. They are moved outside the
# Spacegroup class in order to make it easier to later implement
# caching to avoid reading the database each time a new Spacegroup
# instance is created.

def _skip_to_blank(f, spacegroup, setting):
    """Read lines from f until a blank line is encountered."""
    while True:
        line = f.readline()
        if not line:
            raise SpacegroupNotFoundError(
                'invalid spacegroup `%s`, setting `%s` not found in data base' %
                (spacegroup, setting))
        if not line.strip():
            break


def _skip_to_nonblank(f, spacegroup, setting):
    """Read lines from f until a nonblank line not starting with a
    hash (#) is encountered and returns this and the next line."""
    while True:
        line1 = f.readline()
        if not line1:
            raise SpacegroupNotFoundError(
                'invalid spacegroup %s, setting %i not found in data base' %
                (spacegroup, setting))
        line1.strip()
        if line1 and not line1.startswith('#'):
            line2 = f.readline()
            break
    return line1, line2


def _read_datafile_entry(spg, no, symbol, setting, f):
    """Read space group data from f to spg."""

    floats = {'0.0': 0.0, '1.0': 1.0, '0': 0.0, '1': 1.0, '-1': -1.0}
    for n, d in [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 6), (5, 6)]:
        floats['{0}/{1}'.format(n, d)] = n / d
        floats['-{0}/{1}'.format(n, d)] = -n / d

    spg._no = no
    spg._symbol = symbol.strip()
    spg._setting = setting
    spg._centrosymmetric = bool(int(f.readline().split()[1]))
    # primitive vectors
    f.readline()
    spg._scaled_primitive_cell = np.array([[float(floats.get(s, s))
                                            for s in f.readline().split()]
                                           for i in range(3)],
                                          dtype=np.float)
    # primitive reciprocal vectors
    f.readline()
    spg._reciprocal_cell = np.array([[int(i)
                                      for i in f.readline().split()]
                                     for i in range(3)],
                                    dtype=np.int)
    # subtranslations
    spg._nsubtrans = int(f.readline().split()[0])
    spg._subtrans = np.array([[float(floats.get(t, t))
                               for t in f.readline().split()]
                              for i in range(spg._nsubtrans)],
                             dtype=np.float)
    # symmetry operations
    nsym = int(f.readline().split()[0])
    symop = np.array([[float(floats.get(s, s)) for s in f.readline().split()]
                      for i in range(nsym)],
                     dtype=np.float)
    spg._nsymop = nsym
    spg._rotations = np.array(symop[:, :9].reshape((nsym, 3, 3)), dtype=np.int)
    spg._translations = symop[:, 9:]


def _read_datafile(spg, spacegroup, setting, f):
    if isinstance(spacegroup, int):
        pass
    elif isinstance(spacegroup, basestring):
        spacegroup = ' '.join(spacegroup.strip().split())
        compact_spacegroup = ''.join(spacegroup.split())
    else:
        raise SpacegroupValueError('`spacegroup` must be of type int or str')
    while True:
        line1, line2 = _skip_to_nonblank(f, spacegroup, setting)
        _no, _symbol = line1.strip().split(None, 1)
        _symbol = format_symbol(_symbol)
        compact_symbol = ''.join(_symbol.split())
        _setting = int(line2.strip().split()[1])
        _no = int(_no)
        if ((isinstance(spacegroup, int) and _no == spacegroup and
             _setting == setting) or
            (isinstance(spacegroup, basestring) and
             compact_symbol == compact_spacegroup)):
            _read_datafile_entry(spg, _no, _symbol, _setting, f)
            break
        else:
            _skip_to_blank(f, spacegroup, setting)


def parse_sitesym(symlist, sep=','):
    """Parses a sequence of site symmetries in the form used by
    International Tables and returns corresponding rotation and
    translation arrays.

    Example:

    >>> symlist = [
    ...     'x,y,z',
    ...     '-y+1/2,x+1/2,z',
    ...     '-y,-x,-z',
    ... ]
    >>> rot, trans = parse_sitesym(symlist)
    >>> rot
    array([[[ 1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  1]],
    <BLANKLINE>
           [[ 0, -1,  0],
            [ 1,  0,  0],
            [ 0,  0,  1]],
    <BLANKLINE>
           [[ 0, -1,  0],
            [-1,  0,  0],
            [ 0,  0, -1]]])
    >>> trans
    array([[ 0. ,  0. ,  0. ],
           [ 0.5,  0.5,  0. ],
           [ 0. ,  0. ,  0. ]])
    """
    nsym = len(symlist)
    rot = np.zeros((nsym, 3, 3), dtype='int')
    trans = np.zeros((nsym, 3))
    for i, sym in enumerate(symlist):
        for j, s in enumerate(sym.split(sep)):
            s = s.lower().strip()
            while s:
                sign = 1
                if s[0] in '+-':
                    if s[0] == '-':
                        sign = -1
                    s = s[1:]
                if s[0] in 'xyz':
                    k = ord(s[0]) - ord('x')
                    rot[i, j, k] = sign
                    s = s[1:]
                elif s[0].isdigit() or s[0] == '.':
                    n = 0
                    while n < len(s) and (s[n].isdigit() or s[n] in '/.'):
                        n += 1
                    t = s[:n]
                    s = s[n:]
                    if '/' in t:
                        q, r = t.split('/')
                        trans[i, j] = float(q) / float(r)
                    else:
                        trans[i, j] = float(t)
                else:
                    raise SpacegroupValueError(
                        'Error parsing %r. Invalid site symmetry: %s' %
                        (s, sym))
    return rot, trans


def spacegroup_from_data(no=None, symbol=None, setting=None,
                         centrosymmetric=None, scaled_primitive_cell=None,
                         reciprocal_cell=None, subtrans=None, sitesym=None,
                         rotations=None, translations=None, datafile=None):
    """Manually create a new space group instance.  This might be
    useful when reading crystal data with its own spacegroup
    definitions."""
    if no is not None and setting is not None:
        spg = Spacegroup(no, setting, datafile)
    elif symbol is not None:
        spg = Spacegroup(symbol, None, datafile)
    else:
        raise SpacegroupValueError('either *no* and *setting* '
                                   'or *symbol* must be given')
    have_sym = False
    if centrosymmetric is not None:
        spg._centrosymmetric = bool(centrosymmetric)
    if scaled_primitive_cell is not None:
        spg._scaled_primitive_cell = np.array(scaled_primitive_cell)
    if reciprocal_cell is not None:
        spg._reciprocal_cell = np.array(reciprocal_cell)
    if subtrans is not None:
        spg._subtrans = np.atleast_2d(subtrans)
        spg._nsubtrans = spg._subtrans.shape[0]
    if sitesym is not None:
        spg._rotations, spg._translations = parse_sitesym(sitesym)
        have_sym = True
    if rotations is not None:
        spg._rotations = np.atleast_3d(rotations)
        have_sym = True
    if translations is not None:
        spg._translations = np.atleast_2d(translations)
        have_sym = True
    if have_sym:
        if spg._rotations.shape[0] != spg._translations.shape[0]:
            raise SpacegroupValueError('inconsistent number of rotations and '
                                       'translations')
        spg._nsymop = spg._rotations.shape[0]
    return spg


def get_spacegroup(atoms, symprec=1e-5):
    """Determine the spacegroup to which belongs the Atoms object.

    This requires spglib: https://atztogo.github.io/spglib/ .

    Parameters:

    atoms: Atoms object
        Types, positions and unit-cell.
    symprec: float
        Symmetry tolerance, i.e. distance tolerance in Cartesian
        coordinates to find crystal symmetry.

    The Spacegroup object is returned.
    """

    # Example:
    # (We don't include the example in docstring to appease doctests
    #  when import fails)
    # >>> from ase.build import bulk
    # >>> atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    # >>> sg = get_spacegroup(atoms)
    # >>> sg
    # Spacegroup(225, setting=1)
    # >>> sg.no
    # 225

    try:
        import spglib  # For version 1.9 or later
    except ImportError:
        from pyspglib import spglib  # For versions 1.8.x or before

    sg = spglib.get_spacegroup((atoms.get_cell(),
                                atoms.get_scaled_positions(),
                                atoms.get_atomic_numbers()),
                               symprec=symprec)
    if sg is None:
        raise RuntimeError('Spacegroup not found')
    sg_no = int(sg[sg.find('(') + 1:sg.find(')')])
    return Spacegroup(sg_no)

    # no spglib, we use our own spacegroup finder. Not as fast as spglib.
    # we center the Atoms positions on each atom in the cell, and find the
    # spacegroup of highest symmetry
    #
    # XXX That function is not finished.
    # found = None
    # for kind, pos in enumerate(atoms.get_scaled_positions()):
    #     sg = _get_spacegroup(atoms, symprec=symprec, center=kind)
    #     if found is None or sg.no > found.no:
    #         found = sg

    # return found


def _get_spacegroup(atoms, symprec=1e-5, center=None):
    """ASE implementation of get_spacegroup, pure python."""
    raise NotImplementedError('get_spacegroup() is not finished')

    # we try all available spacegroups from 230 to 1, backwards
    # a Space group is the collection of all symmetry operations which lets the
    # unit cell invariant.
    found = None
    positions = atoms.get_scaled_positions(wrap=True)  # in the lattice frame

    # make sure we are insensitive to translation. this choice is arbitrary and
    # could lead to a 'slightly' wrong guess for the Space group, e.g. do not
    # guess centro-symmetry.
    if center is not None:
        try:
            positions -= positions[center]
        except IndexError:
            pass

    # search space groups from the highest symmetry to the lowest
    # retain the first match
    for nb in range(230, 0, -1):
        sg = Spacegroup(nb)
        #
        # now we scan all atoms in the cell and look for equivalent sites
        sites, kinds = sg.equivalent_sites(positions,
                                           onduplicates='keep',
                                           symprec=symprec)

        # the equivalent sites should match all other atom locations in the
        # cell as the spacegroup transforms the unit cell in itself
        # we test on the number of equivalent sites
        if len(sites) == len(positions):
            # store the space group into the list
            found = sg
            break

    return found
