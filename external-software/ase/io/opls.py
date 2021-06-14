from __future__ import print_function
import time
import numpy as np

from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
from ase.utils import basestring


def twochar(name):
    if len(name) > 1:
        return name[:2]
    else:
        return name + ' '


class BondData:
    def __init__(self, name_value_hash):
        self.nvh = name_value_hash

    def name_value(self, aname, bname):
        name1 = twochar(aname) + '-' + twochar(bname)
        name2 = twochar(bname) + '-' + twochar(aname)
        if name1 in self.nvh:
            return name1, self.nvh[name1]
        if name2 in self.nvh:
            return name2, self.nvh[name2]
        return None, None

    def value(self, aname, bname):
        return self.name_value(aname, bname)[1]


class CutoffList(BondData):
    def max(self):
        return max(self.nvh.values())


class AnglesData:
    def __init__(self, name_value_hash):
        self.nvh = name_value_hash

    def name_value(self, aname, bname, cname):
        for name in [
            (twochar(aname) + '-' + twochar(bname) + '-' + twochar(cname)),
            (twochar(cname) + '-' + twochar(bname) + '-' + twochar(aname))]:
            if name in self.nvh:
                return name, self.nvh[name]
        return None, None


class DihedralsData:
    def __init__(self, name_value_hash):
        self.nvh = name_value_hash

    def name_value(self, aname, bname, cname, dname):
        for name in [
            (twochar(aname) + '-' + twochar(bname) + '-' +
             twochar(cname) + '-' + twochar(dname)),
            (twochar(dname) + '-' + twochar(cname) + '-' +
             twochar(bname) + '-' + twochar(aname))]:
            if name in self.nvh:
                return name, self.nvh[name]
        return None, None


class OPLSff:
    def __init__(self, fileobj=None, warnings=0):
        self.warnings = warnings
        self.data = {}
        if fileobj is not None:
            self.read(fileobj)

    def read(self, fileobj, comments='#'):
        if isinstance(fileobj, basestring):
            fileobj = open(fileobj)

        def read_block(name, symlen, nvalues):
            """Read a data block.

            name: name of the block to store in self.data
            symlen: length of the symbol
            nvalues: number of values expected
            """

            if name not in self.data:
                self.data[name] = {}
            data = self.data[name]

            def add_line():
                line = fileobj.readline().strip()
                if not len(line):  # end of the block
                    return False
                line = line.split('#')[0]  # get rid of comments
                if len(line) > symlen:
                    symbol = line[:symlen]
                    words = line[symlen:].split()
                    if len(words) >= nvalues:
                        if nvalues == 1:
                            data[symbol] = float(words[0])
                        else:
                            data[symbol] = [float(word)
                                            for word in words[:nvalues]]
                return True

            while add_line():
                pass

        read_block('one', 2, 3)
        read_block('bonds', 5, 2)
        read_block('angles', 8, 2)
        read_block('dihedrals', 11, 4)
        read_block('cutoffs', 5, 1)

        self.bonds = BondData(self.data['bonds'])
        self.angles = AnglesData(self.data['angles'])
        self.dihedrals = DihedralsData(self.data['dihedrals'])
        self.cutoffs = CutoffList(self.data['cutoffs'])

    def write_lammps(self, atoms, prefix='lammps'):
        """Write input for a LAMMPS calculation."""
        self.prefix = prefix

        if hasattr(atoms, 'connectivities'):
            connectivities = atoms.connectivities
        else:
            btypes, blist = self.get_bonds(atoms)
            atypes, alist = self.get_angles()
            dtypes, dlist = self.get_dihedrals(alist, atypes)
            connectivities = {
                'bonds': blist,
                'bond types': btypes,
                'angles': alist,
                'angle types': atypes,
                'dihedrals': dlist,
                'dihedral types': dtypes}

            self.write_lammps_definitions(atoms, btypes, atypes, dtypes)
            self.write_lammps_in()

        self.write_lammps_atoms(atoms, connectivities)

    def write_lammps_in(self):
        fileobj = self.prefix + '_in'
        if isinstance(fileobj, basestring):
            fileobj = open(fileobj, 'w')
        fileobj.write("""# LAMMPS relaxation (written by ASE)

units           metal
atom_style      full
boundary        p p p
#boundary       p p f

""")
        fileobj.write('read_data ' + self.prefix + '_atoms\n')
        fileobj.write('include  ' + self.prefix + '_opls\n')
        fileobj.write("""
kspace_style    pppm 1e-5
#kspace_modify  slab 3.0

neighbor        1.0 bin
neigh_modify    delay 0 every 1 check yes

thermo          1000
thermo_style    custom step temp press cpu pxx pyy pzz pxy pxz pyz ke pe etotal vol lx ly lz atoms

dump            1 all xyz 1000 dump_relax.xyz
dump_modify     1 sort id

restart         100000 test_relax

min_style       fire
minimize        1.0e-14 1.0e-5 100000 100000
""")
        fileobj.close()

    def write_lammps_atoms(self, atoms, connectivities):
        """Write atoms input for LAMMPS"""

        fname = self.prefix + '_atoms'
        fileobj = open(fname, 'w')

        # header
        fileobj.write(fileobj.name + ' (by ' + str(self.__class__) + ')\n\n')
        fileobj.write(str(len(atoms)) + ' atoms\n')
        fileobj.write(str(len(atoms.types)) + ' atom types\n')
        blist = connectivities['bonds']
        if len(blist):
            btypes = connectivities['bond types']
            fileobj.write(str(len(blist)) + ' bonds\n')
            fileobj.write(str(len(btypes)) + ' bond types\n')
        alist = connectivities['angles']
        if len(alist):
            atypes = connectivities['angle types']
            fileobj.write(str(len(alist)) + ' angles\n')
            fileobj.write(str(len(atypes)) + ' angle types\n')
        dlist = connectivities['dihedrals']
        if len(dlist):
            dtypes = connectivities['dihedral types']
            fileobj.write(str(len(dlist)) + ' dihedrals\n')
            fileobj.write(str(len(dtypes)) + ' dihedral types\n')

        # cell
        p = Prism(atoms.get_cell())
        xhi, yhi, zhi, xy, xz, yz = p.get_lammps_prism_str()
        fileobj.write('\n0.0 %s  xlo xhi\n' % xhi)
        fileobj.write('0.0 %s  ylo yhi\n' % yhi)
        fileobj.write('0.0 %s  zlo zhi\n' % zhi)

        # atoms
        fileobj.write('\nAtoms\n\n')
        tag = atoms.get_tags()
        if atoms.has('molid'):
            molid = atoms.get_array('molid')
        else:
            molid = [1] * len(atoms)
        for i, r in enumerate(
            p.positions_to_lammps_strs(atoms.get_positions())):
            q = self.data['one'][atoms.types[tag[i]]][2]
            fileobj.write('%6d %3d %3d %s %s %s %s' % ((i + 1, molid[i],
                                                        tag[i] + 1,
                                                        q) + tuple(r)))
            fileobj.write(' # ' + atoms.types[tag[i]] + '\n')

        # velocities
        velocities = atoms.get_velocities()
        if velocities is not None:
            fileobj.write('\nVelocities\n\n')
            for i, v in enumerate(velocities):
                fileobj.write('%6d %g %g %g\n' %
                              (i + 1, v[0], v[1], v[2]))

        # masses
        fileobj.write('\nMasses\n\n')
        for i, typ in enumerate(atoms.types):
            cs = atoms.split_symbol(typ)[0]
            fileobj.write('%6d %g # %s -> %s\n' %
                          (i + 1,
                           atomic_masses[chemical_symbols.index(cs)],
                           typ, cs))

        # bonds
        if len(blist):
            fileobj.write('\nBonds\n\n')
            for ib, bvals in enumerate(blist):
                fileobj.write('%8d %6d %6d %6d ' %
                              (ib + 1, bvals[0] + 1, bvals[1] + 1,
                               bvals[2] + 1))
                try:
                    fileobj.write('# ' + btypes[bvals[0]])
                except:
                    pass
                fileobj.write('\n')

        # angles
        if len(alist):
            fileobj.write('\nAngles\n\n')
            for ia, avals in enumerate(alist):
                fileobj.write('%8d %6d %6d %6d %6d ' %
                              (ia + 1, avals[0] + 1,
                               avals[1] + 1, avals[2] + 1, avals[3] + 1))
                try:
                    fileobj.write('# ' + atypes[avals[0]])
                except:
                    pass
                fileobj.write('\n')

        # dihedrals
        if len(dlist):
            fileobj.write('\nDihedrals\n\n')
            for i, dvals in enumerate(dlist):
                fileobj.write('%8d %6d %6d %6d %6d %6d ' %
                              (i + 1, dvals[0] + 1,
                               dvals[1] + 1, dvals[2] + 1,
                               dvals[3] + 1, dvals[4] + 1))
                try:
                    fileobj.write('# ' + dtypes[dvals[0]])
                except:
                    pass
                fileobj.write('\n')

    def update_neighbor_list(self, atoms):
        cut = 0.5 * max(self.data['cutoffs'].values())
        self.nl = NeighborList([cut] * len(atoms), skin=0,
                               bothways=True, self_interaction=False)
        self.nl.update(atoms)
        self.atoms = atoms

    def get_bonds(self, atoms):
        """Find bonds and return them and their types"""
        cutoffs = CutoffList(self.data['cutoffs'])
        self.update_neighbor_list(atoms)

        types = atoms.get_types()
        tags = atoms.get_tags()
        cell = atoms.get_cell()
        bond_list = []
        bond_types = []
        for i, atom in enumerate(atoms):
            iname = types[tags[i]]
            indices, offsets = self.nl.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                if j <= i:
                    continue  # do not double count
                jname = types[tags[j]]
                cut = cutoffs.value(iname, jname)
                if cut is None:
                    if self.warnings > 1:
                        print('Warning: cutoff %s-%s not found'
                              % (iname, jname))
                    continue  # don't have it
                dist = np.linalg.norm(atom.position - atoms[j].position -
                                      np.dot(offset, cell))
                if dist > cut:
                    continue  # too far away
                name, val = self.bonds.name_value(iname, jname)
                if name is None:
                    if self.warnings:
                        print('Warning: potential %s-%s not found'
                              % (iname, jname))
                    continue  # don't have it
                if name not in bond_types:
                    bond_types.append(name)
                bond_list.append([bond_types.index(name), i, j])
        return bond_types, bond_list

    def get_angles(self, atoms=None):
        cutoffs = CutoffList(self.data['cutoffs'])
        if atoms is not None:
            self.update_neighbor_list(atoms)
        else:
            atoms = self.atoms

        types = atoms.get_types()
        tags = atoms.get_tags()
        cell = atoms.get_cell()
        ang_list = []
        ang_types = []

        # center atom *-i-*
        for i, atom in enumerate(atoms):
            iname = types[tags[i]]
            indicesi, offsetsi = self.nl.get_neighbors(i)

            # search for first neighbor j-i-*
            for j, offsetj in zip(indicesi, offsetsi):
                jname = types[tags[j]]
                cut = cutoffs.value(iname, jname)
                if cut is None:
                    continue  # don't have it
                dist = np.linalg.norm(atom.position - atoms[j].position -
                                      np.dot(offsetj, cell))
                if dist > cut:
                    continue  # too far away

                # search for second neighbor j-i-k
                for k, offsetk in zip(indicesi, offsetsi):
                    if k <= j:
                        continue  # avoid double count
                    kname = types[tags[k]]
                    cut = cutoffs.value(iname, kname)
                    if cut is None:
                        continue  # don't have it
                    dist = np.linalg.norm(atom.position -
                                          np.dot(offsetk, cell) -
                                          atoms[k].position)
                    if dist > cut:
                        continue  # too far away
                    name, val = self.angles.name_value(jname, iname,
                                                       kname)
                    if name is None:
                        if self.warnings > 1:
                            print('Warning: angles %s-%s-%s not found'
                                  % (jname, iname, kname))
                        continue  # don't have it
                    if name not in ang_types:
                        ang_types.append(name)
                    ang_list.append([ang_types.index(name), j, i, k])

        return ang_types, ang_list

    def get_dihedrals(self, ang_types, ang_list):
        'Dihedrals derived from angles.'

        cutoffs = CutoffList(self.data['cutoffs'])

        atoms = self.atoms
        types = atoms.get_types()
        tags = atoms.get_tags()
        cell = atoms.get_cell()

        dih_list = []
        dih_types = []

        def append(name, i, j, k, l):
            if name not in dih_types:
                dih_types.append(name)
            index = dih_types.index(name)
            if (([index, i, j, k, l] not in dih_list) and
                ([index, l, k, j, i] not in dih_list)):
                dih_list.append([index, i, j, k, l])

        for angle in ang_types:
            l, i, j, k = angle
            iname = types[tags[i]]
            jname = types[tags[j]]
            kname = types[tags[k]]

            # search for l-i-j-k
            indicesi, offsetsi = self.nl.get_neighbors(i)
            for l, offsetl in zip(indicesi, offsetsi):
                if l == j:
                    continue  # avoid double count
                lname = types[tags[l]]
                cut = cutoffs.value(iname, lname)
                if cut is None:
                    continue  # don't have it
                dist = np.linalg.norm(atoms[i].position - atoms[l].position -
                                      np.dot(offsetl, cell))
                if dist > cut:
                    continue  # too far away
                name, val = self.dihedrals.name_value(lname, iname,
                                                      jname, kname)
                if name is None:
                    continue  # don't have it
                append(name, l, i, j, k)

            # search for i-j-k-l
            indicesk, offsetsk = self.nl.get_neighbors(k)
            for l, offsetl in zip(indicesk, offsetsk):
                if l == j:
                    continue  # avoid double count
                lname = types[tags[l]]
                cut = cutoffs.value(kname, lname)
                if cut is None:
                    continue  # don't have it
                dist = np.linalg.norm(atoms[k].position - atoms[l].position -
                                      np.dot(offsetl, cell))
                if dist > cut:
                    continue  # too far away
                name, val = self.dihedrals.name_value(iname, jname,
                                                      kname, lname)
                if name is None:
                    continue  # don't have it
                append(name, i, j, k, l)

        return dih_types, dih_list

    def write_lammps_definitions(self, atoms, btypes, atypes, dtypes):
        """Write force field definitions for LAMMPS."""

        fileobj = self.prefix + '_opls'
        if isinstance(fileobj, basestring):
            fileobj = open(fileobj, 'w')

        fileobj.write('# OPLS potential\n')
        fileobj.write('# write_lammps' +
                      str(time.asctime(time.localtime(time.time()))))

        # bonds
        if len(btypes):
            fileobj.write('\n# bonds\n')
            fileobj.write('bond_style      harmonic\n')
            for ib, btype in enumerate(btypes):
                fileobj.write('bond_coeff %6d' % (ib + 1))
                for value in self.bonds.nvh[btype]:
                    fileobj.write(' ' + str(value))
                fileobj.write(' # ' + btype + '\n')

        # angles
        if len(atypes):
            fileobj.write('\n# angles\n')
            fileobj.write('angle_style      harmonic\n')
            for ia, atype in enumerate(atypes):
                fileobj.write('angle_coeff %6d' % (ia + 1))
                for value in self.angles.nvh[atype]:
                    fileobj.write(' ' + str(value))
                fileobj.write(' # ' + atype + '\n')

        # dihedrals
        if len(dtypes):
            fileobj.write('\n# dihedrals\n')
            fileobj.write('dihedral_style      opls\n')
            for i, dtype in enumerate(dtypes):
                fileobj.write('dihedral_coeff %6d' % (i + 1))
                for value in self.dihedrals.nvh[dtype]:
                    fileobj.write(' ' + str(value))
                fileobj.write(' # ' + dtype + '\n')

        # Lennard Jones settings
        fileobj.write('\n# L-J parameters\n')
        fileobj.write('pair_style lj/cut/coul/long 10.0 7.4' +
                      ' # consider changing these parameters\n')
        fileobj.write('special_bonds lj/coul 0.0 0.0 0.5\n')
        data = self.data['one']
        for ia, atype in enumerate(atoms.types):
            if len(atype) < 2:
                atype = atype + ' '
            fileobj.write('pair_coeff ' + str(ia + 1) + ' ' + str(ia + 1))
            for value in data[atype][:2]:
                fileobj.write(' ' + str(value))
            fileobj.write(' # ' + atype + '\n')
        fileobj.write('pair_modify shift yes mix geometric\n')

        # Charges
        fileobj.write('\n# charges\n')
        for ia, atype in enumerate(atoms.types):
            if len(atype) < 2:
                atype = atype + ' '
            fileobj.write('set type ' + str(ia + 1))
            fileobj.write(' charge ' + str(data[atype][2]))
            fileobj.write(' # ' + atype + '\n')


class OPLSStructure(Atoms):
    default_map = {
        'BR': 'Br',
        'Be': 'Be',
        'C0': 'Ca',
        'Li': 'Li',
        'Mg': 'Mg',
        'Al': 'Al',
        'Ar': 'Ar'}

    def __init__(self, filename=None, *args, **kwargs):
        Atoms.__init__(self, *args, **kwargs)
        if filename:
            self.read_extended_xyz(filename)
        else:
            self.types = []
            for atom in self:
                if atom.symbol not in self.types:
                    self.types.append(atom.symbol)
                atom.tag = self.types.index(atom.symbol)

    def append(self, atom):
        """Append atom to end."""
        self.extend(Atoms([atom]))

    def read_extended_xyz(self, fileobj, map={}):
        """Read extended xyz file with labeled atoms."""
        atoms = read(fileobj)
        self.set_cell(atoms.get_cell())
        self.set_pbc(atoms.get_pbc())

        types = []
        types_map = {}
        for atom, type in zip(atoms, atoms.get_array('type')):
            if type not in types:
                types_map[type] = len(types)
                types.append(type)
            atom.tag = types_map[type]
            self.append(atom)
        self.types = types

        # copy extra array info
        for name, array in atoms.arrays.items():
            if name not in self.arrays:
                self.new_array(name, array)

    def split_symbol(self, string, translate=default_map):

        if string in translate:
            return translate[string], string
        if len(string) < 2:
            return string, None
        return string[0], string[1]

    def get_types(self):
        return self.types

    def colored(self, elements):
        res = Atoms()
        res.set_cell(self.get_cell())
        for atom in self:
            elem = self.types[atom.tag]
            if elem in elements:
                elem = elements[elem]
            res.append(Atom(elem, atom.position))
        return res

    def update_from_lammps_dump(self, fileobj, check=True):
        atoms = read(fileobj, format='lammps-dump')

        if len(atoms) != len(self):
            raise RuntimeError('Structure in ' + str(fileobj) +
                               ' has wrong length: %d != %d' %
                               (len(atoms), len(self)))

        if check:
            for a, b in zip(self, atoms):
                # check that the atom types match
                if not (a.tag + 1 == b.number):
                    raise RuntimeError('Atoms index %d are of different '
                                       'type (%d != %d)'
                                       % (a.index, a.tag + 1, b.number))

        self.set_cell(atoms.get_cell())
        self.set_positions(atoms.get_positions())
        if atoms.get_velocities() is not None:
            self.set_velocities(atoms.get_velocities())
        # XXX what about energy and forces ???

    def read_connectivities(self, fileobj, update_types=False):
        """Read positions, connectivities, etc.

        update_types: update atom types from the masses
        """
        if isinstance(fileobj, basestring):
            fileobj = open(fileobj, 'r')

        lines = fileobj.readlines()
        lines.pop(0)

        def next_entry():
            line = lines.pop(0).strip()
            if(len(line) > 0):
                lines.insert(0, line)

        def next_key():
            while(len(lines)):
                line = lines.pop(0).strip()
                if(len(line) > 0):
                    lines.pop(0)
                    return line
            return None

        next_entry()
        header = {}
        while(True):
            line = lines.pop(0).strip()
            if len(line):
                w = line.split()
                if len(w) == 2:
                    header[w[1]] = int(w[0])
                else:
                    header[w[1] + ' ' + w[2]] = int(w[0])
            else:
                break

        while(not lines.pop(0).startswith('Atoms')):
            pass
        lines.pop(0)

        natoms = len(self)
        positions = np.empty((natoms, 3))
        for i in range(natoms):
            w = lines.pop(0).split()
            assert(int(w[0]) == (i + 1))
            positions[i] = np.array([float(w[4 + c]) for c in range(3)])
            # print(w, positions[i])

        key = next_key()

        velocities = None
        if key == 'Velocities':
            velocities = np.empty((natoms, 3))
            for i in range(natoms):
                w = lines.pop(0).split()
                assert(int(w[0]) == (i + 1))
                velocities[i] = np.array([float(w[1 + c]) for c in range(3)])
            key = next_key()

        if key == 'Masses':
            ntypes = len(self.types)
            masses = np.empty((ntypes))
            for i in range(ntypes):
                w = lines.pop(0).split()
                assert(int(w[0]) == (i + 1))
                masses[i] = float(w[1])

            if update_types:
                # get the elements from the masses
                # this ensures that we have the right elements
                # even when reading from a lammps dump file
                def newtype(element, types):
                    if len(element) > 1:
                        # can not extend, we are restricted to
                        # two characters
                        return element
                    count = 0
                    for type in types:
                        if type[0] == element:
                            count += 1
                    label = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                    return (element + label[count])

                symbolmap = {}
                typemap = {}
                types = []
                ams = atomic_masses[:]
                ams[np.isnan(ams)] = 0
                for i, mass in enumerate(masses):
                    m2 = (ams - mass)**2
                    symbolmap[self.types[i]] = chemical_symbols[m2.argmin()]
                    typemap[self.types[i]] = newtype(
                        chemical_symbols[m2.argmin()], types)
                    types.append(typemap[self.types[i]])
                for atom in self:
                    atom.symbol = symbolmap[atom.symbol]
                self.types = types

            key = next_key()

        def read_list(key_string, length, debug=False):
            if key != key_string:
                return [], key

            lst = []
            while(len(lines)):
                w = lines.pop(0).split()
                if len(w) > length:
                    lst.append([(int(w[1 + c]) - 1) for c in range(length)])
                else:
                    return lst, next_key()
            return lst, None

        bonds, key = read_list('Bonds', 3)
        angles, key = read_list('Angles', 4)
        dihedrals, key = read_list('Dihedrals', 5, True)

        self.connectivities = {
            'bonds': bonds,
            'angles': angles,
            'dihedrals': dihedrals
        }

        if 'bonds' in header:
            assert(len(bonds) == header['bonds'])
            self.connectivities['bond types'] = list(
                range(header['bond types']))
        if 'angles' in header:
            assert(len(angles) == header['angles'])
            self.connectivities['angle types'] = list(
                range(header['angle types']))
        if 'dihedrals' in header:
            assert(len(dihedrals) == header['dihedrals'])
            self.connectivities['dihedral types'] = list(range(
                header['dihedral types']))
