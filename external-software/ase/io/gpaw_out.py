import re
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.singlepoint import SinglePointKPoint
from ase.utils import basestring


def read_gpaw_out(fileobj, index):
    notfound = []

    def index_startswith(lines, string):
        if not isinstance(string, basestring):
            # assume it's a list
            for entry in string:
                try:
                    return index_startswith(lines, entry)
                except ValueError:
                    pass
            raise ValueError

        if string in notfound:
            raise ValueError
        for i, line in enumerate(lines):
            if line.startswith(string):
                return i
        notfound.append(string)
        raise ValueError

    def index_pattern(lines, pattern):
        repat = re.compile(pattern)
        if pattern in notfound:
            raise ValueError
        for i, line in enumerate(lines):
            if repat.match(line):
                return i
        notfound.append(pattern)
        raise ValueError

    def read_forces(lines, ii):
        f = []
        for i in range(ii + 1, ii + 1 + len(atoms)):
            try:
                x, y, z = lines[i].split()[-3:]
                f.append((float(x), float(y), float(z)))
            except (ValueError, IndexError) as m:
                raise IOError('Malformed GPAW log file: %s' % m)
        return f, i

    lines = [line.lower() for line in fileobj.readlines()]
    images = []
    while True:
        try:
            i = index_startswith(lines, 'reference energy:')
            Eref = float(lines[i].split()[-1])
        except ValueError:
            Eref = None
        try:
            i = lines.index('unit cell:\n')
        except ValueError:
            pass
        else:
            if lines[i + 2].startswith('  -'):
                del lines[i + 2]  # old format
            cell = []
            pbc = []
            for line in lines[i + 2:i + 5]:
                words = line.split()
                if len(words) == 5:  # old format
                    cell.append(float(words[2]))
                    pbc.append(words[1] == 'yes')
                else:                # new format with GUC
                    cell.append([float(word) for word in words[3:6]])
                    pbc.append(words[2] == 'yes')

        try:
            i = lines.index('positions:\n')
        except ValueError:
            break

        symbols = []
        positions = []
        for line in lines[i + 1:]:
            words = line.split()
            if len(words) < 5:
                break
            n, symbol, x, y, z = words[:5]
            symbols.append(symbol.split('.')[0].title())
            positions.append([float(x), float(y), float(z)])
        if len(symbols):
            atoms = Atoms(symbols=symbols, positions=positions,
                          cell=cell, pbc=pbc)
        else:
            atoms = Atoms(cell=cell, pbc=pbc)
        lines = lines[i + 5:]
        try:
            ii = index_pattern(lines, '\\d+ k-point')
            word = lines[ii].split()
            kx = int(word[2])
            ky = int(word[4])
            kz = int(word[6])
            bz_kpts = (kx, ky, kz)
            ibz_kpts = int(lines[ii + 1].split()[0])
        except (ValueError, TypeError, IndexError):
            bz_kpts = None
            ibz_kpts = None

        try:
            i = index_startswith(lines, 'energy contributions relative to')
        except ValueError:
            e = energy_contributions = None
        else:
            energy_contributions = {}
            for line in lines[i + 2:i + 8]:
                fields = line.split(':')
                energy_contributions[fields[0]] = float(fields[1])
            line = lines[i + 10]
            assert (line.startswith('zero kelvin:') or
                    line.startswith('extrapolated:'))
            e = float(line.split()[-1])

        try:
            ii = index_pattern(lines, '(fixed )?fermi level(s)?:')
        except ValueError:
            eFermi = None
        else:
            fields = lines[ii].split()
            try:
                def strip(string):
                    for rubbish in '[],':
                        string = string.replace(rubbish, '')
                    return string
                eFermi = [float(strip(fields[-2])),
                          float(strip(fields[-1]))]
            except ValueError:
                eFermi = float(fields[-1])

        # read Eigenvalues and occupations
        ii1 = ii2 = 1e32
        try:
            ii1 = index_startswith(lines, ' band   eigenvalues  occupancy')
        except ValueError:
            pass
        try:
            ii2 = index_startswith(lines, ' band  eigenvalues  occupancy')
        except ValueError:
            pass
        ii = min(ii1, ii2)
        if ii == 1e32:
            kpts = None
        else:
            ii += 1
            words = lines[ii].split()
            vals = []
            while(len(words) > 2):
                vals.append([float(w) for w in words])
                ii += 1
                words = lines[ii].split()
            vals = np.array(vals).transpose()
            kpts = [SinglePointKPoint(1, 0, 0)]
            kpts[0].eps_n = vals[1]
            kpts[0].f_n = vals[2]
            if vals.shape[0] > 3:
                kpts.append(SinglePointKPoint(1, 1, 0))
                kpts[1].eps_n = vals[3]
                kpts[1].f_n = vals[4]
        # read charge
        try:
            ii = index_startswith(lines, 'total charge:')
        except ValueError:
            q = None
        else:
            q = float(lines[ii].split()[2])
        # read dipole moment
        try:
            ii = index_startswith(lines, 'dipole moment:')
        except ValueError:
            dipole = None
        else:
            line = lines[ii]
            for x in '()[],':
                line = line.replace(x, '')
            dipole = np.array([float(c) for c in line.split()[2:5]])

        try:
            ii = index_startswith(lines, 'local magnetic moments')
        except ValueError:
            magmoms = None
        else:
            magmoms = []
            for j in range(ii + 1, ii + 1 + len(atoms)):
                magmom = lines[j].split()[-1].rstrip(')')
                magmoms.append(float(magmom))

        try:
            ii = lines.index('forces in ev/ang:\n')
        except ValueError:
            f = None
        else:
            f, i = read_forces(lines, ii)

        try:
            ii = index_startswith(lines, 'vdw correction:')
        except ValueError:
            pass
        else:
            line = lines[ii + 1]
            assert line.startswith('energy:')
            e = float(line.split()[-1])
            f, i = read_forces(lines, ii + 3)

        if len(images) > 0 and e is None:
            break

        if q is not None and len(atoms) > 0:
            n = len(atoms)
            atoms.set_initial_charges([q / n] * n)
        if magmoms is not None:
            atoms.set_initial_magnetic_moments(magmoms)
        if e is not None or f is not None:
            calc = SinglePointDFTCalculator(atoms, energy=e, forces=f,
                                            dipole=dipole, magmoms=magmoms,
                                            efermi=eFermi,
                                            bzkpts=bz_kpts, ibzkpts=ibz_kpts)
            calc.eref = Eref
            calc.name = 'gpaw'
            if energy_contributions is not None:
                calc.energy_contributions = energy_contributions
            if kpts is not None:
                calc.kpts = kpts
            atoms.set_calculator(calc)

        images.append(atoms)
        lines = lines[i:]

    if len(images) == 0:
        raise IOError('Corrupted GPAW-text file!')

    return images[index]
