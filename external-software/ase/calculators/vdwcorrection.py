"""van der Waals correction schemes for DFT"""
from __future__ import print_function
import numpy as np
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator
from ase.utils import convert_string_to_fd
from scipy.special import erfinv, erfc
from ase.neighborlist import neighbor_list


# dipole polarizabilities and C6 values from
# X. Chu and A. Dalgarno, J. Chem. Phys. 121 (2004) 4083
# atomic units, a_0^3
vdWDB_Chu04jcp = {
    # Element: [alpha, C6]; units [Bohr^3, Hartree * Bohr^6]
    'H': [4.5, 6.5],  # [exact, Tkatchenko PRL]
    'He': [1.38, 1.42],
    'Li': [164, 1392],
    'Be': [38, 227],
    'B': [21, 99.5],
    'C': [12, 46.6],
    'N': [7.4, 24.2],
    'O': [5.4, 15.6],
    'F': [3.8, 9.52],
    'Ne': [2.67, 6.20],
    'Na': [163, 1518],
    'Mg': [71, 626],
    'Al': [60, 528],
    'Si': [37, 305],
    'P': [25, 185],
    'S': [19.6, 134],
    'Cl': [15, 94.6],
    'Ar': [11.1, 64.2],
    'Ca': [160, 2163],
    'Sc': [120, 1383],
    'Ti': [98, 1044],
    'V': [84, 832],
    'Cr': [78, 602],
    'Mn': [63, 552],
    'Fe': [56, 482],
    'Co': [50, 408],
    'Ni': [48, 373],
    'Cu': [42, 253],
    'Zn': [40, 284],
    'As': [29, 246],
    'Se': [25, 210],
    'Br': [20, 162],
    'Kr': [16.7, 130],
    'Sr': [199, 3175],
    'Te': [40, 445],
    'I': [35, 385]}

vdWDB_alphaC6 = vdWDB_Chu04jcp

# dipole polarizabilities and C6 values from
# V. G. Ruiz et al. Phys. Rev. Lett 108 (2012) 146103
# atomic units, a_0^3
vdWDB_Ruiz12prl = {
    'Ag' : [50.6, 339],
    'Au' : [36.5, 298],
    'Pd' : [23.7, 158],
    'Pt' : [39.7, 347],
}

vdWDB_alphaC6.update(vdWDB_Ruiz12prl)

# C6 values and vdW radii from
# S. Grimme, J Comput Chem 27 (2006) 1787-1799
vdWDB_Grimme06jcc = {
    # Element: [C6, R0]; units [J nm^6 mol^{-1}, Angstrom]
    'H': [0.14, 1.001],
    'He': [0.08, 1.012],
    'Li': [1.61, 0.825],
    'Be': [1.61, 1.408],
    'B': [3.13, 1.485],
    'C': [1.75, 1.452],
    'N': [1.23, 1.397],
    'O': [0.70, 1.342],
    'F': [0.75, 1.287],
    'Ne': [0.63, 1.243],
    'Na': [5.71, 1.144],
    'Mg': [5.71, 1.364],
    'Al': [10.79, 1.639],
    'Si': [9.23, 1.716],
    'P': [7.84, 1.705],
    'S': [5.57, 1.683],
    'Cl': [5.07, 1.639],
    'Ar': [4.61, 1.595],
    'K': [10.80, 1.485],
    'Ca': [10.80, 1.474],
    'Sc': [10.80, 1.562],
    'Ti': [10.80, 1.562],
    'V': [10.80, 1.562],
    'Cr': [10.80, 1.562],
    'Mn': [10.80, 1.562],
    'Fe': [10.80, 1.562],
    'Co': [10.80, 1.562],
    'Ni': [10.80, 1.562],
    'Cu': [10.80, 1.562],
    'Zn': [10.80, 1.562],
    'Ga': [16.99, 1.650],
    'Ge': [17.10, 1.727],
    'As': [16.37, 1.760],
    'Se': [12.64, 1.771],
    'Br': [12.47, 1.749],
    'Kr': [12.01, 1.727],
    'Rb': [24.67, 1.628],
    'Sr': [24.67, 1.606],
    'Y-Cd': [24.67, 1.639],
    'In': [37.32, 1.672],
    'Sn': [38.71, 1.804],
    'Sb': [38.44, 1.881],
    'Te': [31.74, 1.892],
    'I': [31.50, 1.892],
    'Xe': [29.99, 1.881]}


# Optimal range parameters sR for different XC functionals
# to be used with the Tkatchenko-Scheffler scheme
# Reference: M.A. Caro arXiv:1704.00761 (2017)
sR_opt={'PBE': 0.940,
        'RPBE': 0.590,
        'revPBE': 0.585,
        'PBEsol': 1.055,
        'BLYP': 0.625,
        'AM05': 0.840,
        'PW91': 0.965}


def get_logging_file_descriptor(calculator):
    if hasattr(calculator, 'log'):
        fd = calculator.log
        if hasattr(fd, 'write'):
            return fd
        if hasattr(fd, 'fd'):
            return fd.fd
    if hasattr(calculator, 'txt'):
        return calculator.txt


class vdWTkatchenko09prl(Calculator):
    """vdW correction after Tkatchenko and Scheffler PRL 102 (2009) 073005."""
    implemented_properties = ['energy', 'forces']

    def __init__(self,
                 hirshfeld=None, vdwradii=None, calculator=None,
                 Rmax=10.,  # maximal radius for periodic calculations
                 Ldecay=1., # decay length for the smoothing in periodic calculations
                 vdWDB_alphaC6=vdWDB_alphaC6,
                 txt=None, sR=None
                ):
        """Constructor

        Parameters
        ==========
        hirshfeld: the Hirshfeld partitioning object
        calculator: the calculator to get the PBE energy
        """
        self.hirshfeld = hirshfeld
        if calculator is None:
            self.calculator = self.hirshfeld.get_calculator()
        else:
            self.calculator = calculator

        if txt is None:
            txt = get_logging_file_descriptor(self.calculator)
        self.txt = convert_string_to_fd(txt)

        self.vdwradii = vdwradii
        self.vdWDB_alphaC6 = vdWDB_alphaC6
        self.Rmax = Rmax
        self.Ldecay = Ldecay
        self.atoms = None

        if sR is None:
            try:
                xc_name = self.calculator.get_xc_functional()
                self.sR = sR_opt[xc_name]
            except KeyError:
                raise ValueError('Tkatchenko-Scheffler dispersion correction not implemented for %s functional' % xc_name)
        else:
            self.sR = sR
        self.d = 20

        Calculator.__init__(self)

    @property
    def implemented_properties(self):
        return self.calculator.implemented_properties

    def calculation_required(self, atoms, quantities):
        if self.calculator.calculation_required(atoms, quantities):
            return True
        for quantity in quantities:
            if quantity not in self.results:
                return True
        return False

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=[]):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.update(atoms, properties)

    def update(self, atoms=None, properties=['energy', 'forces']):
        if not self.calculation_required(atoms, properties):
            return

        if atoms is None:
            atoms = self.calculator.get_atoms()

        properties = list(properties)
        for name in 'energy', 'forces':
            if name not in properties:
                properties.append(name)

        for name in properties:
            self.results[name] = self.calculator.get_property(name, atoms)
        self.atoms = atoms.copy()

        if self.vdwradii is not None:
            # external vdW radii
            vdwradii = self.vdwradii
            assert(len(atoms) == len(vdwradii))
        else:
            vdwradii = []
            for atom in atoms:
                self.vdwradii.append(vdWDB_Grimme06jcc[atom.symbol][1])

        if self.hirshfeld is None:
            volume_ratios = [1.] * len(atoms)
        elif hasattr(self.hirshfeld, '__len__'):  # a list
            assert(len(atoms) == len(self.hirshfeld))
            volume_ratios = self.hirshfeld
        else:  # should be an object
            self.hirshfeld.initialize()
            volume_ratios = self.hirshfeld.get_effective_volume_ratios()

        # correction for effective C6
        na = len(atoms)
        C6eff_a = np.empty((na))
        alpha_a = np.empty((na))
        R0eff_a = np.empty((na))
        for a, atom in enumerate(atoms):
            # free atom values
            alpha_a[a], C6eff_a[a] = self.vdWDB_alphaC6[atom.symbol]
            # correction for effective C6
            C6eff_a[a] *= Hartree * volume_ratios[a]**2 * Bohr**6
            R0eff_a[a] = vdwradii[a] * volume_ratios[a]**(1 / 3.)
        C6eff_aa = np.empty((na, na))
        for a in range(na):
            for b in range(a, na):
                C6eff_aa[a, b] = (2 * C6eff_a[a] * C6eff_a[b] /
                                  (alpha_a[b] / alpha_a[a] * C6eff_a[a] +
                                   alpha_a[a] / alpha_a[b] * C6eff_a[b]))
                C6eff_aa[b, a] = C6eff_aa[a, b]

        # New implementation by Miguel Caro (complaints etc to mcaroba@gmail.com)
        # If all 3 PBC are False, we do the summation over the atom
        # pairs in the simulation box. If any of them is True, we
        # use the cutoff radius instead
        pbc_c = atoms.get_pbc()
        EvdW = 0.0
        forces = 0. * self.results['forces']
        # PBC: we build a neighbor list according to the Reff criterion
        if pbc_c.any():
            # Effective cutoff radius
            tol = 1.e-5
            Reff = self.Rmax + self.Ldecay * erfinv(1. - 2.*tol)
            # Build list of neighbors
            n_list = neighbor_list(quantities = "ijdDS",
                                   a = atoms,
                                   cutoff = Reff,
                                   self_interaction=False)
            atom_list = [[] for _ in range(0, len(atoms))]
            d_list = [[] for _ in range(0, len(atoms))]
            v_list = [[] for _ in range(0, len(atoms))]
            #r_list = [[] for _ in range(0, len(atoms))]
            # Look for neighbor pairs
            for k in range(0, len(n_list[0])):
                i = n_list[0][k]
                j = n_list[1][k]
                dist = n_list[2][k]
                vect = n_list[3][k] # vect is the distance rj - ri
                #repl = n_list[4][k]
                if j >= i:
                    atom_list[i].append( j )
                    d_list[i].append( dist )
                    v_list[i].append( vect )
                    #r_list[i].append( repl )
        # Not PBC: we loop over all atoms in the unit cell only
        else:
            atom_list = []
            d_list = []
            v_list = []
            #r_list = []
            # Do this to avoid double counting
            for i in range(0, len(atoms)):
                atom_list.append( range(i+1, len(atoms)) )
                d_list.append( [atoms.get_distance(i, j) for j in range(i+1, len(atoms))] )
                v_list.append( [atoms.get_distance(i, j, vector=True) for j in range(i+1, len(atoms))] )
                #r_list.append( [[0,0,0] for j in range(i+1, len(atoms))]) # No PBC means we are in the same cell
        # Here goes the calculation, valid with and without PBC because we loop over
        # independent pairwise *interactions*
        for i in range(0,len(atoms)):
            #for j, r, vect, repl in zip(atom_list[i], d_list[i], v_list[i], r_list[i]):
            for j, r, vect in zip(atom_list[i], d_list[i], v_list[i]):
                r6 = r**6
                Edamp, Fdamp = self.damping(r,
                                            R0eff_a[i],
                                            R0eff_a[j],
                                            d=self.d,
                                            sR=self.sR)
                if pbc_c.any():
                    smooth = 0.5 * erfc((r - self.Rmax) / self.Ldecay)
                    smooth_der = -1. / np.sqrt(np.pi) / self.Ldecay * np.exp(
                                  -((r - self.Rmax) / self.Ldecay)**2 )
                else:
                    smooth = 1.
                    smooth_der = 0.
                # Here we compute the contribution to the energy
                # Self interactions (only possible in PBC) are double counted. We correct it here
                if i == j:
                    EvdW -= (Edamp * C6eff_aa[i, j] / r6) / 2. * smooth
                else:
                    EvdW -= (Edamp * C6eff_aa[i, j] / r6) * smooth
                # Here we compute the contribution to the forces
                # We neglect the C6eff contribution to the forces (which can actually be larger
                # than the other contributions)
                # Self interactions do not contribute to the forces
                if i != j:
                    # Force on i due to j
                    force_ij = -(
                                  (Fdamp - 6 * Edamp / r) * C6eff_aa[i, j] / r6 * smooth
                                 +(Edamp * C6eff_aa[i, j] / r6) * smooth_der
                                ) * vect / r
                    # Forces go both ways for every interaction
                    forces[i] += force_ij
                    forces[j] -= force_ij
        self.results['energy'] += EvdW
        self.results['forces'] += forces



        if self.txt:
            print(('\n' + self.__class__.__name__), file=self.txt)
            print('vdW correction: %g' % (EvdW), file=self.txt)
            print('Energy:         %g' % self.results['energy'],
                  file=self.txt)
            print('\nForces in eV/Ang:', file=self.txt)
            symbols = self.atoms.get_chemical_symbols()
            for ia, symbol in enumerate(symbols):
                print('%3d %-2s %10.5f %10.5f %10.5f' %
                      ((ia, symbol) + tuple(self.results['forces'][ia])),
                      file=self.txt)
            self.txt.flush()

    def damping(self, RAB, R0A, R0B,
                d=20,   # steepness of the step function for PBE
                sR=0.94):
        """Damping factor.

        Standard values for d and sR as given in
        Tkatchenko and Scheffler PRL 102 (2009) 073005."""
        scale = 1.0 / (sR * (R0A + R0B))
        x = RAB * scale
        chi = np.exp(-d * (x - 1.0))
        return 1.0 / (1.0 + chi), d * scale * chi / (1.0 + chi)**2
