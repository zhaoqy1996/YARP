from __future__ import print_function

import traceback

import numpy as np

from ase import Atoms

# This module the InterfaceTester class which tests the extent to
# which an object behaves like an ASE calculator.
#
# It runs the ASE interface methods and performs a few very basic checks
# on the returned objects, then writes a list of errors.
#
# Future improvements: Check that arrays are padded correctly, verify
# more information about shapes, maybe do some complicated state
# changes and check that the calculator behaves properly.


class Args:
    # This class is just syntantical sugar to pass args and kwargs
    # easily when testing many methods after one another.
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def unpack(self):
        return self.args, self.kwargs


args = Args


class Error:
    def __init__(self, code, error, methname):
        self.code = code
        self.error = error
        self.methname = methname
        self.txt = traceback.format_exc()
        self.callstring = None


class InterfaceChecker:
    def __init__(self, obj):
        self.obj = obj
        self.returnvalues = {}
        self.errors = []

    def _check(self, methname, args=args(), rtype=None):
        args, kwargs = args.unpack()

        try:
            meth = getattr(self.obj, methname)
        except AttributeError as err:
            return Error('MISSING', err, methname)

        try:
            value = meth(*args, **kwargs)
        except NotImplementedError as err:
            return Error('not implemented', err, methname)
        except Exception as err:
            return Error(err.__class__.__name__, err, methname)
        else:
            self.returnvalues[methname] = value

        if rtype is not None:
            if not isinstance(value, rtype):
                return Error('TYPE', TypeError('got %s but expected %s'
                                               % (type(value), rtype)),
                             methname)
        return None

    def check(self, methname, args=args(), rtype=None):
        pargs, kwargs = args.unpack()
        
        def get_string_repr(obj):
            if isinstance(obj, Atoms):
                return '<Atoms>'
            else:
                return repr(obj)

        pargsstrs = [get_string_repr(obj) for obj in pargs]
        kwargsstrs = ['%s=%s' % (key, get_string_repr(kwargs[key]))
                      for key in sorted(kwargs)]
        pargskwargsstr = ', '.join(pargsstrs + kwargsstrs)
        err = self._check(methname, args, rtype)
        callstring = '%s(%s)' % (methname, pargskwargsstr)
        if err is None:
            status = 'ok'
        else:
            status = err.code
            err.callstring = callstring
            self.errors.append(err)
        print('%16s : %s' % (status, callstring))


def check_interface(calc):
    tester = InterfaceChecker(calc)
    c = tester.check

    system = calc.get_atoms()

    # Methods specified by ase.calculators.interface.Calculator
    c('get_atoms', rtype=Atoms)
    c('get_potential_energy', rtype=float)
    c('get_potential_energy', args(atoms=system), rtype=float)
    c('get_potential_energy', args(atoms=system, force_consistent=True),
      rtype=float)
    c('get_forces', args(system), np.ndarray)
    c('get_stress', args(system), np.ndarray)
    c('calculation_required', args(system, []), rtype=bool)

    # Methods specified by ase.calculators.interface.DFTCalculator
    c('get_number_of_bands', rtype=int)
    c('get_xc_functional', rtype=str)
    c('get_bz_k_points', rtype=np.ndarray)
    c('get_number_of_spins', rtype=int)
    c('get_spin_polarized', rtype=bool)
    c('get_ibz_k_points', rtype=np.ndarray)
    c('get_k_point_weights', rtype=np.ndarray)

    for meth in ['get_pseudo_density', 'get_effective_potential']:
        c(meth, rtype=np.ndarray)
        c(meth, args(spin=0, pad=False), rtype=np.ndarray)
        spinpol = tester.returnvalues.get('get_spin_polarized')
        if spinpol:
            c(meth, args(spin=1, pad=True), rtype=np.ndarray)

    for pad in [False, True]:
        c('get_pseudo_density', args(spin=None, pad=pad), rtype=np.ndarray)

    for meth in ['get_pseudo_density', 'get_effective_potential']:
        c(meth, args(spin=0, pad=False), rtype=np.ndarray)
        spinpol = tester.returnvalues.get('get_spin_polarized')
        if spinpol:
            c(meth, args(spin=1, pad=True), rtype=np.ndarray)

    nbands = tester.returnvalues.get('get_number_of_bands')
    if nbands is not None and isinstance(nbands, int) and nbands > 0:
        c('get_pseudo_wave_function', args(band=nbands - 1), rtype=np.ndarray)
        c('get_pseudo_wave_function',
          args(band=nbands - 1, kpt=0, spin=0, broadcast=False, pad=False),
          rtype=np.ndarray)
    c('get_eigenvalues', args(kpt=0, spin=0), rtype=np.ndarray)
    c('get_occupation_numbers', args(kpt=0, spin=0), rtype=np.ndarray)
    c('get_fermi_level', rtype=float)
    # c('initial_wanner', ........) what the heck?
    # c('get_wannier_localization_matrix', ...)  No.
    c('get_magnetic_moment', args(atoms=system), rtype=float)
    # c('get_number_of_grid_points', rtype=tuple) # Hmmmm.  Not for now...

    # Optional methods sometimes invoked by ase.atoms.Atoms
    c('get_magnetic_moments', rtype=np.ndarray)
    c('get_charges', rtype=np.ndarray)
    c('get_potential_energies', rtype=np.ndarray)
    c('get_stresses', rtype=np.ndarray)
    c('get_dipole_moment', rtype=np.ndarray)

    real_errs = [err for err in tester.errors
                 if not isinstance(err.error, NotImplementedError)]
    if len(real_errs) > 0:
        print()
        print('Errors')
        print('======')
        for err in tester.errors:
            print('%s: %s' % (err.code, err.callstring))
            print(err.txt)
            print()

    return tester.errors


def main_gpaw():
    from gpaw import GPAW
    from ase.build import molecule
    system = molecule('H2')
    system.center(vacuum=1.5)
    system.pbc = 1
    calc = GPAW(h=0.3, mode='lcao', txt=None)
    system.set_calculator(calc)
    system.get_potential_energy()
    check_interface(calc)


def main_octopus():
    from octopus import Octopus
    from ase.build import molecule
    system = molecule('H2')
    system.center(vacuum=1.5)
    system.pbc = 1
    calc = Octopus()
    system.set_calculator(calc)
    system.get_potential_energy()
    check_interface(calc)
    

if __name__ == '__main__':
    main_gpaw()
