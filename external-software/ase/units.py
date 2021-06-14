"""ase.units

Physical constants and units derived from CODATA for converting
to and from ase internal units.


"""


from math import pi, sqrt


# the version we actually use
__codata_version__ = '2014'


# Instead of a plain dict, if the units are in the __dict__ of a
# dict subclass, they can be accessed as attributes in a similar way
# to a module.
class Units(dict):
    """Dictionary for units that supports .attribute access."""
    def __init__(self, *args, **kwargs):
        super(Units, self).__init__(*args, **kwargs)
        self.__dict__ = self


# this is the hard-coded CODATA values
# all other units are dynamically derived from these values upon import of the
# module
CODATA = {
    # the "original" CODATA version ase used ever since
    # Constants from Konrad Hinsen's PhysicalQuantities module (1986 CODATA)
    # Add the constant pi used to define the mu0 and hbar here for reference
    # as well
    '1986': {'_c': 299792458.,              # speed of light, m/s
             '_mu0': 4.e-7 * pi,            # permeability of vacuum
             '_Grav': 6.67259e-11,          # gravitational constant
             '_hplanck': 6.6260755e-34,     # Planck constant, J s
             '_e': 1.60217733e-19,          # elementary charge
             '_me': 9.1093897e-31,          # electron mass
             '_mp': 1.6726231e-27,          # proton mass
             '_Nav': 6.0221367e23,          # Avogadro number
             '_k': 1.380658e-23,            # Boltzmann constant, J/K
             '_amu': 1.6605402e-27},         # atomic mass unit, kg

    # CODATA 1998 taken from
    # http://dx.doi.org/10.1103/RevModPhys.72.351
    '1998': {'_c': 299792458.,
             '_mu0': 4.0e-7 * pi,
             '_Grav': 6.673e-11,
             '_hplanck': 6.62606876e-34,
             '_e': 1.602176462e-19,
             '_me': 9.10938188e-31,
             '_mp': 1.67262158e-27,
             '_Nav': 6.02214199e23,
             '_k': 1.3806503e-23,
             '_amu': 1.66053873e-27},

    # CODATA 2002 taken from
    # http://dx.doi.org/10.1103/RevModPhys.77.1
    '2002': {'_c': 299792458.,
             '_mu0': 4.0e-7 * pi,
             '_Grav': 6.6742e-11,
             '_hplanck': 6.6260693e-34,
             '_e': 1.60217653e-19,
             '_me': 9.1093826e-31,
             '_mp': 1.67262171e-27,
             '_Nav': 6.0221415e23,
             '_k': 1.3806505e-23,
             '_amu': 1.66053886e-27},

    # CODATA 2006 taken from
    # http://dx.doi.org/10.1103/RevModPhys.80.633
    '2006': {'_c': 299792458.,
             '_mu0': 4.0e-7 * pi,
             '_Grav': 6.67428e-11,
             '_hplanck': 6.62606896e-34,
             '_e': 1.602176487e-19,
             '_me': 9.10938215e-31,
             '_mp': 1.672621637e-27,
             '_Nav': 6.02214179e23,
             '_k': 1.3806504e-23,
             '_amu': 1.660538782e-27},

    # CODATA 2010 taken from
    # http://dx.doi.org/10.1103/RevModPhys.84.1527
    '2010': {'_c': 299792458.,
             '_mu0': 4.0e-7 * pi,
             '_Grav': 6.67384e-11,
             '_hplanck': 6.62606957e-34,
             '_e': 1.602176565e-19,
             '_me': 9.10938291e-31,
             '_mp': 1.672621777e-27,
             '_Nav': 6.02214129e23,
             '_k': 1.3806488e-23,
             '_amu': 1.660538921e-27},

    # CODATA 2014 taken from
    # http://arxiv.org/pdf/1507.07956.pdf
    '2014': {'_c': 299792458.,
             '_mu0': 4.0e-7 * pi,
             '_Grav': 6.67408e-11,
             '_hplanck': 6.626070040e-34,
             '_e': 1.6021766208e-19,
             '_me': 9.10938356e-31,
             '_mp': 1.672621898e-27,
             '_Nav': 6.022140857e23,
             '_k': 1.38064852e-23,
             '_amu': 1.660539040e-27}}


def create_units(codata_version):
    """
    Function that creates a dictionary containing all units previously hard
    coded in ase.units depending on a certain CODATA version. Note that
    returned dict has attribute access it can be used in place of the module
    or to update your local or global namespace.

    Parameters:

    codata_version: str
        The CODATA version to be used. Implemented are

        * '1986'
        * '1998'
        * '2002'
        * '2006'
        * '2010'
        * '2014'

    Returns:

    units: dict
        Dictionary that contains all formerly hard coded variables from
        ase.units as key-value pairs. The dict supports attribute access.

    Raises:

    NotImplementedError
        If the required CODATA version is not known.
    """

    try:
        u = Units(CODATA[codata_version])
    except KeyError:
        raise NotImplementedError('CODATA version "{0}" not implemented'
                                  .format(__codata_version__))

    # derived from the CODATA values
    u['_eps0'] = (1 / u['_mu0'] / u['_c']**2)  # permittivity of vacuum
    u['_hbar'] = u['_hplanck'] / (2 * pi)  # Planck constant / 2pi, J s

    u['Ang'] = u['Angstrom'] = 1.0
    u['nm'] = 10.0
    u['Bohr'] = (4e10 * pi * u['_eps0'] * u['_hbar']**2 /
                 u['_me'] / u['_e']**2)  # Bohr radius

    u['eV'] = 1.0
    u['Hartree'] = (u['_me'] * u['_e']**3 / 16 / pi**2 /
                    u['_eps0']**2 / u['_hbar']**2)
    u['kJ'] = 1000.0 / u['_e']
    u['kcal'] = 4.184 * u['kJ']
    u['mol'] = u['_Nav']
    u['Rydberg'] = 0.5 * u['Hartree']
    u['Ry'] = u['Rydberg']
    u['Ha'] = u['Hartree']

    u['second'] = 1e10 * sqrt(u['_e'] / u['_amu'])
    u['fs'] = 1e-15 * u['second']

    u['kB'] = u['_k'] / u['_e']  # Boltzmann constant, eV/K

    u['Pascal'] = (1 / u['_e']) / 1e30  # J/m^3
    u['GPa'] = 1e9 * u['Pascal']

    u['Debye'] = 1.0 / 1e11 / u['_e'] / u['_c']
    u['alpha'] = (u['_e']**2 / (4 * pi * u['_eps0']) /
                  u['_hbar'] / u['_c'])  # fine structure constant
    u['invcm'] = (100 * u['_c'] * u['_hplanck'] /
                  u['_e'])  # cm^-1 energy unit

    # Derived atomic units that have no assigned name:
    # atomic unit of time, s:
    u['_aut'] = u['_hbar'] / (u['alpha']**2 * u['_me'] * u['_c']**2)
    # atomic unit of velocity, m/s:
    u['_auv'] = u['_e']**2 / u['_hbar'] / (4 * pi * u['_eps0'])
    # atomic unit of force, N:
    u['_auf'] = u['alpha']**3 * u['_me']**2 * u['_c']**3 / u['_hbar']
    # atomic unit of pressure, Pa:
    u['_aup'] = u['alpha']**5 * u['_me']**4 * u['_c']**5 / u['_hbar']**3

    u['AUT'] = u['second'] * u['_aut']

    # SI units
    u['m'] = 1e10 * u['Ang']  # metre
    u['kg'] = 1. / u['_amu']  # kilogram
    u['s'] = u['second']  # second
    u['A'] = 1.0 / u['_e'] / u['s']  # ampere
    # derived
    u['J'] = u['kJ'] / 1000  # Joule = kg * m**2 / s**2
    u['C'] = 1.0 / u['_e']  # Coulomb = A * s

    return u


# Define all the expected symbols with dummy values so that introspection
# will know that they exist when the module is imported, even though their
# values are immediately overwritten.
# pylint: disable=invalid-name
(_Grav, _Nav, _amu, _auf, _aup, _aut, _auv, _c, _e, _eps0,
 _hbar, _hplanck, _k, _me, _mp, _mu0, alpha, eV, fs, invcm,
 kB, kJ, kcal, kg, m, mol, nm, s, second, A, AUT, Ang, Angstrom,
 Bohr, C, Debye, GPa, Ha, Hartree, J, Pascal, Ry, Rydberg) = [0.0] * 43

# Now update the module scope:
globals().update(create_units(__codata_version__))
