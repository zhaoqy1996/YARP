"""
The ASE Calculator for OpenMX <http://www.openmx-square.org>: Python interface
to the software package for nano-scale material simulations based on density
functional theories.
    Copyright (C) 2018 Jae Hwan Shim and JaeJun Yu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 2.1 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with ASE.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
from ase.calculators.calculator import Parameters
from ase.calculators.openmx.default_settings import default_dictionary
from ase.units import Ha, Ry


# Keys that have length 3
tuple_integer_keys = [
    'scf.Ngrid',
    'scf.Kgrid',
    'Dos.Kgrid',
]
tuple_float_keys = [
    'scf.Electric.Field',
    'scf.fixed.grid'
]
tuple_bool_keys = [

]
integer_keys = [
    'level.of.stdout',
    'level.of.fileout',
    'Species.Number',
    'Atoms.Number',
    'scf.maxIter',
    'scf.Mixing.History',
    'scf.Mixing.StartPulay',
    'scf.Mixing.EveryPulay',
    '1DFFT.NumGridK',
    '1DFFT.NumGridR',
    'orbitalOpt.scf.maxIter',
    'orbitalOpt.Opt.maxIter',
    'orbitalOpt.Opt.Method',
    'orbitalOpt.HistoryPulay',
    'Num.CntOrb.Atoms',
    'orderN.KrylovH.order',
    'orderN.KrylovS.order',
    'MD.maxIter',
    'MD.Opt.DIIS.History',
    'MD.Opt.StartDIIS',
    'Band.Nkpath',
    'num.HOMOs',
    'num.LUMOs',
    'MO.Nkpoint',
    'MD.Current.Iter'
    ]
float_keys = [
    'scf.Constraint.NC.Spin.v',
    'scf.ElectronicTemperature',
    'scf.energycutoff',
    'scf.Init.Mixing.Weight',
    'scf.Min.Mixing.Weight',
    'scf.Max.Mixing.Weight',
    'scf.Kerker.factor',
    'scf.criterion',
    'scf.system.charge',
    '1DFFT.EnergyCutoff',
    'orbitalOpt.SD.step',
    'orbitalOpt.criterion',
    'orderN.HoppingRanges',
    'MD.TimeStep',
    'MD.Opt.criterion',
    'NH.Mass.HeatBath',
    'scf.NC.Mag.Field.Spin',
    'scf.NC.Mag.Field.Orbital',
]
string_keys = [
    'System.CurrentDirectory',
    'System.Name',
    'DATA.PATH',
    'Atoms.SpeciesAndCoordinates.Unit',
    'Atoms.UnitVectors.Unit',
    'scf.XcType',
    'scf.SpinPolarization',
    'scf.Hubbard.Occupation',
    'scf.EigenvalueSolver',
    'scf.Mixing.Type',
    'orbitalOpt.Method',
    'orbitalOpt.StartPulay',
    'MD.Type',
    'Wannier.Initial.Projectors.Unit'
]
bool_keys = [
    'scf.partialCoreCorrection',
    'scf.Hubbard.U',
    'scf.Constraint.NC.Spin',
    'scf.ProExpn.VNA',
    'scf.SpinOrbit.Coupling'
    'CntOrb.fileout',
    'orderN.Exact.Inverse.S',
    'orderN.Recalc.Buffer',
    'orderN.Expand.Core',
    'Band.Dispersion',
    'scf.restart',
    'MO.fileout',
    'Dos.fileout',
    'HS.fileout',
    'Voronoi.charge',
    'scf.NC.Zeeman.Spin',
    'scf.stress.tensor'
]
list_int_keys = []
list_bool_keys = []
list_float_keys = [
    'Dos.Erange',
]
matrix_keys = [
    'Definition.of.Atomic.Species',
    'Atoms.SpeciesAndCoordinates',
    'Atoms.UnitVectors',
    'Hubbard.U.values',
    'Atoms.Cont.Orbitals',
    'MD.Fixed.XYZ',
    'MD.TempControl',
    'MD.Init.Velocity',
    'Band.KPath.UnitCell',
    'Band.kpath',
    'MO.kpoint',
    'Wannier.Initial.Projectors'
]
unit_dat_keywords = {
    'Hubbard.U.Values': 'eV',
    'scf.Constraint.NC.Spin.v': 'eV',
    'scf.ElectronicTemperature': 'K',
    'scf.energycutoff': 'Ry',
    'scf.criterion': 'Ha',
    'scf.Electric.Field': 'GV / m',
    'OneDFFT.EnergyCutoff': 'Ry',
    'orbitalOpt.criterion': '(Ha/Borg)**2',
    'MD.Opt.criterion': 'Ha/Bohr',
    'MD.TempControl': 'K',
    'NH.Mass.HeatBath': '_amu',
    'MD.Init.Velocity': 'm/s',
    'Dos.Erange': 'eV',
    'scf.NC.Mag.Field.Spin': 'Tesla',
    'scf.NC.Mag.Field.Orbital': 'Tesla'
                     }


omx_parameter_defaults = dict(
    scf_ngrid=None,
    scf_kgrid=None,
    dos_kgrid=None,
    scf_electric_field=None,
    level_of_stdout=None,
    level_of_fileout=None,
    species_number=None,
    atoms_number=None,
    scf_maxiter=None,
    scf_mixing_history=None,
    scf_mixing_startpulay=None,
    scf_mixing_everypulay=None,
    onedfft_numgridk=None,  # 1Dfft
    onedfft_numgridr=None,  # 1Dfft
    orbitalopt_scf_maxiter=None,
    orbitalopt_opt_maxiter=None,
    orbitalopt_opt_method=None,
    orbitalopt_historypulay=None,
    num_cntorb_atoms=None,
    ordern_krylovh_order=None,
    ordern_krylovs_order=None,
    md_maxiter=None,
    md_opt_diis_history=None,
    md_opt_startdiis=None,
    band_nkpath=None,
    num_homos=None,
    num_lumos=None,
    mo_nkpoint=None,
    md_current_iter=None,
    scf_constraint_nc_spin_v=None,
    scf_electronictemperature=None,
    scf_fixed_grid=None,
    scf_energycutoff=None,
    scf_init_mixing_weight=None,
    scf_min_mixing_weight=None,
    scf_max_mixing_weight=None,
    scf_kerker_factor=None,
    scf_criterion=None,
    scf_system_charge=None,
    onedfft_energycutoff=None,  # 1Dfft
    orbitalopt_sd_step=None,
    orbitalopt_criterion=None,
    ordern_hoppingranges=None,
    md_timestep=None,
    md_opt_criterion=None,
    nh_mass_heatbath=None,
    scf_nc_mag_field_spin=None,
    scf_nc_mag_field_orbital=None,
    system_currentdirectory=None,
    system_name=None,
    data_path=None,
    atoms_speciesandcoordinates_unit=None,
    atoms_unitvectors_unit=None,
    scf_xctype=None,
    scf_spinpolarization=None,
    scf_hubbard_occupation=None,
    scf_eigenvaluesolver=None,
    scf_mixing_type=None,
    orbitalopt_method=None,
    orbitalopt_startpulay=None,
    md_type=None,
    wannier_initial_projectors_unit=None,
    scf_partialcorecorrection=None,
    scf_hubbard_u=None,
    scf_constraint_nc_spin=None,
    scf_proexpn_vna=None,
    scf_spinorbit_coupling=None,
    cntorb_fileout=None,
    ordern_exact_inverse_s=None,
    ordern_recalc_buffer=None,
    ordern_expand_core=None,
    band_dispersion=None,
    scf_restart=None,
    mo_fileout=None,
    dos_fileout=None,
    hs_fileout=None,
    voronoi_charge=None,
    scf_nc_zeeman_spin=None,
    scf_stress_tensor=None,
    dos_erange=None,
    definition_of_atomic_species=None,
    atoms_speciesandcoordinates=None,
    atoms_unitvectors=None,
    hubbard_u_values=None,
    atoms_cont_orbitals=None,
    md_fixed_xyz=None,
    md_tempcontrol=None,
    md_init_velocity=None,
    band_kpath_unitcell=None,
    band_kpath=None,
    mo_kpoint=None,
    wannier_initial_projectors=None,
    xc='LDA',  # Begining of standard parameters
    maxiter=200,
    energy_cutoff=150 * Ry,
    kpts=(4, 4, 4),
    band_kpts=tuple(),  # To seperate monkhorst and band kpts
    eigensolver='Band',
    spinpol=None,
    convergence=1e-6 * Ha,
    external=None,
    mixer='Rmm-Diis',
    charge=None,
    smearing=None,
    restart=None,  # Begining of calculator parameters
    mpi=None,
    pbs=None,
    debug=False,
    nohup=True,
    dft_data_dict=None)


class OpenMXParameters(Parameters):
    """
    Parameters class for the OpenMX calculator. OpenMX parameters are defined
    here. If values seems unreasonable, for example, energy_cutoff=0.01, it
    gives warning. Changing standard parameters to openmx kewords is not a job
    for this class. We translate the variable right before we write. Hence,
    translation processes are written in `writers.py`. Here we only deals with
    default parameters and the reasonable boundary for that value.

    (1, 1, 1) < scf_kgrid < (16, 16, 16)
    1 < scf_maxiter < 10000
    1e-10 < scf_criterion < 1e-1
    100 < scf_energycutoff < 600
    100 * Ha < convergence < 600 * Ha

    """

    allowed_xc = [
            'LDA',
            'GGA', 'PBE', 'GGA-PBE',
            'LSDA',
            'LSDA-PW'
            'LSDA-CA'
            'CA',
            'PW',
        ]

    def __init__(self, **kwargs):
        kw = omx_parameter_defaults.copy()
        kw.update(kwargs)
        Parameters.__init__(self, **kw)

        if self.kpts == (1, 1, 1):
            print("When only the gamma point is considered, the eigenvalue \
                  solver is changed to 'Cluster' with the periodic boundary \
                  condition.")
            self.eigensolver = 'Cluster'
            self.mpi = None
            self.pbs = None

        from copy import deepcopy
        dft_data_dict = deepcopy(default_dictionary)
        if self.dft_data_dict is not None:
            dft_data_dict.update(self.dft_data_dict)
        self.dft_data_dict = dft_data_dict

        # keys = {k: v for k, v in kwargs.items() if not(v is None or v == [])}
