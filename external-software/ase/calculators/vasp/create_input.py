from __future__ import print_function
# Copyright (C) 2008 CSC - Scientific Computing Ltd.
"""This module defines an ASE interface to VASP.

Developed on the basis of modules by Jussi Enkovaara and John
Kitchin.  The path of the directory containing the pseudopotential
directories (potpaw,potpaw_GGA, potpaw_PBE, ...) should be set
by the environmental flag $VASP_PP_PATH.

The user should also set the environmental flag $VASP_SCRIPT pointing
to a python script looking something like::

   import os
   exitcode = os.system('vasp')

Alternatively, user can set the environmental flag $VASP_COMMAND pointing
to the command use the launch vasp e.g. 'vasp' or 'mpirun -n 16 vasp'

http://cms.mpi.univie.ac.at/vasp/
"""

import os
import sys
import warnings
import shutil
from os.path import join, isfile, islink

import numpy as np

from ase.calculators.calculator import kpts2ndarray
from ase.utils import basestring

from ase.calculators.vasp.setups import setups_defaults

# Parameters that can be set in INCAR. The values which are None
# are not written and default parameters of VASP are used for them.

float_keys = [
    'aexx',       # Fraction of exact/DFT exchange
    'aggac',      # Fraction of gradient correction to correlation
    'aggax',      # Fraction of gradient correction to exchange
    'aldac',      # Fraction of LDA correlation energy
    'amin',       #
    'amix',       #
    'amix_mag',   #
    'bmix',       # tags for mixing
    'bmix_mag',   #
    'cshift',     # Complex shift for dielectric tensor calculation (LOPTICS)
    'deper',      # relative stopping criterion for optimization of eigenvalue
    'ebreak',     # absolute stopping criterion for optimization of eigenvalues
                  # (EDIFF/N-BANDS/4)
    'efield',     # applied electrostatic field
    'emax',       # energy-range for DOSCAR file
    'emin',       #
    'enaug',      # Density cutoff
    'encut',      # Planewave cutoff
    'encutgw',    # energy cutoff for response function
    'encutfock',  # FFT grid in the HF related routines
    'hfscreen',   # attribute to change from PBE0 to HSE
    'kspacing',   # determines the number of k-points if the KPOINTS
                  # file is not present. KSPACING is the smallest
                  # allowed spacing between k-points in units of
                  # $\AA$^{-1}$.
    'potim',      # time-step for ion-motion (fs)
    'nelect',     # total number of electrons
    'param1',     # Exchange parameter
    'param2',     # Exchange parameter
    'pomass',     # mass of ions in am
    'pstress',    # add this stress to the stress tensor, and energy E = V *
                  # pstress
    'sigma',      # broadening in eV
    'smass',      # Nose mass-parameter (am)
    'spring',     # spring constant for NEB
    'time',       # special control tag
    'weimin',     # maximum weight for a band to be considered empty
    'zab_vdw',    # vdW-DF parameter
    'zval',       # ionic valence
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's
    # group at UT Austin
    'jacobian',   # Weight of lattice to atomic motion
    'ddr',        # (DdR) dimer separation
    'drotmax',    # (DRotMax) number of rotation steps per translation step
    'dfnmin',     # (DFNMin) rotational force below which dimer is not rotated
    'dfnmax',     # (DFNMax) rotational force below which dimer rotation stops
    'sltol',      # convergence ratio for minimum eigenvalue
    'sdr',        # finite difference for setting up Lanczos matrix and step
                  # size when translating
    'maxmove',    # Max step for translation for IOPT > 0
    'invcurv',   # Initial curvature for LBFGS (IOPT = 1)
    'timestep',   # Dynamical timestep for IOPT = 3 and IOPT = 7
    'sdalpha',    # Ratio between force and step size for IOPT = 4
    # The next keywords pertain to IOPT = 7 (i.e. FIRE)
    'ftimemax',   # Max time step
    'ftimedec',   # Factor to dec. dt
    'ftimeinc',   # Factor to inc. dt
    'falpha',     # Parameter for velocity damping
    'falphadec',  # Factor to dec. alpha
    'clz',        # electron count for core level shift
    'vdw_radius',  # Cutoff radius for Grimme's DFT-D2 and DFT-D3 and
                   # Tkatchenko and Scheffler's DFT-TS dispersion corrections
    'vdw_scaling',  # Global scaling parameter for Grimme's DFT-D2 dispersion
                    # correction
    'vdw_d',      # Global damping parameter for Grimme's DFT-D2 and Tkatchenko
                  # and Scheffler's DFT-TS dispersion corrections
    'vdw_cnradius',  # Cutoff radius for calculating coordination number in
                    # Grimme's DFT-D3 dispersion correction
    'vdw_s6',     # Damping parameter for Grimme's DFT-D2 and DFT-D3 and
                  # Tkatchenko and Scheffler's DFT-TS dispersion corrections
    'vdw_s8',     # Damping parameter for Grimme's DFT-D3 dispersion correction
    'vdw_sr',     # Scaling parameter for Grimme's DFT-D2 and DFT-D3 and
                  # Tkatchenko and Scheffler's DFT-TS dispersion correction
    'vdw_a1',     # Damping parameter for Grimme's DFT-D3 dispersion correction
    'vdw_a2',     # Damping parameter for Grimme's DFT-D3 dispersion correction
    'eb_k',       # solvent permitivity in Vaspsol
    'tau',        # surface tension parameter in Vaspsol
    'langevin_gamma_l',  # Friction for lattice degrees of freedom
    'pmass',      # Mass for latice degrees of freedom
    'bparam',     # B parameter for nonlocal VV10 vdW functional
    'cparam',     # C parameter for nonlocal VV10 vdW functional
    'aldax',      # Fraction of LDA exchange (for hybrid calculations)
    'tebeg',      #
    'teend',      # temperature during run
    'andersen_prob',  # Probability of collision in Andersen thermostat
    'apaco',      # Distance cutoff for pair correlation function calc.
    'auger_ecblo',  # Undocumented parameter for Auger calculations
    'auger_edens',  # Density of electrons in conduction band
    'auger_hdens',  # Density of holes in valence band
    'auger_efermi',  # Fixed Fermi level for Auger calculations
    'auger_evbhi',  # Upper bound for valence band maximum
    'auger_ewidth',  # Half-width of energy window function
    'auger_occ_fac_eeh',  # Undocumented parameter for Auger calculations
    'auger_occ_fac_ehh',  # Undocumented parameter for Auger calculations
    'auger_temp',  # Temperature for Auger calculation
    'dq',         # Finite difference displacement magnitude (NMR)
    'avgap',      # Average gap (Model GW)
    'bpotim',     # Undocumented Bond-Boost parameter (GH patches)
    'qrr',        # Undocumented Bond-Boost parameter (GH patches)
    'prr',        # Undocumented Bond-Boost parameter (GH patches)
    'rcut',       # Undocumented Bond-Boost parameter (GH patches)
    'dvmax',      # Undocumented Bond-Boost parameter (GH patches)
    'bfgsinvcurv',  # Initial curvature for BFGS (GH patches)
    'damping',    # Damping parameter for LBFGS (GH patches)
    'efirst',     # Energy of first NEB image (GH patches)
    'elast',      # Energy of final NEB image (GH patches)
    'fmagval',    # Force magnitude convergence criterion (GH patches)
    'cmbj',       # Undocumented MetaGGA parameter
    'cmbja',      # Undocumented MetaGGA parameter
    'cmbjb',      # Undocumented MetaGGA parameter
    'sigma_nc_k',  # Width of ion gaussians (VASPsol)
    'sigma_k',    # Width of dielectric cavidty (VASPsol)
    'nc_k',       # Cavity turn-on density (VASPsol)
    'lambda_d_k',  # Debye screening length (VASPsol)
    'ediffsol',   # Tolerance for solvation convergence (VASPsol)
    'deg_threshold',  # Degeneracy threshold
    'omegamin',   # Minimum frequency for dense freq. grid
    'omegamax',   # Maximum frequency for dense freq. grid
    'rtime',      # Undocumented parameter
    'wplasma',    # Undocumented parameter
    'wplasmai',   # Undocumented parameter
    'dfield',     # Undocumented parameter
    'omegatl',    # Maximum frequency for coarse freq. grid
    'encutgwsoft',  # Soft energy cutoff for response kernel
    'encutlf',    # Undocumented parameter
    'scissor',    # Scissor correction for GW/BSE calcs
    'dimer_dist',  # Distance between dimer images
    'step_size',  # Step size for finite difference in dimer calculation
    'step_max',   # Maximum step size for dimer calculation
    'minrot',     # Minimum rotation allowed in dimer calculation
    'dummy_mass',  # Mass of dummy atom(s?)
    'shaketol',   # Tolerance for SHAKE algorithm
    'shaketolsoft',  # Soft tolerance for SHAKE algorithm
    'shakesca',   # Scaling of each step taken in SHAKE algorithm
    'hills_stride',  # Undocumented metadynamics parameter
    'hills_h',    # Height (in eV) of gaussian bias for metadynamics
    'hills_w',    # Width of gaussian bias for metadynamics
    'hills_k',    # Force constant coupling dummy&real for metadynamics
    'hills_m',    # Mass of dummy particle for use in metadynamics
    'hills_temperature',  # Temp. of dummy particle for metadynamics
    'hills_andersen_prob',  # Probability of thermostat coll. for metadynamics
    'hills_sqq',  # Nose-hoover particle mass for metadynamics
    'dvvdelta0',  # Undocumented parameter
    'dvvvnorm0',  # Undocumented parameter
    'dvvminpotim',  # Undocumented parameter
    'dvvmaxpotim',  # Undocumented parameter
    'efermi',     # Undocumented parameter
    'enchg',      # Undocumented charge fitting parameter
    'tau0',       # Undocumented charge fitting parameter
    'encut4o',    # Cutoff energy for 4-center integrals (HF)
    'param3',     # Undocumented HF parameter
    'model_eps0',  # Undocumented HF parameter
    'model_alpha',  # Undocumented HF parameter
    'qmaxfockae',  # Undocumented HF parameter
    'hfscreenc',  # Range-separated screening length for correlations
    'hfrcut',     # Cutoff radius for HF potential kernel
    'encutae',    # Undocumented parameter for all-electron density calc.
    'encutsubrotscf',  # Undocumented subspace rotation SCF parameter
    'enini',      # Cutoff energy for wavefunctions (?)
    'wc',         # Undocumented mixing parameter
    'enmax',      # Cutoff energy for wavefunctions (?)
    'scalee',     # Undocumented parameter
    'eref',       # Reference energy
    'epsilon',    # Dielectric constant of bulk charged cells
    'rcmix',      # Mixing parameter for core density in rel. core calcs.
    'esemicore',  # Energetic lower bound for states considered "semicore"
    'external_pressure',  # Pressure for NPT calcs., equivalent to PSTRESS
    'lj_radius',  # Undocumented classical vdW parameter
    'lj_epsilon',  # Undocumented classical vdW parameter
    'lj_sigma',   # Undocumented classical vdW parameter
    'mbd_beta',   # TS MBD vdW correction damping parameter
    'scsrad',     # Cutoff radius for dipole-dipole interaction tensor in SCS
    'hitoler',    # Iterative Hirschfeld partitioning tolerance
    'lambda',     # "Spring constant" for magmom constraint calcs.
    'kproj_threshold',  # Threshold for k-point projection scheme
    'maxpwamp',   # Undocumented HF parameter
    'vcutoff',    # Undocumented parameter
    'mdtemp',     # Temperature for AIMD
    'mdgamma',    # Undocumented AIMD parameter
    'mdalpha',    # Undocumented AIMD parameter
    'ofield_kappa',  # Bias potential strength for interface pinning method
    'ofield_q6_near',  # Steinhardt-Nelson Q6 parameters for interface pinning
    'ofield_q6_far',  # Steinhardt-Nelson Q6 parameters for interface pinning
    'ofield_a',   # Target order parameter for interface pinning method
    'pthreshold',  # Don't print timings for routines faster than this value
    'qltol',      # Eigenvalue tolerance for Lanczos iteration (instanton)
    'qdr',        # Step size for building Lanczos matrix & CG (instanton)
    'qmaxmove',   # Max step size (instanton)
    'qdt',        # Timestep for quickmin minimization (instanton)
    'qtpz',       # Temperature (instanton)
    'qftol',      # Tolerance (instanton)
]

exp_keys = [
    'ediff',      # stopping-criterion for electronic upd.
    'ediffg',     # stopping-criterion for ionic upd.
    'symprec',    # precession in symmetry routines
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's
    # group at UT Austin
    'fdstep',     # Finite diference step for IOPT = 1 or 2
]

string_keys = [
    'algo',       # algorithm: Normal (Davidson) | Fast | Very_Fast (RMM-DIIS)
    'gga',        # xc-type: PW PB LM or 91 (LDA if not set)
    'metagga',    #
    'prec',       # Precission of calculation (Low, Normal, Accurate)
    'system',     # name of System
    'precfock',    # FFT grid in the HF related routines
    'radeq',      # Which type of radial equations to use for rel. core calcs.
    'localized_basis',  # Basis to use in CRPA
    'proutine',   # Select profiling routine
]

int_keys = [
    'ialgo',      # algorithm: use only 8 (CG) or 48 (RMM-DIIS)
    'ibrion',     # ionic relaxation: 0-MD 1-quasi-New 2-CG
    'icharg',     # charge: 0-WAVECAR 1-CHGCAR 2-atom 10-const
    'idipol',     # monopol/dipol and quadropole corrections
    'images',     # number of images for NEB calculation
    'iniwav',     # initial electr wf. : 0-lowe 1-rand
    'isif',       # calculate stress and what to relax
    'ismear',     # part. occupancies: -5 Blochl -4-tet -1-fermi 0-gaus >0 MP
    'ispin',      # spin-polarized calculation
    'istart',     # startjob: 0-new 1-cont 2-samecut
    'isym',       # symmetry: 0-nonsym 1-usesym 2-usePAWsym
    'iwavpr',     # prediction of wf.: 0-non 1-charg 2-wave 3-comb
    'kpar',       # k-point parallelization paramater
    'ldauprint',  # 0-silent, 1-occ. matrix written to OUTCAR, 2-1+pot. matrix
                  # written
    'ldautype',   # L(S)DA+U: 1-Liechtenstein 2-Dudarev 4-Liechtenstein(LDAU)
    'lmaxmix',    #
    'lorbit',     # create PROOUT
    'maxmix',     #
    'ngx',        # FFT mesh for wavefunctions, x
    'ngxf',       # FFT mesh for charges x
    'ngy',        # FFT mesh for wavefunctions, y
    'ngyf',       # FFT mesh for charges y
    'ngz',        # FFT mesh for wavefunctions, z
    'ngzf',       # FFT mesh for charges z
    'nbands',     # Number of bands
    'nblk',       # blocking for some BLAS calls (Sec. 6.5)
    'nbmod',      # specifies mode for partial charge calculation
    'nelm',       # nr. of electronic steps (default 60)
    'nelmdl',     # nr. of initial electronic steps
    'nelmin',
    'nfree',      # number of steps per DOF when calculting Hessian using
                  # finite differences
    'nkred',      # define sub grid of q-points for HF with
                  # nkredx=nkredy=nkredz
    'nkredx',      # define sub grid of q-points in x direction for HF
    'nkredy',      # define sub grid of q-points in y direction for HF
    'nkredz',      # define sub grid of q-points in z direction for HF
    'nomega',     # number of frequency points
    'nomegar',    # number of frequency points on real axis
    'npar',       # parallelization over bands
    'nsim',       # evaluate NSIM bands simultaneously if using RMM-DIIS
    'nsw',        # number of steps for ionic upd.
    'nupdown',    # fix spin moment to specified value
    'nwrite',     # verbosity write-flag (how much is written)
    'vdwgr',      # extra keyword for Andris program
    'vdwrn',      # extra keyword for Andris program
    'voskown',    # use Vosko, Wilk, Nusair interpolation
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's
    # group at UT Austin
    'ichain',     # Flag for controlling which method is being used (0=NEB,
                  # 1=DynMat, 2=Dimer, 3=Lanczos) if ichain > 3, then both
                  # IBRION and POTIM are automatically set in the INCAR file
    'iopt',       # Controls which optimizer to use.  for iopt > 0, ibrion = 3
                  # and potim = 0.0
    'snl',        # Maximum dimentionality of the Lanczos matrix
    'lbfgsmem',   # Steps saved for inverse Hessian for IOPT = 1 (LBFGS)
    'fnmin',      # Max iter. before adjusting dt and alpha for IOPT = 7 (FIRE)
    'icorelevel',  # core level shifts
    'clnt',       # species index
    'cln',        # main quantum number of excited core electron
    'cll',        # l quantum number of excited core electron
    'ivdw',       # Choose which dispersion correction method to use
    'nbandsgw',   # Number of bands for GW
    'nbandso',    # Number of occupied bands for electron-hole treatment
    'nbandsv',    # Number of virtual bands for electron-hole treatment
    'ncore',      # Number of cores per band, equal to number of cores divided
                  # by npar
    'mdalgo',     # Determines which MD method of Tomas Bucko to use
    'nedos',      # Number of grid points in DOS
    'turbo',      # Ewald, 0 = Normal, 1 = PME
    'omegapar',   # Number of groups for response function calc.
    'taupar',     # Number of groups in real time for response function calc.
    'antires',    # How to treat antiresonant part of response function
    'magatom',    # Index of atom at which to place magnetic field (NMR)
    'jatom',      # Index of atom at which magnetic moment is evaluated (NMR)
    'ichibare',   # chi_bare stencil size (NMR)
    'nbas',       # Undocumented Bond-Boost parameter (GH patches)
    'rmds',       # Undocumented Bond-Boost parameter (GH patches)
    'ilbfgsmem',  # Number of histories to store for LBFGS (GH patches)
    'vcaimages',  # Undocumented parameter (GH patches)
    'ntemper',    # Undocumented subspace diagonalization param. (GH patches)
    'ncshmem',    # Share memory between this many cores on each process
    'lmaxtau',    # Undocumented MetaGGA parameter (prob. max ang.mom. for tau)
    'kinter',     # Additional finer grid (?)
    'ibse',       # Type of BSE calculation
    'nbseeig',    # Number of BSE wfns to write
    'naturalo',   # Use NATURALO (?)
    'nbandsexact',  # Undocumented parameter
    'nbandsgwlow',  # Number of bands for which shifts are calculated
    'nbandslf',   # Number of bands included in local field effect calc.
    'omegagrid',  # Undocumented parameter
    'telescope',  # Undocumented parameter
    'maxmem',     # Amount of memory to allocate per core in MB
    'nelmhf',     # Number of iterations for HF part (GW)
    'dim',        # Undocumented parameter
    'nkredlf',    # Reduce k-points for local field effects
    'nkredlfx',   # Reduce k-points for local field effects in X
    'nkredlfy',   # Reduce k-points for local field effects in Y
    'nkredlfz',   # Reduce k-points for local field effects in Z
    'lmaxmp2',    # Undocumented parameter
    'switch',     # Undocumented dimer parameter
    'findiff',    # Use forward (1) or central (2) finite difference for dimer
    'engine',     # Undocumented dimer parameter
    'restartcg',  # Undocumented dimer parameter
    'thermostat',  # Deprecated parameter for selecting MD method (use MDALGO)
    'scaling',    # After how many steps velocities should be rescaled
    'shakemaxiter',  # Maximum # of iterations in SHAKE algorithm
    'equi_regime',  # Number of steps to equilibrate for
    'hills_bin',  # Update metadynamics bias after this many steps
    'hills_maxstride',  # Undocumented metadynamics parameter
    'dvvehistory',  # Undocumented parameter
    'ipead',      # Undocumented parameter
    'ngaus',      # Undocumented charge fitting parameter
    'exxoep',     # Undocumented HF parameter
    'fourorbit',  # Undocumented HF parameter
    'model_gw',   # Undocumented HF parameter
    'hflmax',     # Maximum L quantum number for HF calculation
    'lmaxfock',   # Maximum L quantum number for HF calc. (same as above)
    'lmaxfockae',  # Undocumented HF parameter
    'nmaxfockae',  # Undocumented HF parameter
    'nblock_fock',  # Undocumented HF parameter
    'idiot',      # Determines which warnings/errors to print
    'nrmm',       # Number of RMM-DIIS iterations
    'mremove',    # Undocumented mixing parameter
    'inimix',     # Undocumented mixing parameter
    'mixpre',     # Undocumented mixing parameter
    'nelmall',    # Undocumented parameter
    'nblock',     # How frequently to write data
    'kblock',     # How frequently to write data
    'npaco',      # Undocumented pair correlation function parameter
    'lmaxpaw',    # Max L quantum number for on-site charge expansion
    'irestart',   # Undocumented parameter
    'nreboot',    # Undocumented parameter
    'nmin',       # Undocumented parameter
    'nlspline',   # Undocumented parameter
    'ispecial',   # "Select undocumented and unsupported special features"
    'rcrep',      # Number of steps between printing relaxed core info
    'rcndl',      # Wait this many steps before updating core density
    'rcstrd',     # Relax core density after this many SCF steps
    'vdw_idampf',  # Select type of damping function for TS vdW
    'i_constrained_m',  # Select type of magmom. constraint to use
    'igpar',      # "G parallel" direction for Berry phase calculation
    'nppstr',     # Number of kpts in "igpar' direction for Berry phase calc.
    'nbands_out',  # Undocumented QP parameter
    'kpts_out',   # Undocumented QP parameter
    'isp_out',    # Undocumented QP parameter
    'nomega_out',  # Undocumented QP parameter
    'maxiter_ft',  # Max iterations for sloppy Remez algorithm
    'nmaxalt',    # Max sample points for alternant in Remez algorithms
    'itmaxlsq',   # Max iterations in LSQ search algorithm
    'ndatalsq',   # Number of sample points for LSQ search algorithm
    'ncore_in_image1',  # Undocumented parameter
    'kimages',    # Undocumented parameter
    'ncores_per_band',  # Undocumented parameter
    'maxlie',     # Max iterations in CRPA diagonalization routine
    'ncrpalow',   # Undocumented CRPA parameter
    'ncrpahigh',  # Undocumented CRPA parameter
    'nwlow',      # Undocumented parameter
    'nwhigh',     # Undocumented parameter
    'nkopt',      # Number of k-points to include in Optics calculation
    'nkoffopt',   # K-point "counter offset" for Optics
    'nbvalopt',   # Number of valence bands to write in OPTICS file
    'nbconopt',   # Number of conduction bands to write in OPTICS file
    'plevel',     # No timings for routines with "level" higher than this
    'qnl',        # Lanczos matrix size (instanton)
]

bool_keys = [
    'addgrid',    # finer grid for augmentation charge density
    'kgamma',     # The generated kpoint grid (from KSPACING) is either
                  # centred at the $\Gamma$
                  # point (e.g. includes the $\Gamma$ point)
                  # (KGAMMA=.TRUE.)
    'laechg',     # write AECCAR0/AECCAR1/AECCAR2
    'lasph',      # non-spherical contributions to XC energy (and pot for
                  # VASP.5.X)
    'lasync',     # overlap communcation with calculations
    'lcharg',     #
    'lcorr',      # Harris-correction to forces
    'ldau',       # L(S)DA+U
    'ldiag',      # algorithm: perform sub space rotation
    'ldipol',     # potential correction mode
    'lelf',       # create ELFCAR
    'lepsilon',   # enables to calculate and to print the BEC tensors
    'lhfcalc',    # switch to turn on Hartree Fock calculations
    'loptics',    # calculate the frequency dependent dielectric matrix
    'lpard',      # evaluate partial (band and/or k-point) decomposed charge
                  # density
    'lplane',     # parallelisation over the FFT grid
    'lscalapack',  # switch off scaLAPACK
    'lscalu',     # switch of LU decomposition
    'lsepb',      # write out partial charge of each band separately?
    'lsepk',      # write out partial charge of each k-point separately?
    'lthomas',    #
    'luse_vdw',   # Invoke vdW-DF implementation by Klimes et. al
    'lvdw',   # Invoke DFT-D2 method of Grimme
    'lvhar',      # write Hartree potential to LOCPOT (vasp 5.x)
    'lvtot',      # create WAVECAR/CHGCAR/LOCPOT
    'lwave',      #
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's
    # group at UT Austin
    'lclimb',     # Turn on CI-NEB
    'ltangentold',  # Old central difference tangent
    'ldneb',      # Turn on modified double nudging
    'lnebcell',   # Turn on SS-NEB
    'lglobal',    # Optmize NEB globally for LBFGS (IOPT = 1)
    'llineopt',   # Use force based line minimizer for translation (IOPT = 1)
    'lbeefens',   # Switch on print of BEE energy contributions in OUTCAR
    'lbeefbas',   # Switch off print of all BEEs in OUTCAR
    'lcalcpol',   # macroscopic polarization (vasp5.2). 'lcalceps'
    'lcalceps',   # Macroscopic dielectric properties and Born effective charge
                  # tensors (vasp 5.2)

    'lvdw',       # Turns on dispersion correction
    'lvdw_ewald',  # Turns on Ewald summation for Grimme's DFT-D2 and
                   # Tkatchenko and Scheffler's DFT-TS dispersion correction
    'lspectral',  # Use the spectral method to calculate independent particle
                  # polarizability
    'lrpa',       # Include local field effects on the Hartree level only
    'lwannier90',  # Switches on the interface between VASP and WANNIER90
    'lsorbit',    # Enable spin-orbit coupling
    'lsol',       # turn on solvation for Vaspsol
    'lautoscale',  # automatically calculate inverse curvature for VTST LBFGS
    'interactive',  # Enables interactive calculation for VaspInteractive
    'lauger',      # Perform Auger calculation (Auger)
    'lauger_eeh',  # Calculate EEH processes (Auger)
    'lauger_ehh',  # Calculate EHH processes (Auger)
    'lauger_collect',  # Collect wfns before looping over k-points (Auger)
    'lauger_dhdk',  # Auto-determine E. window width from E. derivs. (Auger)
    'lauger_jit',  # Distribute wavefunctions for k1-k4 (Auger)
    'orbitalmag',  # Enable orbital magnetization (NMR)
    'lchimag',    # Use linear response for shielding tensor (NMR)
    'lwrtcur',    # Write response of current to mag. field to file (NMR)
    'lnmr_sym_red',  # Reduce symmetry for finite difference (NMR)
    'lzora',      # Use ZORA approximation in linear-response NMR (NMR)
    'lbone',      # Use B-component in AE one-center terms for LR NMR (NMR)
    'lmagbloch',  # Use Bloch summations to obtain orbital magnetization (NMR)
    'lgauge',     # Use gauge transformation for zero moment terms (NMR)
    'lbfconst',   # Use constant B-field with sawtooth vector potential (NMR)
    'nucind',     # Use nuclear independent calculation (NMR)
    'lnicsall',   # Use all grid points for 'nucind' calculation (NMR)
    'llraug',     # Use two-center corrections for induced B-field (NMR)
    'lbbm',       # Undocumented Bond-Boost parameter (GH patches)
    'lnoncollinear',  # Do non-collinear spin polarized calculation
    'bfgsdfp',    # Undocumented BFGS parameter (GH patches)
    'linemin',    # Use line minimization (GH patches)
    'ldneborg',   # Undocumented NEB parameter (GH patches)
    'dseed',      # Undocumented dimer parameter (GH patches)
    'linteract',  # Undocumented parameter (GH patches)
    'lmpmd',      # Undocumented parameter (GH patches)
    'ltwodim',    # Makes stress tensor two-dimensional (GH patches)
    'fmagflag',   # Use force magnitude as convergence criterion (GH patches)
    'ltemper',    # Use subspace diagonalization (?) (GH patches)
    'qmflag',     # Undocumented FIRE parameter (GH patches)
    'lmixtau',    # Undocumented MetaGGA parameter
    'ljdftx',     # Undocumented VASPsol parameter (VASPsol)
    'lrhob',      # Write the bound charge density (VASPsol)
    'lrhoion',    # Write the ionic charge density (VASPsol)
    'lnabla',     # Undocumented parameter
    'linterfast',  # Interpolate in K using linear response routines
    'lvel',       # Undocumented parameter
    'lrpaforce',  # Calculate RPA forces
    'lhartree',   # Use IP approx. in BSE (testing only)
    'ladder',     # Use ladder diagrams
    'lfxc',       # Use approximate ladder diagrams
    'lrsrpa',     # Undocumented parameter
    'lsingles',   # Calculate HF singles
    'lfermigw',   # Iterate Fermi level
    'ltcte',      # Undocumented parameter
    'ltete',      # Undocumented parameter
    'ltriplet',   # Undocumented parameter
    'lfxceps',    # Undocumented parameter
    'lfxheg',     # Undocumented parameter
    'l2order',    # Undocumented parameter
    'lmp2lt',     # Undocumented parameter
    'lgwlf',      # Undocumented parameter
    'lusew',      # Undocumented parameter
    'selfenergy',  # Undocumented parameter
    'oddonlygw',  # Avoid gamma point in response function calc.
    'evenonlygw',  # Avoid even points in response function calc.
    'lspectralgw',  # More accurate self-energy calculation
    'fletcher_reeves',  # Undocumented dimer parameter
    'lidm_selective',  # Undocumented dimer parameter
    'lblueout',   # Write output of blue-moon algorithm
    'hills_variable_w',  # Enable variable-width metadynamics bias
    'dvvminus',   # Undocumented parameter
    'lpead',      # Calculate cell-periodic orbital derivs. using finite diff.
    'skip_edotp',  # Skip updating elec. polarization during scf
    'skip_scf',   # Skip calculation w/ local field effects
    'lchgfit',    # Turn on charge fitting
    'lgausrc',    # Undocumented charge fitting parameter
    'lstockholder',  # Enable ISA charge fitting (?)
    'lsymgrad',   # Restore symmetry of gradient (HF)
    'lhfone',     # Calculate one-center terms (HF)
    'lrscor',     # Include long-range correlation (HF)
    'lrhfcalc',   # Include long-range HF (HF)
    'lmodelhf',   # Model HF calculation (HF)
    'shiftred',   # Undocumented HF paramter
    'hfkident',   # Undocumented HF parameter
    'oddonly',    # Undocumented HF parameter
    'evenonly',   # Undocumented HF parameter
    'lfockaedft',  # Undocumented HF parameter
    'lsubsrot',   # Enable subspace rotation diagonalization
    'mixfirst',   # Mix before diagonalization
    'lvcader',    # Calculate derivs. w.r.t. VCA parameters
    'lcompat',    # Enable "full compatibility"
    'lmusic',     # "Joke" parameter
    'ldownsample',  # Downsample WAVECAR to fewer k-points
    'lscaaware',  # Disable ScaLAPACK for some things but not all
    'lorbitalreal',  # Undocumented parameter
    'lmetagga',   # Undocumented parameter
    'lspiral',    # Undocumented parameter
    'lzeroz',     # Undocumented parameter
    'lmono',      # Enable "monopole" corrections
    'lrelcore',   # Perform relaxed core calculation
    'lmimicfc',   # Mimic frozen-core calcs. for relaxed core calcs.
    'lmatchrw',   # Match PS partial waves at RWIGS? (otherwise PAW cutoff)
    'ladaptelin',  # Linearize core state energies to avoid divergences
    'lonlysemicore',  # Only linearize semi-core state energies
    'gga_compat',  # Enable backwards-compatible symmetrization of GGA derivs.
    'lrelvol',    # Undocumented classical vdW parameter
    'lj_only',    # Undocumented classical vdW parameter
    'lvdwscs',    # Include self-consistent screening in TS vdW correction
    'lcfdm',      # Use coupled fluctuating dipoles model for TS vdW
    'lvdw_sametype',  # Include interactions between atoms of the same type
    'lrescaler0',  # Rescale damping parameters in SCS vdW correction
    'lscsgrad',   # Calculate gradients for TS+SCS vdW correction energies
    'lvdwexpansion',  # Write 2-6 body contribs. to MBD vdW correction energy
    'lvdw_relvolone',  # Undocumented classical vdW parameter
    'lberry',     # Enable Berry-phase calculation
    'lpade_fit',  # Undocumented QP parameter
    'lkproj',     # Enable projection onto k-points
    'l_wr_moments',  # Undocumented parameter
    'l_wr_density',  # Undocumented parameter
    'lkotani',    # Undocumented parameter
    'ldyson',     # Undocumented parameter
    'laddherm',   # Undocumented parameter
    'lcrpaplot',  # Plot bands used in CRPA response func. calc.
    'lplotdis',   # Plot disentangled bands in CRPA response func. calc.
    'ldisentangle',  # Disentangle bands in CRPA
    'lweighted',  # "Weighted" CRPA approach
    'luseorth_lcaos',  # Use orthogonalized LCAOs in CRPA
    'lfrpa',      # Use full RPA in CRPA
    'lregularize',  # Regularize projectors in CRPA
    'ldrude',     # Include Drude term in CRPA
    'ldmatrix',   # Undocumented parameter
    'lefg',       # Calculate electric field gradient at atomic nuclei
    'lhyperfine',  # Enable Hyperfine calculation
    'lwannier',   # Enable Wannier interface
    'localize',   # Undocumented Wannier parameter
    'lintpol_wpot',  # Interpolate WPOT for Wannier
    'lintpol_orb',  # Interpolate orbitals for Wannier
    'lintpol_kpath',  # Interpolate bandstructure on given kpath for Wannier
    'lintpol_kpath_orb',  # Interpolate orbitals on given kpath for Wannier
    'lread_eigenvalues',  # Use Eigenvalues from EIGENVALUES.INT file
    'lintpol_velocity',  # Interpolate electron velocity for Wannier
    'lintpol_conductivity',  # Interpolate conductivity for Wannier
    'lwannierinterpol',  # Undocumented Wannier parameter
    'wanproj',    # Undocumented Wannier parameter
    'lorbmom',    # Undocumented LDA+U parameter
    'lwannier90_run',  # Undocumented WANNIER90 parameter
    'lwrite_wanproj',  # Write UWAN files for WANNIER90
    'lwrite_unk',  # Write UNK files for WANNIER90
    'lwrite_mmn_amn',  # Write MMN and AMN files for WANNIER90
    'lread_amn',  # Read AMN files instead of recomputing (WANNIER90)
    'lrhfatm',    # Undocumented HF parameter
    'lvpot',      # Calculate unscreened potential
    'lwpot',      # Calculate screened potential
    'lwswq',      # Undocumented parameter
    'pflat',      # Only print "flat" timings to OUTCAR
    'qifcg',      # Use CG instead of quickmin (instanton)
    'qdo_ins',    # Find instanton
    'qdo_pre',    # Calculate prefactor (instanton)
]

list_int_keys = [
    'iband',      # bands to calculate partial charge for
    'kpuse',      # k-point to calculate partial charge for
    'ldaul',      # DFT+U parameters, overruled by dict key 'ldau_luj'
    'random_seed',  # List of ints used to seed RNG for advanced MD routines
                    # (Bucko)
    'auger_bmin_eeh',  # 4 ints | Various undocumented parameters for Auger
    'auger_bmax_eeh',  # 4 ints | calculations
    'auger_bmin_ehh',  # 4 ints |
    'auger_bmax_ehh',  # 4 ints |
    'balist',     # nbas ints | Undocumented Bond-Boost parameter (GH patches)
    'kpoint_bse',  # 4 ints | Undocumented parameter
    'nsubsys',    # <=3 ints | Last atom # for each of up to 3 thermostats
    'vdw_refstate',  # ntyp ints | Undocumented classical vdW parameter
    'vdw_mbd_size',  # 3 ints | Supercell size for TS MBD vdW correction
    'nbands_index',  # nbands_out ints | Undocumented QP parameter
    'kpts_index',  # kpts_out ints | Undocumented QP parameter
    'isp_index',  # isp_out ints | Undocumented QP parameter
    'nomega_index',  # nomega_out ints | Undocumented QP parameter
    'ntarget_states',  # nbands ints | Undocumented CRPA parameter
    'wanproj_i',  # nions ints | Undocumented Wannier parameter
    'wanproj_l',  # ? ints | Undocumented Wannier parameter
    ]

list_bool_keys = [
    'lattice_constraints',  # 3 bools | Undocumented advanced MD parameter
    'lrctype',    # ntyp bools | Enable relaxed-core calc. for these atoms
    'lvdw_onecell',  # 3 bools | Enable periodicity in A, B, C vector for vdW
    ]

list_float_keys = [
    'dipol',      # center of cell for dipol
    'eint',       # energy range to calculate partial charge for
    'ferwe',      # Fixed band occupation (spin-paired)
    'ferdo',      # Fixed band occupation (spin-plarized)
    'magmom',     # initial magnetic moments
    'ropt',       # number of grid points for non-local proj in real space
    'rwigs',      # Wigner-Seitz radii
    'ldauu',      # ldau parameters, has potential to redundant w.r.t. dict
    'ldauj',      # key 'ldau_luj', but 'ldau_luj' can't be read direct from
                  # the INCAR (since it needs to know information about atomic
                  # species. In case of conflict 'ldau_luj' gets written out
                  # when a calculation is set up
    'vdw_c6',     # List of floats of C6 parameters (J nm^6 mol^-1) for each
                  # species (DFT-D2 and DFT-TS)
    'vdw_c6au',   # List of floats of C6 parameters (a.u.) for each species
                  # (DFT-TS)
    'vdw_r0',     # List of floats of R0 parameters (angstroms) for each
                  # species (DFT-D2 and DFT-TS)
    'vdw_r0au',   # List of floats of R0 parameters (a.u.) for each species
                  # (DFT-TS)
    'vdw_alpha',  # List of floats of free-atomic polarizabilities for each
                  # species (DFT-TS)
    'langevin_gamma',  # List of floats for langevin friction coefficients
    'auger_emin_eeh',  # 4 floats | Various undocumented parameters for Auger
    'auger_emax_eeh',  # 4 floats | calculations
    'auger_emin_ehh',  # 4 floats |
    'auger_emax_ehh',  # 4 floats |
    'avecconst',  # 3 floats | magnitude of magnetic moment (NMR)
    'magdipol',   # 3 floats | magnitude of magnetic dipole (NMR)
    'bconst',     # 3 floats | magnitude of constant magnetic field (NMR)
    'magpos',     # 3 floats | position for magnetic moment w/ 'nucind' (NMR)
    'bext',       # 3 floats | Undocumented (probably external magnetic field)
    'core_c',     # ntyp floats | pseudo-core charge magnitude (VASPsol)
    'sigma_rc_k',  # ntyp floats | width of pseudo-core gaussians (VASPsol)
    'darwinr',    # ntypd (?) floats | Undocumented parameter
    'darwinv',    # ntypd (?) floats | Undocumented parameter
    'dummy_k',    # ? floats | Force const. connecting dummy atoms to sys.
    'dummy_r0',   # ? floats | Minimum dist., ang., etc. for dummy atom DOFs
    'dummy_positions',  # 3 floats | Position of dummy atom(s?)
    'psubsys',    # <=3 floats | Coll. prob. for each of up to 3 thermostats
    'tsubsys',    # <=3 floats | Temp. for each of up to 3 thermostats
    'increm',     # ? floats | Undocumented advanced MD parameter
    'value_min',  # ? floats | Undocumented advanced MD parameter
    'value_max',  # ? floats | Undocumented advanced MD parameter
    'hills_position',  # ? floats | Dummy particle(s) pos. for metadynamics
    'hills_velocity',  # ? floats | Dummy particle(s) vel. for metadynamics
    'spring_k',   # ? floats | Spring constant for harmonic constraints
    'spring_r0',  # ? floats | Spring minima for harmonic constraints
    'spring_v0',  # ? floats | Initial velocity of harmonic constraints
    'hills_wall_lower',  # ? floats | Undocumented metadynamics parameter
    'hills_wall_upper',  # ? floats | Undocumented metadynamics parameter
    'efield_pead',  # 3 floats | homogeneous electric field for PEAD calc.
    'zct',        # ? floats | Undocumented charge fitting parameter
    'rgaus',      # ? floats | Undocumented charge fitting parameter
    'hfalpha',    # 10 floats | Undocumented HF parameter
    'mcalpha',    # 10 floats | Undocumented HF parameter
    'saxis',      # 3 floats | Coordinate for collinear spin calculations
    'vca',        # ? floats | Atom weight for VCA calculations
    'stm',        # 7 floats | "range for STM data"
    'qspiral',    # 3 floats | Undocumented parameter
    'external_stress',  # 6 floats | Target stress (adds w/ external_pressure)
    'm_constr',   # 3*nions floats | Local magmom assigned to each spin DOF
    'quad_efg',   # ntyp floats | Nuclear quadrupole moments
    'ngyromag',   # ntyp floats | Nuclear gyromagnetic ratios
    'rcrhocut',   # ntyp floats | Core density cutoff rad. for HF relcore calc
    'ofield_k',   # 3 floats | Undocumented parameter
    'paripot',    # ? floats | Undocumented parameter
    'smearings',  # ? floats | ismear,sigma smearing params to loop over
    'wanproj_e',  # 2 floats | Undocumented Wannier parameter
]

special_keys = [
    'lreal',      # non-local projectors in real space
]

dict_keys = [
    'ldau_luj',   # dictionary with L(S)DA+U parameters, e.g. {'Fe':{'L':2,
                  # 'U':4.0, 'J':0.9}, ...}
]

keys = [
    # 'NBLOCK' and KBLOCK       inner block; outer block
    # 'NPACO' and APACO         distance and nr. of slots for P.C.
    # 'WEIMIN, EBREAK, DEPER    special control tags
]


class GenerateVaspInput(object):
    # Parameters corresponding to 'xc' settings.  This may be modified
    # by the user in-between loading calculators.vasp submodule and
    # instantiating the calculator object with calculators.vasp.Vasp()
    xc_defaults = {
        'lda': {'pp': 'LDA'},
        # GGAs
        'pw91': {'pp': 'GGA', 'gga': '91'},
        'pbe': {'pp': 'PBE', 'gga': 'PE'},
        'pbesol': {'gga': 'PS'},
        'revpbe': {'gga': 'RE'},
        'rpbe': {'gga': 'RP'},
        'am05': {'gga': 'AM'},
        # Meta-GGAs
        'tpss': {'metagga': 'TPSS'},
        'revtpss': {'metagga': 'RTPSS'},
        'm06l': {'metagga': 'M06L'},
        'ms0': {'metagga': 'MS0'},
        'ms1': {'metagga': 'MS1'},
        'ms2': {'metagga': 'MS2'},
        'scan': {'metagga': 'SCAN'},
        'scan-rvv10': {'metagga': 'SCAN', 'luse_vdw': True, 'bparam': 15.7},
        # vdW-DFs
        'vdw-df': {'gga': 'RE', 'luse_vdw': True, 'aggac': 0.},
        'optpbe-vdw': {'gga': 'OR', 'luse_vdw': True, 'aggac': 0.0},
        'optb88-vdw': {'gga': 'BO', 'luse_vdw': True, 'aggac': 0.0,
                       'param1': 1.1 / 6.0, 'param2': 0.22},
        'optb86b-vdw': {'gga': 'MK', 'luse_vdw': True, 'aggac': 0.0,
                        'param1': 0.1234, 'param2': 1.0},
        'vdw-df2': {'gga': 'ML', 'luse_vdw': True, 'aggac': 0.0,
                    'zab_vdw': -1.8867},
        'beef-vdw': {'gga': 'BF', 'luse_vdw': True,
                     'zab_vdw': -1.8867},
        # Hartree-Fock and hybrids
        'hf': {'lhfcalc': True, 'aexx': 1.0, 'aldac': 0.0,
               'aggac': 0.0},
        'b3lyp': {'gga': 'B3', 'lhfcalc': True, 'aexx': 0.2,
                  'aggax': 0.72, 'aggac': 0.81, 'aldac': 0.19},
        'pbe0': {'gga': 'PE', 'lhfcalc': True},
        'hse03': {'gga': 'PE', 'lhfcalc': True, 'hfscreen': 0.3},
        'hse06': {'gga': 'PE', 'lhfcalc': True, 'hfscreen': 0.2},
        'hsesol': {'gga': 'PS', 'lhfcalc': True, 'hfscreen': 0.2}}

    def __init__(self, restart=None):
        self.float_params = {}
        self.exp_params = {}
        self.string_params = {}
        self.int_params = {}
        self.bool_params = {}
        self.list_bool_params = {}
        self.list_int_params = {}
        self.list_float_params = {}
        self.special_params = {}
        self.dict_params = {}
        for key in float_keys:
            self.float_params[key] = None
        for key in exp_keys:
            self.exp_params[key] = None
        for key in string_keys:
            self.string_params[key] = None
        for key in int_keys:
            self.int_params[key] = None
        for key in bool_keys:
            self.bool_params[key] = None
        for key in list_bool_keys:
            self.list_bool_params[key] = None
        for key in list_int_keys:
            self.list_int_params[key] = None
        for key in list_float_keys:
            self.list_float_params[key] = None
        for key in special_keys:
            self.special_params[key] = None
        for key in dict_keys:
            self.dict_params[key] = None

        # Initialize internal dictionary of input parameters which are
        # not regular VASP keys
        self.input_params = {
            'xc': None,  # Exchange-correlation recipe (e.g. 'B3LYP')
            'pp': None,  # Pseudopotential file (e.g. 'PW91')
            'setups': None,  # Special setups (e.g pv, sv, ...)
            'txt': '-',  # Where to send information
            'kpts': (1, 1, 1),  # k-points
            # Option to use gamma-sampling instead of Monkhorst-Pack:
            'gamma': False,
            # number of points between points in band structures:
            'kpts_nintersections': None,
            # Option to write explicit k-points in units
            # of reciprocal lattice vectors:
            'reciprocal': False,
            # Switch to disable writing constraints to POSCAR
            'ignore_constraints': False
        }

    def set_xc_params(self, xc):
        """Set parameters corresponding to XC functional"""
        xc = xc.lower()
        if xc is None:
            pass
        elif xc not in self.xc_defaults:
            xc_allowed = ', '.join(self.xc_defaults.keys())
            raise ValueError(
                '{0} is not supported for xc! Supported xc values'
                'are: {1}'.format(xc, xc_allowed))
        else:
            # XC defaults to PBE pseudopotentials
            if 'pp' not in self.xc_defaults[xc]:
                self.set(pp='PBE')
            self.set(**self.xc_defaults[xc])

    def set(self, **kwargs):

        if ((('ldauu' in kwargs) and
             ('ldaul' in kwargs) and
             ('ldauj' in kwargs) and
             ('ldau_luj' in kwargs))):
            raise NotImplementedError(
                'You can either specify ldaul, ldauu, and ldauj OR '
                'ldau_luj. ldau_luj is not a VASP keyword. It is a '
                'dictionary that specifies L, U and J for each '
                'chemical species in the atoms object. '
                'For example for a water molecule:'
                '''ldau_luj={'H':{'L':2, 'U':4.0, 'J':0.9},
                      'O':{'L':2, 'U':4.0, 'J':0.9}}''')

        if 'xc' in kwargs:
            self.set_xc_params(kwargs['xc'])
        for key in kwargs:
            if key in self.float_params:
                self.float_params[key] = kwargs[key]
            elif key in self.exp_params:
                self.exp_params[key] = kwargs[key]
            elif key in self.string_params:
                self.string_params[key] = kwargs[key]
            elif key in self.int_params:
                self.int_params[key] = kwargs[key]
            elif key in self.bool_params:
                self.bool_params[key] = kwargs[key]
            elif key in self.list_bool_params:
                self.list_bool_params[key] = kwargs[key]
            elif key in self.list_int_params:
                self.list_int_params[key] = kwargs[key]
            elif key in self.list_float_params:
                self.list_float_params[key] = kwargs[key]
            elif key in self.special_params:
                self.special_params[key] = kwargs[key]
            elif key in self.dict_params:
                self.dict_params[key] = kwargs[key]
            elif key in self.input_params:
                self.input_params[key] = kwargs[key]
            else:
                raise TypeError('Parameter not defined: ' + key)

    def check_xc(self):
        """Make sure the calculator has functional & pseudopotentials set up

        If no XC combination, GGA functional or POTCAR type is specified,
        default to PW91. Otherwise, try to guess the desired pseudopotentials.
        """

        p = self.input_params

        # There is no way to correctly guess the desired
        # set of pseudopotentials without 'pp' being set.
        # Usually, 'pp' will be set by 'xc'.
        if 'pp' not in p or p['pp'] is None:
            if self.string_params['gga'] is None:
                p.update({'pp': 'lda'})
            elif self.string_params['gga'] == '91':
                p.update({'pp': 'pw91'})
            elif self.string_params['gga'] == 'PE':
                p.update({'pp': 'pbe'})
            else:
                raise NotImplementedError(
                    "Unable to guess the desired set of pseudopotential"
                    "(POTCAR) files. Please do one of the following: \n"
                    "1. Use the 'xc' parameter to define your XC functional."
                    "These 'recipes' determine the pseudopotential file as "
                    "well as setting the INCAR parameters.\n"
                    "2. Use the 'gga' settings None (default), 'PE' or '91'; "
                    "these correspond to LDA, PBE and PW91 respectively.\n"
                    "3. Set the POTCAR explicitly with the 'pp' flag. The "
                    "value should be the name of a folder on the VASP_PP_PATH"
                    ", and the aliases 'LDA', 'PBE' and 'PW91' are also"
                    "accepted.\n")

        if (p['xc'] is not None and
                p['xc'].lower() == 'lda' and
                p['pp'].lower() != 'lda'):
            warnings.warn("XC is set to LDA, but PP is set to "
                          "{0}. \nThis calculation is using the {0} "
                          "POTCAR set. \n Please check that this is "
                          "really what you intended!"
                          "\n".format(p['pp'].upper()))

    def initialize(self, atoms):
        """Initialize a VASP calculation

        Constructs the POTCAR file (does not actually write it).
        User should specify the PATH
        to the pseudopotentials in VASP_PP_PATH environment variable

        The pseudopotentials are expected to be in:
        LDA:  $VASP_PP_PATH/potpaw/
        PBE:  $VASP_PP_PATH/potpaw_PBE/
        PW91: $VASP_PP_PATH/potpaw_GGA/

        if your pseudopotentials are somewhere else, or named
        differently you may make symlinks at the paths above that
        point to the right place. Alternatively, you may pass the full
        name of a folder on the VASP_PP_PATH to the 'pp' parameter.
        """

        p = self.input_params

        self.check_xc()
        self.all_symbols = atoms.get_chemical_symbols()
        self.natoms = len(atoms)
        self.spinpol = atoms.get_initial_magnetic_moments().any()
        atomtypes = atoms.get_chemical_symbols()

        # Determine the number of atoms of each atomic species
        # sorted after atomic species
        special_setups = []
        symbols = []
        symbolcount = {}

        # Default setup lists are available: 'minimal', 'recommended' and 'GW'
        # These may be provided as a string e.g.::
        #
        #     calc = Vasp(setups='recommended')
        #
        # or in a dict with other specifications e.g.::
        #
        #    calc = Vasp(setups={'base': 'minimal', 'Ca': '_sv', 2: 'O_s'})
        #
        # Where other keys are either atom identities or indices, and the
        # corresponding values are suffixes or the full name of the setup
        # folder, respectively.

        # Default to minimal basis
        if p['setups'] is None:
            p['setups'] = {'base': 'minimal'}

        # String shortcuts are initialised to dict form
        elif isinstance(p['setups'], str):
            if p['setups'].lower() in ('minimal', 'recommended', 'gw'):
                p['setups'] = {'base': p['setups']}

        # Dict form is then queried to add defaults from setups.py.
        if 'base' in p['setups']:
            setups = setups_defaults[p['setups']['base'].lower()]
        else:
            setups = {}

        # Override defaults with user-defined setups
        if p['setups'] is not None:
            setups.update(p['setups'])

        for m in setups:
            try:
                special_setups.append(int(m))
            except ValueError:
                continue

        for m, atom in enumerate(atoms):
            symbol = atom.symbol
            if m in special_setups:
                pass
            else:
                if symbol not in symbols:
                    symbols.append(symbol)
                    symbolcount[symbol] = 1
                else:
                    symbolcount[symbol] += 1

        # Build the sorting list
        self.sort = []
        self.sort.extend(special_setups)

        for symbol in symbols:
            for m, atom in enumerate(atoms):
                if m in special_setups:
                    pass
                else:
                    if atom.symbol == symbol:
                        self.sort.append(m)
        self.resort = list(range(len(self.sort)))
        for n in range(len(self.resort)):
            self.resort[self.sort[n]] = n
        self.atoms_sorted = atoms[self.sort]

        # Check if the necessary POTCAR files exists and
        # create a list of their paths.
        self.symbol_count = []
        for m in special_setups:
            self.symbol_count.append([atomtypes[m], 1])
        for m in symbols:
            self.symbol_count.append([m, symbolcount[m]])

        sys.stdout.flush()

        # Potpaw folders may be identified by an alias or full name
        for pp_alias, pp_folder in (('lda', 'potpaw'),
                                    ('pw91', 'potpaw_GGA'),
                                    ('pbe', 'potpaw_PBE')):
            if p['pp'].lower() == pp_alias:
                break
        else:
            pp_folder = p['pp']

        if 'VASP_PP_PATH' in os.environ:
            pppaths = os.environ['VASP_PP_PATH'].split(':')
        else:
            pppaths = []
        self.ppp_list = []
        # Setting the pseudopotentials, first special setups and
        # then according to symbols
        for m in special_setups:
            if m in setups:
                special_setup_index = m
            elif str(m) in setups:
                special_setup_index = str(m)
            else:
                raise Exception("Having trouble with special setup index {0}."
                                " Please use an int.".format(m))
            potcar = join(pp_folder,
                          setups[special_setup_index],
                          'POTCAR')
            for path in pppaths:
                filename = join(path, potcar)

                if isfile(filename) or islink(filename):
                    self.ppp_list.append(filename)
                    break
                elif isfile(filename + '.Z') or islink(filename + '.Z'):
                    self.ppp_list.append(filename + '.Z')
                    break
            else:
                print('Looking for %s' % potcar)
                raise RuntimeError('No pseudopotential for %s!' % symbol)

        for symbol in symbols:
            try:
                potcar = join(pp_folder, symbol + setups[symbol],
                              'POTCAR')
            except (TypeError, KeyError):
                potcar = join(pp_folder, symbol, 'POTCAR')
            for path in pppaths:
                filename = join(path, potcar)

                if isfile(filename) or islink(filename):
                    self.ppp_list.append(filename)
                    break
                elif isfile(filename + '.Z') or islink(filename + '.Z'):
                    self.ppp_list.append(filename + '.Z')
                    break
            else:
                print('''Looking for %s
                The pseudopotentials are expected to be in:
                LDA:  $VASP_PP_PATH/potpaw/
                PBE:  $VASP_PP_PATH/potpaw_PBE/
                PW91: $VASP_PP_PATH/potpaw_GGA/''' % potcar)
                raise RuntimeError('No pseudopotential for %s!' % symbol)
        self.converged = None
        self.setups_changed = None

    def write_input(self, atoms, directory='./'):
        from ase.io.vasp import write_vasp
        write_vasp(join(directory, 'POSCAR'),
                   self.atoms_sorted,
                   symbol_count=self.symbol_count,
                   ignore_constraints=self.input_params['ignore_constraints'])
        self.write_incar(atoms, directory=directory)
        self.write_potcar(directory=directory)
        self.write_kpoints(directory=directory)
        self.write_sort_file(directory=directory)
        self.copy_vdw_kernel(directory=directory)

    def copy_vdw_kernel(self, directory='./'):
        """Method to copy the vdw_kernel.bindat file.
        Set ASE_VASP_VDW environment variable to the vdw_kernel.bindat
        folder location. Checks if LUSE_VDW is enabled, and if no location
        for the vdW kernel is specified, a warning is issued."""

        vdw_env = 'ASE_VASP_VDW'
        kernel = 'vdw_kernel.bindat'
        dst = os.path.join(directory, kernel)

        # No need to copy the file again
        if isfile(dst):
            return

        if self.bool_params['luse_vdw']:
            src = None
            if vdw_env in os.environ:
                src = os.path.join(os.environ[vdw_env],
                                   kernel)

            if not src or not isfile(src):
                warnings.warn(('vdW has been enabled, however no'
                               ' location for the {} file'
                               ' has been specified.'
                               ' Set {} environment variable to'
                               ' copy the vdW kernel.').format(
                                   kernel, vdw_env))
            else:
                shutil.copyfile(src, dst)

    def clean(self):
        """Method which cleans up after a calculation.

        The default files generated by Vasp will be deleted IF this
        method is called.

        """
        files = ['CHG', 'CHGCAR', 'POSCAR', 'INCAR', 'CONTCAR',
                 'DOSCAR', 'EIGENVAL', 'IBZKPT', 'KPOINTS', 'OSZICAR',
                 'OUTCAR', 'PCDAT', 'POTCAR', 'vasprun.xml',
                 'WAVECAR', 'XDATCAR', 'PROCAR', 'ase-sort.dat',
                 'LOCPOT', 'AECCAR0', 'AECCAR1', 'AECCAR2']
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

    def write_incar(self, atoms, directory='./', **kwargs):
        """Writes the INCAR file."""
        # jrk 1/23/2015 I added this flag because this function has
        # two places where magmoms get written. There is some
        # complication when restarting that often leads to magmom
        # getting written twice. this flag prevents that issue.
        magmom_written = False
        incar = open(join(directory, 'INCAR'), 'w')
        incar.write('INCAR created by Atomic Simulation Environment\n')
        for key, val in self.float_params.items():
            if val is not None:
                incar.write(' %s = %5.6f\n' % (key.upper(), val))
        for key, val in self.exp_params.items():
            if val is not None:
                incar.write(' %s = %5.2e\n' % (key.upper(), val))
        for key, val in self.string_params.items():
            if val is not None:
                incar.write(' %s = %s\n' % (key.upper(), val))
        for key, val in self.int_params.items():
            if val is not None:
                incar.write(' %s = %d\n' % (key.upper(), val))
                if key == 'ichain' and val > 0:
                    incar.write(' IBRION = 3\n POTIM = 0.0\n')
                    for key, val in self.int_params.items():
                        if key == 'iopt' and val is None:
                            print('WARNING: optimization is '
                                  'set to LFBGS (IOPT = 1)')
                            incar.write(' IOPT = 1\n')
                    for key, val in self.exp_params.items():
                        if key == 'ediffg' and val is None:
                            RuntimeError('Please set EDIFFG < 0')

        for key, val in self.list_bool_params.items():
            if val is None:
                pass
            else:
                incar.write(' %s = ' % key.upper())
                [incar.write('%s ' % _to_vasp_bool(x)) for x in val]
                incar.write('\n')

        for key, val in self.list_int_params.items():
            if val is None:
                pass
            elif key == 'ldaul' and (self.dict_params['ldau_luj'] is not None):
                pass
            else:
                incar.write(' %s = ' % key.upper())
                [incar.write('%d ' % x) for x in val]
                incar.write('\n')

        for key, val in self.list_float_params.items():
            if val is None:
                pass
            elif ((key in ('ldauu', 'ldauj')) and
                  (self.dict_params['ldau_luj'] is not None)):
                pass
            elif key == 'magmom':
                incar.write(' %s = ' % key.upper())
                magmom_written = True
                # Work out compact a*x b*y notation and write in this form
                list = [[1, val[0]]]
                for n in range(1, len(val)):
                    if val[n] == val[n - 1]:
                        list[-1][0] += 1
                    else:
                        list.append([1, val[n]])
                    [incar.write('%i*%.4f ' % (mom[0], mom[1]))
                     for mom in list]
                    incar.write('\n')
            else:
                    incar.write(' %s = ' % key.upper())
                    [incar.write('%.4f ' % x) for x in val]
                    incar.write('\n')

        for key, val in self.bool_params.items():
            if val is not None:
                incar.write(' %s = ' % key.upper())
                if val:
                    incar.write('.TRUE.\n')
                else:
                    incar.write('.FALSE.\n')
        for key, val in self.special_params.items():
            if val is not None:
                incar.write(' %s = ' % key.upper())
                if key == 'lreal':
                    if isinstance(val, basestring):
                        incar.write(val + '\n')
                    elif isinstance(val, bool):
                        if val:
                            incar.write('.TRUE.\n')
                        else:
                            incar.write('.FALSE.\n')
        for key, val in self.dict_params.items():
            if val is not None:
                if key == 'ldau_luj':
                    llist = ulist = jlist = ''
                    for symbol in self.symbol_count:
                        #  default: No +U
                        luj = val.get(symbol[0], {'L': -1, 'U': 0.0, 'J': 0.0})
                        llist += ' %i' % luj['L']
                        ulist += ' %.3f' % luj['U']
                        jlist += ' %.3f' % luj['J']
                    incar.write(' LDAUL =%s\n' % llist)
                    incar.write(' LDAUU =%s\n' % ulist)
                    incar.write(' LDAUJ =%s\n' % jlist)

        if self.spinpol and not magmom_written:
            if not self.int_params['ispin']:
                incar.write(' ispin = 2\n'.upper())
            # Write out initial magnetic moments
            magmom = atoms.get_initial_magnetic_moments()[self.sort]
            # unpack magmom array if three components specified
            if magmom.ndim > 1:
                magmom = [item for sublist in magmom for item in sublist]
            list = [[1, magmom[0]]]
            for n in range(1, len(magmom)):
                if magmom[n] == magmom[n - 1]:
                    list[-1][0] += 1
                else:
                    list.append([1, magmom[n]])
            incar.write(' magmom = '.upper())
            [incar.write('%i*%.4f ' % (mom[0], mom[1])) for mom in list]
            incar.write('\n')
        incar.close()

    def write_kpoints(self, directory='./', **kwargs):
        """Writes the KPOINTS file."""

        # Don't write anything if KSPACING is being used
        if self.float_params['kspacing'] is not None:
            if self.float_params['kspacing'] > 0:
                return
            else:
                raise ValueError("KSPACING value {0} is not allowable. "
                                 "Please use None or a positive number."
                                 "".format(self.float_params['kspacing']))

        p = self.input_params
        kpoints = open(join(directory, 'KPOINTS'), 'w')
        kpoints.write('KPOINTS created by Atomic Simulation Environment\n')

        if isinstance(p['kpts'], dict):
            p['kpts'] = kpts2ndarray(p['kpts'], atoms=self.atoms)
            p['reciprocal'] = True

        shape = np.array(p['kpts']).shape

        # Wrap scalar in list if necessary
        if shape == ():
            p['kpts'] = [p['kpts']]
            shape = (1, )

        if len(shape) == 1:
            kpoints.write('0\n')
            if shape == (1, ):
                kpoints.write('Auto\n')
            elif p['gamma']:
                kpoints.write('Gamma\n')
            else:
                kpoints.write('Monkhorst-Pack\n')
            [kpoints.write('%i ' % kpt) for kpt in p['kpts']]
            kpoints.write('\n0 0 0\n')
        elif len(shape) == 2:
            kpoints.write('%i \n' % (len(p['kpts'])))
            if p['reciprocal']:
                kpoints.write('Reciprocal\n')
            else:
                kpoints.write('Cartesian\n')
            for n in range(len(p['kpts'])):
                [kpoints.write('%f ' % kpt) for kpt in p['kpts'][n]]
                if shape[1] == 4:
                    kpoints.write('\n')
                elif shape[1] == 3:
                    kpoints.write('1.0 \n')
        kpoints.close()

    def write_potcar(self, suffix="", directory='./'):
        """Writes the POTCAR file."""
        import tempfile
        potfile = open(join(directory, 'POTCAR' + suffix), 'w')
        for filename in self.ppp_list:
            if filename.endswith('R'):
                for line in open(filename, 'r'):
                    potfile.write(line)
            elif filename.endswith('.Z'):
                file_tmp = tempfile.NamedTemporaryFile()
                os.system('gunzip -c %s > %s' % (filename, file_tmp.name))
                for line in file_tmp.readlines():
                    potfile.write(line)
                file_tmp.close()
        potfile.close()

    def write_sort_file(self, directory='./'):
        """Writes a sortings file.

        This file contains information about how the atoms are sorted in
        the first column and how they should be resorted in the second
        column. It is used for restart purposes to get sorting right
        when reading in an old calculation to ASE."""

        file = open(join(directory, 'ase-sort.dat'), 'w')
        for n in range(len(self.sort)):
            file.write('%5i %5i \n' % (self.sort[n], self.resort[n]))

# The below functions are used to restart a calculation and are under early
# constructions

    def read_incar(self, filename='INCAR'):
        """Method that imports settings from INCAR file."""

        self.spinpol = False
        file = open(filename, 'r')
        file.readline()
        lines = file.readlines()
        for line in lines:
            try:
                # Make multiplication, comments, and parameters easier to spot
                line = line.replace("*", " * ")
                line = line.replace("=", " = ")
                line = line.replace("#", "# ")
                data = line.split()
                # Skip empty and commented lines.
                if len(data) == 0:
                    continue
                elif data[0][0] in ['#', '!']:
                    continue
                key = data[0].lower()
                if key in float_keys:
                    self.float_params[key] = float(data[2])
                elif key in exp_keys:
                    self.exp_params[key] = float(data[2])
                elif key in string_keys:
                    self.string_params[key] = str(data[2])
                elif key in int_keys:
                    if key == 'ispin':
                        # JRK added. not sure why we would want to leave ispin
                        # out
                        self.int_params[key] = int(data[2])
                        if int(data[2]) == 2:
                            self.spinpol = True
                    else:
                        self.int_params[key] = int(data[2])
                elif key in bool_keys:
                    if 'true' in data[2].lower():
                        self.bool_params[key] = True
                    elif 'false' in data[2].lower():
                        self.bool_params[key] = False

                elif key in list_bool_keys:
                    self.list_bool_keys[key] = [_from_vasp_bool(x) for x in
                                                _args_without_comment(data[2:])]

                elif key in list_int_keys:
                    self.list_int_params[key] = [int(x) for x in
                                                 _args_without_comment(data[2:])]

                elif key in list_float_keys:
                    if key == 'magmom':
                        lst = []
                        i = 2
                        while i < len(data):
                            if data[i] in ["#", "!"]:
                                break
                            if data[i] == "*":
                                b = lst.pop()
                                i += 1
                                for j in range(int(b)):
                                    lst.append(float(data[i]))
                            else:
                                lst.append(float(data[i]))
                            i += 1
                        self.list_float_params['magmom'] = lst
                        lst = np.array(lst)
                        if self.atoms is not None:
                            self.atoms.set_initial_magnetic_moments(
                                lst[self.resort])
                    else:
                        data = _args_without_comment(data)
                        self.list_float_params[key] = [float(x) for x in data[2:]]
                # elif key in list_keys:
                #     list = []
                #     if key in ('dipol', 'eint', 'ferwe', 'ferdo',
                #                'ropt', 'rwigs',
                #                'ldauu', 'ldaul', 'ldauj', 'langevin_gamma'):
                #         for a in data[2:]:
                #             if a in ["!", "#"]:
                #                 break
                #             list.append(float(a))
                #     elif key in ('iband', 'kpuse', 'random_seed'):
                #         for a in data[2:]:
                #             if a in ["!", "#"]:
                #                 break
                #             list.append(int(a))
                #     self.list_params[key] = list
                #     if key == 'magmom':
                #         list = []
                #         i = 2
                #         while i < len(data):
                #             if data[i] in ["#", "!"]:
                #                 break
                #             if data[i] == "*":
                #                 b = list.pop()
                #                 i += 1
                #                 for j in range(int(b)):
                #                     list.append(float(data[i]))
                #             else:
                #                 list.append(float(data[i]))
                #             i += 1
                #         self.list_params['magmom'] = list
                #         list = np.array(list)
                #         if self.atoms is not None:
                #             self.atoms.set_initial_magnetic_moments(
                #                 list[self.resort])
                elif key in special_keys:
                    if key == 'lreal':
                        if 'true' in data[2].lower():
                            self.special_params[key] = True
                        elif 'false' in data[2].lower():
                            self.special_params[key] = False
                        else:
                            self.special_params[key] = data[2]
            except KeyError:
                raise IOError('Keyword "%s" in INCAR is'
                              'not known by calculator.' % key)
            except IndexError:
                raise IOError('Value missing for keyword "%s".' % key)

    def read_kpoints(self, filename='KPOINTS'):
        file = open(filename, 'r')
        lines = file.readlines()
        file.close()
        ktype = lines[2].split()[0].lower()[0]
        if ktype in ['g', 'm', 'a']:
            if ktype == 'g':
                self.set(gamma=True)
                kpts = np.array([int(lines[3].split()[i]) for i in range(3)])
            elif ktype == 'a':
                kpts = np.array([int(lines[3].split()[i]) for i in range(1)])
            elif ktype == 'm':
                kpts = np.array([int(lines[3].split()[i]) for i in range(3)])
        else:
            if ktype in ['c', 'k']:
                self.set(reciprocal=False)
            else:
                self.set(reciprocal=True)
            kpts = np.array([list(map(float, line.split()))
                             for line in lines[3:]])
        self.set(kpts=kpts)

    def read_potcar(self):
        """ Read the pseudopotential XC functional from POTCAR file.
        """
        file = open('POTCAR', 'r')
        lines = file.readlines()
        file.close()

        # Search for key 'LEXCH' in POTCAR
        xc_flag = None
        for line in lines:
            key = line.split()[0].upper()
            if key == 'LEXCH':
                xc_flag = line.split()[-1].upper()
                break

        if xc_flag is None:
            raise ValueError('LEXCH flag not found in POTCAR file.')

        # Values of parameter LEXCH and corresponding XC-functional
        xc_dict = {'PE': 'PBE', '91': 'PW91', 'CA': 'LDA'}

        if xc_flag not in xc_dict.keys():
            raise ValueError('Unknown xc-functional flag found in POTCAR,'
                             ' LEXCH=%s' % xc_flag)

        self.input_params['pp'] = xc_dict[xc_flag]

    def todict(self):
        """Returns a dictionary of all parameters
        that can be used to construct a new calculator object"""
        dict_list = [
            'float_params',
            'exp_params',
            'string_params',
            'int_params',
            'bool_params',
            'list_bool_params',
            'list_int_params',
            'list_float_params',
            'special_params',
            'dict_params',
            'input_params'
        ]
        dct = {}
        for item in dict_list:
            dct.update(getattr(self, item))
        for key, val in list(dct.items()):
            if val is None:
                del(dct[key])
        return dct


def _args_without_comment(data, marks=['!', '#']):
    """Check split arguments list for a comment, return data up to marker

    INCAR reader splits list arguments on spaces and leaves comment markers as
    individual items. This function returns only the data portion of the list.

    """
    comment_locs = [data.index(mark) for mark in marks
                    if mark in data]
    if comment_locs == []:
        return data
    else:
        return data[:min(comment_locs)]


def _from_vasp_bool(x):
    """Cast vasp boolean to Python bool

    VASP files sometimes use T or F as shorthand for the preferred Boolean
    notation .TRUE. or .FALSE. As capitalisation is pretty inconsistent in
    practice, we allow all cases to be cast to a Python bool.

    """
    assert isinstance(x, str)
    if x.lower() == '.true.' or x.lower() == 't':
        return True
    elif x.lower() == '.false.' or x.lower() == 'f':
        return False
    else:
        raise ValueError('Value "%s" not recognized as bool' % x)


def _to_vasp_bool(x):
    """Convert Python boolean to string for VASP input

    In case the value was modified to a string already, appropriate strings
    will also be accepted and cast to a standard .TRUE. / .FALSE. format.

    """
    if isinstance(x, str):
        if x.lower() in ('.true.', 't'):
            x = True
        elif x.lower() in ('.false.', 'f'):
            x = False
        else:
            raise ValueError('"%s" not recognised as VASP Boolean')
    assert isinstance(x, bool)
    if x:
        return '.TRUE.'
    else:
        return '.FALSE.'
