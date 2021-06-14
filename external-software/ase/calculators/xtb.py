#This module defines an ASE interface to GFN-xTB.
import os
import os.path
from subprocess import Popen, PIPE
import numpy as np
import ase.io
from ase.units import Bohr, Hartree, Rydberg
from ase.calculators.calculator import Calculator, all_changes, Parameters

class xTB:
    """Construct xTB calculator.
    The keyword arguments (kwargs) can be one of the ASE standard keywords: 'acc', 'iterations' 
    and 'opt [level]', detail is shown as following :

    -c,--chrg INT     specify molecular charge as INT
    -u,--uhf INT      specify Nalpha-Nbeta as INT (spin)
    -a,--acc REAL     accuracy for SCC calculation, lower is better (default = 1.0)
    --iteration INT number of iterations in SCC (default = 250)
    --cycles       number of cycles in ANCopt (default = automatic)
    --gfn INT      specify parametrisation of GFN-xTB (default = 2)
    --qmdff        use QMDFF for single point (needs solvent-file)
    --etemp REAL   electronic temperature (default = 300K)
    -g,--gbsa [SOLVENT [STATE]] generalized born (GB) model with solvent accessable surface area (SASA) model
    --pop          requests printout of Mulliken population analysis
    --dipole      requests dipole printout
    --wbo          requests Wiberg bond order printout
    --lmo          requests localization of orbitals
    --scc, --sp    performs a single point calculation
    --esp          calculate electrostatic potential on VdW-grid
    --grad         performs a gradient calculation
    -o,--opt [LEVEL]  call ancopt(3) to perform a geometry optimization,
                   levels from crude, sloppy, loose, normal (default),tight, verytight to extreme can be chosen
    --hess         perform a numerical hessian calculation on input geometry
    --ohess [LEVEL] perform a numerical hessian calculation on an ancopt(3) optimized geometry
    --siman        conformational search by simulated annealing based on  molecular dynamics. Conformers are optimized with ancopt.
    -I,--input FILE   use FILE as input source for xcontrol(7) instructions
    --namespace STRING give this xtb(1) run a namespace on all files, even temporary ones, will be named accordingly
    --[no]restart  restarts calculation from xtbrestart (default = true)
    -P,--parallel INT number of parallel processes
    """

    command = None
    implemented_properties = ['energy', 'forces']
    default_parameters = dict(
        accuracy = 1.0,
        max_iter = 100,
        calc_type = "grad",
        namespace = "xTB",
        opt=False,
        opt_level = "Tight",
        hess=False,
        norestart=False)

    def __init__(self,charge=0, spin=0, accuracy = 1.0, max_iter = 100, calc_type = "grad", namespace = "xTB", opt=False, opt_level = "Tight", hess=False, norestart=False, workdir=None):

        """Construct xTB-calculator object."""
        self.charge = charge
        self.spin = spin
        self.accuracy = accuracy
        self.max_iter = max_iter
        self.calc_type= calc_type
        self.namespace= namespace
        self.opt = opt
        if self.opt is True:
            self.opt_level = opt_level
        self.hess = hess
        self.norestart= norestart

        self.start_dir = None
        self.workdir = workdir
        if self.workdir:
            self.start_dir = os.getcwd()
            if not os.path.isdir(workdir):
                os.mkdir(workdir)
        else:
            self.workdir = '.'
            self.start_dir = '.'

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
            
    def run_executable(self):
        #xTB_exe='/home/zhao922/bin/xTB/xtb_6.2.3/bin/xtb'
        if self.opt is False:
            substring="xtb -c {} -u {} -a {} --iterations {} --{} --namespace '{}' {}_xtb.xyz"
        else:
            substring="xtb -c {} -u {} -a {} --iterations {} --opt --{} --namespace '{}' {}_xtb.xyz"
        code_exe=substring.format(self.charge,self.spin,self.accuracy,self.max_iter,self.calc_type,self.namespace,self.namespace)
        output = Popen(code_exe, shell=True, stdout=PIPE, stderr=PIPE).communicate()[0]
        with open("{}_xtbresult.txt".format(self.namespace),'w') as g: 
            g.write(output)
        
    def calculate(self, atoms):
        os.chdir(self.workdir)
        self.write_inp(atoms)
        self.run_executable()
        self.read(atoms)

    def get_potential_energy(self, atoms, force_consistent=False):
        self.calculate(atoms)
        return  self.energy

    def get_forces(self, atoms):
        #self.calculate(atoms)
        self.read(atoms)
        return self.forces
        
    def write_inp(self, atoms):
        #Write the input xyz_file of xTB.
        fh = open('{}_xtb.xyz'.format(self.namespace), 'w')
        natoms = len(atoms)
        fh.write(' %1d\n\n' % natoms)
        positions = atoms.get_positions()
        atomic_symbols = atoms.get_chemical_symbols()
        for (symbol, pos) in zip(atomic_symbols, positions):
            fh.write('%1s' % symbol)
            for el in pos:
                fh.write(' %21.16f' % el)
            fh.write('\n')
        fh.close()

    def read(self, atoms):
        """Read results from xTB's text-output file `out`."""

        lines = open('{}.gradient'.format(self.namespace), 'r').readlines()

        # total energies
        for ln,line in enumerate(lines):
            fields = line.split()
            if len(fields)==0: continue
            if fields[0]=="cycle":
                N_line = ln
                energy = float(fields[6])
        natoms = len(atoms)
        forces = np.zeros((natoms,3))
        fl = 0
        for ln in range(N_line+natoms*2+2)[-1-natoms:-1]:
            F = []
            fields=lines[ln].split()
            for li in range(3):
                F += [float(fields[li].split('E')[0])*10**float(fields[li].split('E')[1])]
            forces[fl] = F
            fl += 1
        self.energy = energy*Hartree
        self.forces = -forces*Hartree/Bohr
        
