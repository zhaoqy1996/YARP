"""This module defines an ASE interface to GROMACS.

http://www.gromacs.org/
It is VERY SLOW compared to standard Gromacs
(due to slow formatted io required here).

Mainly intended to be the MM part in the ase QM/MM

Markus.Kaukonen@iki.fi

To be done:
1) change the documentation for the new file-io-calculator (test works now)
2) change gromacs program names
-now:     hard coded
-future:  set as dictionary in params_runs

"""

import os
from glob import glob

import numpy as np

from ase.calculators.calculator import FileIOCalculator, all_changes


def do_clean(name='#*'):
    """ remove files matching wildcards """
    myfiles = glob(name)
    for myfile in myfiles:
        try:
            os.remove(myfile)
        except OSError:
            pass


class Gromacs(FileIOCalculator):
    """Class for doing GROMACS calculations.
    Before running a gromacs calculation you must prepare the input files
    separately (pdb2gmx and grompp for instance.)

    Input parameters for gromacs runs (the .mdp file)
    are given in self.params and can be set when initializing the calculator
    or by method set_own.
    for example::
        
        CALC_MM_RELAX = Gromacs()
        CALC_MM_RELAX.set_own_params('integrator', 'steep',
                                     'use steepest descent')

    Run command line arguments for gromacs related programs:
    pdb2gmx, grompp, mdrun, g_energy, g_traj.  These can be given as::
        
        CALC_MM_RELAX = Gromacs()
        CALC_MM_RELAX.set_own_params_runs('force_field', 'oplsaa')
    """

    implemented_properties = ['energy', 'forces']
    command = 'mdrun < PREFIX.files > PREFIX.log'

    default_parameters = dict(
        define='-DFLEXIBLE',
        integrator='cg',
        nsteps='10000',
        nstfout='10',
        nstlog='10',
        nstenergy='10',
        nstlist='10',
        ns_type='grid',
        pbc='xyz',
        rlist='1.15',
        coulombtype='PME-Switch',
        rcoulomb='0.8',
        vdwtype='shift',
        rvdw='0.8',
        rvdw_switch='0.75',
        DispCorr='Ener')

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='gromacs', atoms=None,
                 do_qmmm=False, freeze_qm=False, clean=True,
                 water_model='tip3p', force_field='oplsaa',
                 **kwargs):
        """Construct GROMACS-calculator object.

        Parameters
        ==========
        label: str
            Prefix to use for filenames (label.in, label.txt, ...).
            Default is 'gromacs'.

        do_qmmm : bool
            Is gromacs used as mm calculator for a qm/mm calculation

        freeze_qm : bool
            In qm/mm are the qm atoms kept fixed at their initial positions

        clean :     bool
            Remove gromacs backup files
            and old gormacs.* files

        water_model: str
            Water model to be used in gromacs runs (see gromacs manual)

        force_field: str
            Force field to be used in gromacs runs
        """

        self.do_qmmm = do_qmmm
        self.freeze_qm = freeze_qm
        self.water_model = water_model
        self.force_field = force_field
        self.clean = clean
        self.params_doc = {}
        # add comments for gromacs input file
        self.params_doc['define'] = \
            'flexible/ rigid water'
        self.params_doc['integrator'] = \
            'md: molecular dynamics(Leapfrog), \n' + \
            '; md-vv: molecular dynamics(Velocity Verlet), \n' + \
            '; steep: steepest descent minimization, \n' + \
            '; cg: conjugate cradient minimization \n'

        self.positions = None
        self.atoms = None
        # storage for energy and forces
        #self.energy = None
        #self.forces = None

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)
        self.set(**kwargs)
        # default values for runtime parameters
        # can be changed by self.set_own_params_runs('key', 'value')
        self.params_runs = {}
        self.params_runs['index_filename'] = 'index.ndx'
        self.params_runs['init_structure'] = self.label + '.pdb'
        self.params_runs['water'] = self.water_model
        self.params_runs['force_field'] = self.force_field
        self.params_runs['extra_mdrun_parameters'] = ' -nt 1 '
        self.params_runs['extra_pdb2gmx_parameters'] = ' '
        self.params_runs['extra_grompp_parameters'] = ' '
        self.params_runs['extra_editconf_parameters'] = ' '
        self.params_runs['extra_genbox_parameters'] = ' '

        # these below are required by qm/mm
        self.topology_filename = self.label + '.top'
        self.name = 'Gromacs'

        # clean up gromacs backups
        if self.clean:
            do_clean('gromacs.???')

        # write input files for gromacs program g_energy
        self.write_g_energy_files()

        # a possible prefix for gromacs programs
        if 'GMXCMD_PREF' in os.environ:
            self.prefix = os.environ['GMXCMD_PREF']
        else:
            self.prefix = ''

        # a possible postfix for gromacs programs
        if 'GMXCMD_POST' in os.environ:
            self.postfix = os.environ['GMXCMD_POST']
        else:
            self.postfix = ''

        if self.do_qmmm:
            self.parameters['integrator'] = 'md'
            self.parameters['nsteps'] = '0'

    def generate_g96file(self):
        """ from current coordinates (self.structure_file)
            write a structure file in .g96 format
        """
        from ase.io.gromos import write_gromos
        # generate structure file in g96 format
        write_gromos(self.label + '.g96', self.atoms)

    def run_editconf(self):
        """ run gromacs program editconf, typically to set a simulation box
        writing to the input structure"""
        command = 'editconf' + ' '
        os.system(command +
                  ' -f ' + self.label + '.g96' +
                  ' -o ' + self.label + '.g96' +
                  ' ' +
                  self.params_runs.get('extra_editconf_parameters') +
                  ' > /dev/null 2>&1')

    def run_genbox(self):
        """Run gromacs program genbox, typically to solvate the system
        writing to the input structure
        as extra parameter you need to define the file containing the solvent

        for instance::

           CALC_MM_RELAX = Gromacs()
           CALC_MM_RELAX.set_own_params_runs(
                'extra_genbox_parameters', '-cs spc216.gro')
        """
        command = 'genbox' + ' '
        os.system(command + \
                      ' -cp ' + self.label + '.g96' + \
                      ' -o ' + self.label + '.g96' + \
                      ' -p ' + self.label + '.top' + \
                      ' ' + self.params_runs.get('extra_genbox_parameters') +\
                      ' > /dev/null 2>&1')

    def run(self):
        """ runs a gromacs-mdrun with the
        current atom-configuration """
        from ase.io.gromos import read_gromos

        # clean up gromacs backups
        if self.clean:
            do_clean('#*')

        command = 'mdrun'
        if self.do_qmmm:
            os.system(command \
                          + ' -s ' + self.label + '.tpr' \
                          + ' -o ' + self.label + '.trr ' \
                          + ' -e ' + self.label + '.edr ' \
                          + ' -g ' + self.label + '.log -ffout ' \
                          + ' -rerun ' + self.label + '.g96 ' \
                          + self.params_runs.get('extra_mdrun_parameters') \
                          + ' > mm.log 2>&1')
        else:
            os.system(command \
                          + ' -s ' + self.label + '.tpr ' \
                          + ' -o ' + self.label + '.trr ' \
                          + ' -e ' + self.label + '.edr ' \
                          + ' -g ' + self.label + '.log -ffout ' \
                          + ' -c ' + self.label + '.g96 ' \
                          + self.params_runs.get('extra_mdrun_parameters') \
                          + '  > MM.log 2>&1')
            atoms = read_gromos(self.label + '.g96')
            self.atoms = atoms.copy()

    def generate_topology_and_g96file(self):
        """ from coordinates (self.label.+'pdb')
            and gromacs run input file (self.label + '.mdp)
            generate topology (self.label+'top')
            and structure file in .g96 format (self.label + '.g96')
        """
        from ase.io.gromos import read_gromos
        #generate structure and topology files
        # In case of predefinded topology file this is not done
        command = 'pdb2gmx' + ' '
        os.system(command + \
                      ' -f ' + self.params_runs.get('init_structure') + \
                      ' -o ' + self.label + '.g96' + \
                      ' -p ' + self.label + '.top' + \
                      ' -ff ' + self.params_runs.get('force_field') + \
                      ' -water ' + self.params_runs.get('water') + \
                      ' ' + \
                      self.params_runs.get('extra_pdb2gmx_parameters') +\
                      ' > /dev/null 2>&1')
#                      ' > debug.log 2>&1')

#        print command + \
#                      ' -f ' + self.params_runs.get('init_structure') + \
#                      ' -o ' + self.label+'.g96' + \
#                      ' -p ' + self.label+'.top' + \
#                      ' -ff ' + self.params_runs.get('force_field') + \
#                      ' -water ' + self.params_runs.get('water') + \
#                      ' ' + self.params_runs.get('extra_pdb2gmx_parameters') +\
#                      ' > /dev/null 2>&1'
        atoms = read_gromos(self.label + '.g96')
        self.atoms = atoms.copy()

    def generate_gromacs_run_file(self):
        """ Generates input file for a gromacs mdrun
        based on structure file and topology file
        resulting file is self.label + '.tpr
        """

        #generate gromacs run input file (gromacs.tpr)
        try:
            os.remove(self.label + '.tpr')
        except:
            pass
        command = 'grompp '
        os.system(command + \
                      ' -f ' + self.label + '.mdp' + \
                      ' -c ' + self.label + '.g96' + \
                      ' -p ' + self.label + '.top' + \
                      ' -o ' + self.label + '.tpr -maxwarn 100' + \
                      ' ' + self.params_runs.get('extra_grompp_parameters') +\
                      ' > /dev/null 2>&1')

#        print command + \
#                      ' -f ' + self.label + '.mdp' + \
#                      ' -c ' + self.label + '.g96' + \
#                      ' -p ' + self.label + '.top' + \
#                      ' -o ' + self.label + '.tpr -maxwarn 100' + \
#                      ' ' + self.params_runs.get('extra_grompp_parameters') +\
#                      ' > /dev/null 2>&1'

    def write_g_energy_files(self):
        """write input files for gromacs force and energy calculations
        for gromacs program g_energy"""
        filename = 'inputGenergy.txt'
        output = open(filename, 'w')
        output.write('Potential  \n')
        output.write('   \n')
        output.write('   \n')
        output.close()

        filename = 'inputGtraj.txt'
        output = open(filename, 'w')
        output.write('System  \n')
        output.write('   \n')
        output.write('   \n')
        output.close()

    def set_own_params(self, key, value, docstring=""):
        """Set own gromacs parameter with doc strings."""
        self.parameters[key] = value
        self.params_doc[key] = docstring

    def set_own_params_runs(self, key, value):
        """Set own gromacs parameter for program parameters
        Add spaces to avoid errors """
        self.params_runs[key] = ' ' + value + ' '

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms=None, properties=None, system_changes=None):
        """Write input parameters to input file."""

        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        #print self.parameters
        myfile = open(self.label + '.mdp', 'w')
        for key, val in self.parameters.items():
            if val is not None:
                if (self.params_doc.get(key) == None):
                    docstring = ''
                else:
                    docstring = self.params_doc[key]
                myfile.write('%-35s = %s ; %s\n' \
                            % (key, val, ';' + docstring))
        myfile.close()
        if self.freeze_qm:
            self.add_freeze_group()

    def update(self, atoms):
        """ set atoms and do the calculation """
        from ase.io.gromos import write_gromos
        # performs an update of the atoms
        self.atoms = atoms.copy()
        #must be g96 format for accuracy, alternatively binary formats
        write_gromos(self.label + '.g96', atoms)
        # does run to get forces and energies
        self.calculate()

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):
        """ runs a gromacs-mdrun and
        gets energy and forces
        rest below is to make gromacs calculator
        compactible with ase-Calculator class

        atoms: Atoms object
            Contains positions, unit-cell, ...
        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces'
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these five: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.

        """
        from ase import units

        self.run()
        if self.clean:
            do_clean('#*')
        # get energy
        try:
            os.remove('tmp_ene.del')
        except:
            pass
        command = 'g_energy' + ' '
        os.system(command + \
                      ' -f ' + self.label + '.edr -dp ' + \
                      ' -o ' + self.label + \
                      'Energy.xvg < inputGenergy.txt' + \
                      ' > /dev/null 2>&1')
        os.system('tail -n 1 ' + self.label + \
                      'Energy.xvg > tmp_ene.del')
        line = open('tmp_ene.del', 'r').readline()
        energy = float(line.split()[1])
        #We go for ASE units !
        #self.energy = energy * units.kJ / units.mol
        self.results['energy'] = energy * units.kJ / units.mol
        # energies are about 100 times bigger in Gromacs units
        # when compared to ase units

        #get forces
        try:
            os.remove('tmp_force.del')
        except:
            pass
        #os.system('gmxdump_d -f gromacs.trr > tmp_force.del 2>/dev/null')
        command = 'g_traj' + ' '
        os.system(command +\
                      ' -f ' + self.label + '.trr -s ' \
                      + self.label + '.tpr -of ' \
                      + ' -fp ' + self.label \
                      + 'Force.xvg < inputGtraj.txt ' \
                      + ' > /dev/null 2>&1')
        lines = open(self.label + 'Force.xvg', 'r').readlines()
        forces = []
        forces.append(
            np.array([float(f) for f in lines[-1].split()[1:]]))
        #We go for ASE units !gromacsForce.xvg
        #self.forces = np.array(forces)/ units.nm * units.kJ / units.mol
        #self.forces = np.reshape(self.forces, (-1, 3))
        tmp_forces = np.array(forces) / units.nm * units.kJ / units.mol
        tmp_forces = np.reshape(tmp_forces, (-1, 3))
        self.results['forces'] = tmp_forces
        #self.forces = np.array(forces)

    def add_freeze_group(self):
        """
        Add freeze group (all qm atoms) to the gromacs index file
        and modify the 'self.base_filename'.mdp file to adopt for freeze group.
        The qm regions are read from the file index.ndx

        This is useful if one makes many moves in MM
        and then only a few with both qm and mm moving.

        qse-qm/mm indexing starts from 0
        gromacs indexing starts from 1
        """
        from ase.calculators.ase_qmmm_manyqm import get_qm_atoms

        index_filename = self.params_runs.get('index_filename')
        qms = get_qm_atoms(index_filename)
        infile = open(index_filename, 'r')
        lines = infile.readlines()
        infile.close()
        outfile = open(index_filename, 'w')
        found = False
        for line in lines:
            if ('freezeGroupQM' in line):
                found = True
            outfile.write(line)
        if not found:
            outfile.write('[ freezeGroupQM ] \n')
            for myqm in qms:
                for qmindex in myqm:
                    outfile.write(str(qmindex + 1) + ' ')
            outfile.write('\n')
        outfile.close()

        infile = open(self.label + '.mdp', 'r')
        lines = infile.readlines()
        infile.close()
        outfile = open(self.label + '.mdp', 'w')
        for line in lines:
            outfile.write(line)
        outfile.write('freezegrps = freezeGroupQM \n')
        outfile.write('freezedim  = Y Y Y  \n')
        outfile.close()
        return

    def get_command(self):
        """Return command string for gromacs mdrun.  """
        command = None
        if 'GMXCMD' in os.environ:
            command = self.prefix + os.environ['GMXCMD'] + self.postfix
        return command
