from __future__ import print_function
"""QM/MM interface with QM=FHI-aims, MM=gromacs

QM could be something else, but you need to read in qm-atom charges
from the qm program (in method 'get_qm_charges')


One can have many QM regions, each with a different calculator.
There can be only one MM calculator, which is calculating the whole
system.


Non-bonded interactions:
------------------------
Generally:

Within the same QM-QM:
  by qm calculator
MM-MM:
  by MM calculator
QM-MM:
  by MM using MM vdw parameters and QM charges.
Different QM different QM:
  by MM using QM and MM charges and MM-vdw parameters

The Hirschfeld charges (or other atomic charges)
on QM atoms are calculated by QM in a H terminated cluster in vacuum.
The charge of QM atom next to MM atom (edge-QM-atom)
and its H neighbors are set as in the classical force field.
The extra(missing) charge results from:

1) linkH atoms
2) The edge-QM atoms, and their qm-H neighbors,
   have their original MM charges.
3) and the fact that the charge of the QM fraction
   is not usually an integer when using the original MM charges.
   It is added equally to all QM atoms
   (not being linkH and not being edge-QM-atom or its H neighbor)
   so that the total charge of the MM-fragment involving QM atoms
   will be the same as in the original MM-description.

Vdw interactions are calculated by MM-gromacs for MM and MM-QM inteactions.
The QM-QM vdw interaction s could be done by the FHI-aims if desired
(by modifying the imput for QM-FHI-aims input accordingly.

Bonded interactions::

  E=
  E_qm(QM-H)         ; qm energy of H terminated QM cluster(s)
  + E_mm(ALL ATOMS)  ; mm energy of all atoms,
                     ; except for terms in which all MM-interacting atoms are
                     ; in the same QM region

Forces do not act on link atoms but they are positioned by scaling.
Forces on link atoms are given to their QM and MM neighbors by chain rule.
(see J. Chem. Theory Comput. 2011, 7, 761-777).
The optimal edge-qm-atom-linkH bond length is calculated
by QM in 'get_eq_qm_atom_link_h_distances'
or they are read from a file.


Questions & Comments markus.kaukonen@iki.fi

I'm especially interested in cases when we need two or more
QM regions. For instance two redox centers in a protein,
cathode and anode of a fuel cell ... you name it!


Some things to improve:

1) Water topology issue (at the moment water cannot be in QM),
   Its topology should be put into the main
   topology file, not in a separate file.

2) point charges and periodicity (if desired) to the QM calculation
   (now in vacuum)

3) Eichinger type of link atom treatment with fitted force constants for
   linkH-QMedge (bond strecth)
   linkH-QMedge-QMnextTOedge (angle terms)

4) file io using unformatted formats (.trr) instead of g96
   This is not easily possible without loading extra stuff from
   ftp://ftp.gromacs.org/pub/contrib/xd...e-1.1.1.tar.gz.

5) Utilize gromacs-python wrapper: (just found this today 31.12.2012...)
   http://orbeckst.github.com/GromacsWrapper/index.html#
"""

import sys
import numpy as np


def get_neighbor_list(system):
    """
    Makes a neighbor list of a system (ase Atoms).
    See
    https:\
    //wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#module-calculators
    """

    from ase.neighborlist import NeighborList
    from ase.data import covalent_radii
    import os
    import pickle

    NEIGHBOR_FILE = 'neighbor_list_for_ase_qmmm.txt'

    if os.path.exists(NEIGHBOR_FILE):
        print('Reading qm/mm neighbor list from file:')
        print('neighbor_list_for_ase_qmmm.txt')

        myfile = open(NEIGHBOR_FILE, 'rb')
        neighbor_list = pickle.load(myfile)
    else:
        cut = [covalent_radii[atom.number] for atom in system]
        skin = [0.2 for atom in system]
        
        neighbor_list = NeighborList(cut, skin, \
                                         self_interaction=False, bothways=True)
        neighbor_list.update(system)
        file = open(NEIGHBOR_FILE, 'wb')
        pickle.dump(neighbor_list, file)
        file.close()

    return neighbor_list

def get_qm_atoms(indexfilename='index.ndx'):
    """
    Read the indexes of all QM atoms (there may be many QM regions)
    """

    infile = open(indexfilename,'r')
    lines = infile.readlines()
    infile.close()
    qms = []

    for iline, line in enumerate(lines):
        if (('[ QM' in line) or ('[ qm' in line) or ('[ Qm' in line)) \
                or (('[QM' in line) or ('[qm' in line) or ('[Qm' in line)):
            qm = []
            for checkline in lines[iline+1:]:
                if ('[') in checkline:
                    break
                else:
                    qm = qm + [int(float(s)-1.0) for s in \
                                   checkline.split() if s.isdigit()]
            qm = list(set(qm))
            qms.append(qm)
    return qms

class LinkAtom:
    """
    Class for information about a single link-atom
    (it terminates a QM cluster)

    qm_region_index and link_atom_index refer to the following indexing system:
    [[QM0 link atoms indexes from 0],[QM1 link atoms indexes from 0],...]
    So above the second link atom in second qm region would have
    qm_region_index=1,  link_atom_index=1

    link_atom_index_in_qm tells which index in qm system the link atom has
    for instance
    qm_region_index=1, link_atom_index_in_qm=20
    means that link atom is 21'st atom in the second qm system

    """
    def __init__(self, atom, qm_region_index, link_atom_index):
        """ set initial values to a link atom object """
        self.atom = atom
        self.qm_region_index = qm_region_index
        self.link_atom_index = link_atom_index
        self.link_atom_index_in_qm = None
        self.qm_neighbor = None
        self.mm_neighbor = None
        self.qm2_neighbors = []
        self.qm3_neighbors = []
        self.mm2_neighbors = []
        self.set_qm2_neighbors = set([])
        self.set_qm3_neighbors = set([])
        self.set_mm2_neighbors = set([])
        self.force_constant = 0.0
        self.equilibrium_distance_xh = 0.0
        self.equilibrium_distance_xy = 0.0


    def set_link_atom(self, atom):
        """ set an ase-atom to be the link atom """
        self.atom = atom

    def set_link_atom_qm_region_index(self, qm_region_index):
        """ set to which qm region the link atom belongs to """
        self.qm_region_index = qm_region_index

    def set_link_atom_index_in_qm(self, link_atom_index_in_qm):
        """ set what is my link atom index in this qm region """
        self.link_atom_index_in_qm = link_atom_index_in_qm

    def set_link_atom_qm_neighbor(self, qm_neighbor):
        """ set what index does my qm neighbor have"""
        self.qm_neighbor = qm_neighbor

    def set_link_atom_mm_neighbor(self, mm_neighbor):
        """ set what index does my mm neighbor have"""
        self.mm_neighbor = mm_neighbor

    def set_link_atom_qm2_neighbors(self, qm2_neighbors):
        """ set what index does my second qm neighbor have"""
        self.qm2_neighbors = qm2_neighbors

    def set_link_atom_qm3_neighbors(self, qm3_neighbors):
        """ set what index does my third qm neighbor have"""
        self.qm3_neighbors = qm3_neighbors

    def set_link_atom_mm2_neighbors(self, mm2_neighbors):
        """ set what index does my second mm neighbor have"""
        self.mm2_neighbors = mm2_neighbors

    def set_force_constant(self, force_constant):
        """ set the force constant of bond edge-qm -- linkH (not used)"""
        self.force_constant = force_constant

    def set_equilibrium_distance_xh(self, equilibrium_distance_xh):
        """ set the equilibrium edge-qm -- linkH distance """
        self.equilibrium_distance_xh = equilibrium_distance_xh

    def set_equilibrium_distance_xy(self, equilibrium_distance_xy):
        """set the equilibrium edge-qm --
           edge-mm distance (by MM-force field)"""
        self.equilibrium_distance_xy = equilibrium_distance_xy

    def get_link_atom(self):
        """ get an ase-atom to be the link atom """
        return self.atom

    def get_link_atom_qm_region_index(self):
        """ get to which qm region the link atom belongs to """
        return self.qm_region_index

    def get_link_atom_index_in_qm(self):
        """ get what is my link atom index in this qm region """
        return self.link_atom_index_in_qm

    def get_link_atom_qm_neighbor(self):
        """ get what index does my qm neighbor have"""
        return self.qm_neighbor

    def get_link_atom_mm_neighbor(self):
        """ get what index does my mm neighbor have"""
        return self.mm_neighbor

    def get_link_atom_qm2_neighbors(self):
        """ get what index does my second qm neighbor have"""
        return self.qm2_neighbors

    def get_link_atom_qm3_neighbors(self):
        """ get what index does my third qm neighbor have"""
        return self.qm3_neighbors

    def get_link_atom_mm2_neighbors(self):
        """ get what index does my second mm neighbor have"""
        return self.mm2_neighbors

    def get_force_constant(self):
        """ get the force constant of bond edge-qm -- linkH (not used)"""
        return self.force_constant

    def get_equilibrium_distance_xh(self):
        """ get the equilibrium edge-qm -- linkH distance """
        return self.equilibrium_distance_xh

    def get_equilibrium_distance_xy(self):
        """get the equilibrium edge-qm --
           edge-mm distance (by MM-force field)"""
        return self.equilibrium_distance_xy


class AseQmmmManyqm:
    """ This is a qm/mm interface with qm=FHI-aims, mm=gromacs.
    
    We can have many QM regions, each with a different calculator.
    There can be only one MM calculator, which is calculating the whole
    system.

    Numeration of atoms starts from 0. (in qms, mms)

    In qm calculations link atom(s) come(s) last.

    For any qm region, the optimal bond lengths for all edge_atom-link_atom
    pairs are optimized by QM simultaneously at the beginning of
    the run when the flag link_info='byQM' is used (by method . The positions of other a

    """

    def __init__(self,  nqm_regions,  \
                     qm_calculators, mm_calculator, \
                     link_info='byQM'):
        """ Set initial values to each qm and mm calculator.
        Additionally set information for the qm/mm interface.

        The information about qm and mm indexes is read from
        a file 'index.ndx'
        Which can be generated with a gromacs tool 'make_ndx'
        http://www.gromacs.org/Documentation/Gromacs_Utilities/make_ndx

        Parameters
        ==========
        nqm_regions: int
            how many qm regions

        qm_calculators: list members of a Class defining a Calculator
            ase-qm calculator for each qm region

        mm_calculator: a member of a Class defining a Calculator
             ase-mm calculator for mm (the whole system)

        link_info: str
            can be either
            'byQM': the edge_qm_atom-link_h_atom distances are calculated by QM
            'byFile':the edge_qm_atom-link_h_atom distances are read from a file

        """

        from ase.io import write
        import os, glob

        # clean
        files = glob.glob('test-*')
        for file in files:
            try:
                os.remove(file)
            except OSError:
                pass

        self.atoms = None
        self.positions = None
        self.neighbor_list = None
        self.link_atoms = []

        self.energy = None
        self.e_delta_stretch = None

        self.nqm_regions = nqm_regions
        self.qm_calculators = qm_calculators
        self.mm_calculator = mm_calculator

        self.qmatom_types = []
        self.mmatom_types = []

        #det unique name for each qm region
        # (the output file of each qm calculation)
        #for i in range(len(self.qm_calculators)):
        #    self.qm_calculators[i].set(output_template = 'aims'+str(i))

        self.link_systems = None

        self.equilibrium_distances_xy = []
        self.equilibrium_distances_xh = []
        self.force_constants = []

        # get the sets of qm atoms
        self.qms = get_qm_atoms()
        self.set_qms = set(sum(self.qms, []))
        print('qmsystem(s), indexing from 0:')
        print('')
        for index_out in self.qms:
            index_str = ''
            for index in index_out:
                index_str += str(index) + ' '
            print('%s' % index_str)
            print('')

        if len(self.qms) != nqm_regions:
            print('Number of set of QM atoms does not match with nqm_regions')
            print('self.qms %s' % str(self.qms))
            print('nqm_regions %s' % str(nqm_regions))
            sys.exit()
        if len(self.qms) != len(qm_calculators):
            print('Number of set of QM atoms does not match with')
            print('the number of QM calculators')
            sys.exit()


        #read the actual structure to define link atoms and their neighbors
        system_tmp =  mm_calculator.atoms
        self.positions = system_tmp.get_positions()

        #get neighbor lists
        self.neighbor_list = get_neighbor_list(system_tmp)

        #get the mm-atoms next to link atoms for all qm regions
        (self.mms_edge, self.qms_edge, self.set_mms_edge, self.set_qms_edge) = \
            self.get_edge_qm_and_mm_atoms(self.qms, system_tmp)

        #get the mm atoms being second neighbors to any qm atom
        (self.second_mms, self.set_second_mms) = \
            self.get_next_neighbors(self.mms_edge, self.set_qms)

        #get the qm atoms being second neighbors to link atom
        (self.second_qms, self.set_second_qms) = \
            self.get_next_neighbors(self.qms_edge, \
                                        self.set_mms_edge)

        #get the qm atoms being  neighbors to link atom (edge-qm atoms)
        # and their neighbors which have only single neighbor
        # (for example edge-QM(C)-H or edge-QM(C)=O; for charge exclusion)
        self.constant_charge_qms = \
            self.get_constant_charge_qms\
            (self.set_qms_edge, self.set_second_qms)

        #get the qm atoms being third neighbors to link atom
        (self.third_qms, self.set_third_qms) = \
            self.get_next_neighbors\
            (self.second_qms, self.set_qms_edge)
        print('self.qms %s' % self.qms)
        print('QM edge, MM edge %s'
              % str(self.qms_edge)+' '+ str(self.mms_edge))
        print('MM second N of Link %s' % str(self.second_mms))
        print('QM second N of Link %s' % str(self.second_qms))
        print('QM third N of Link %s' % str(self.third_qms))

        #check that MM topology exists:
        if not(os.path.exists(mm_calculator.topology_filename)):
            print('NO TOPOLOGY FILE:', mm_calculator.topology_filename)
            print('use: CALC_MM.generate_topology_and_g96file()')
            sys.exit()

        #check that MM run file (.tpr) exists:
        if not(os.path.exists(mm_calculator.label+'.tpr')):
            print('NO MM run file FILE:', mm_calculator.label + '.tpr')
            print('use: CALC_MM.write_input(atoms)')
            print('use: CALC_MM.generate_gromacs_run_file()')
            sys.exit()
        #check that force field files exist
        if 'GMXDATA' in os.environ:
            gromacs_home = os.environ['GMXDATA'].split(':')[0]
        else:
            gromacs_home = '/usr/local/gromacs/share/gromacs/'
        ff_filename = gromacs_home+ '/top/' \
            + mm_calculator.force_field + '.ff/ffbonded.itp'
        if not(os.path.exists(mm_calculator.topology_filename)):
            print('NO force field file:', ff_filename)
            print('use: GMXDATA environmental variable')
            sys.exit()

        if link_info == 'byFILE':
            self.read_eq_distances_from_file()
        else:
            
            #get QM-MM bond lengths
            self.get_eq_distances_xy(\
                topfilename=mm_calculator.topology_filename,\
                    force_field= mm_calculator.force_field)

            #get QM-linkH distances by QM for all link atoms
            self.get_eq_qm_atom_link_h_distances(system_tmp)


        # write current link-info data to file (it can be later used,
        # so XH bondconstants are already calculated by QM
        # Also one can manually change the XY bond lengths
        self.write_eq_distances_to_file(\
            self.qms_edge)

        
        #get target charge of each qm-region
        self.classical_target_charge_sums = \
            self.get_classical_target_charge_sums\
            (self.mm_calculator.topology_filename, self.qms)


        #get a list of link H atoms
        self.link_atoms = self.get_link_atoms(\
            self.qms_edge, self.mms_edge,\
                self.force_constants,\
                self.equilibrium_distances_xh, \
                self.equilibrium_distances_xy)

        self.qmsystems = self.define_QM_clusters_in_vacuum(system_tmp)

        for iqm, qm in enumerate(self.qmsystems):
            write('test-qm-'+str(iqm)+'.xyz', qm)
        

        #attach calculators to qm regions
        for iqm, qm in enumerate(self.qmsystems):
            self.qmsystems[iqm].set_calculator(self.qm_calculators[iqm])

        #attach calculators to the mm region (the whole system)
        self.mm_system = system_tmp
        self.mm_system.set_calculator(self.mm_calculator)

        #initialize total energy and forces of qm regions
        #and the mm energy
        self.qm_energies = []
        self.qm_forces = []
        self.qm_charges = []
        self.sum_qm_charge = []
        for iqm, qm in enumerate(self.qmsystems):
            self.qm_energies.append(0.0)
            self.qm_forces.append(None)
            self.qm_charges.append(None)
            self.sum_qm_charge.append(None)
        self.mm_energy = None
        #set initial zero forces
        self.forces = np.zeros((len(self.positions), 3))
        self.charges = np.zeros((len(self.positions), 1))

        try:
            os.remove(self.mm_calculator.topology_filename+'.orig')
        except:
            pass
        print('%s' % str(self.mm_calculator.topology_filename))
        os.system('cp ' + self.mm_calculator.topology_filename + ' ' +\
                      self.mm_calculator.topology_filename + '.orig')

        #remove some classical bonded interaction in the topology file
        # this need to be done only once, because the bond topology
        # is unchanged during a QM/MM run
        #(QM charges can be updated in the topology, however)

        # the original topology is generated when calling Gromacs(
        # in the main script setting up QM, MM and minimization

        if (self.mm_calculator.name == 'Gromacs'):
            self.kill_top_lines_containing_only_qm_atoms\
                (self.mm_calculator.topology_filename, self.qms, \
                     self.mm_calculator.topology_filename)
        else:
            print('Only Gromacs MM-calculator implemented in ASE-QM/MM')
            sys.exit()
        #exclude qm-qm non-bonded interactions in MM-gromacs
        self.add_exclusions()
        #generate input file for gromacs run
        self.mm_calculator.generate_gromacs_run_file()

        ######### end of Init #####################################

    def get_forces(self, atoms):
        """get forces acting on all atoms except link atoms """
        self.update(atoms)
        return self.forces

    def get_potential_energy(self, atoms):
        """ get the total energy of the MM and QM system(s) """
        self.update(atoms)
        return self.energy

    def update(self, atoms):
        """Updates and does a check to see if a calculation is required"""



        if self.calculation_required(atoms):
            # performs an update of the atoms and qm systems
            self.atoms = atoms.copy()
            self.positions = atoms.get_positions()
            self.mm_system = atoms.copy()

            #get the positions of link H atoms
            self.link_atoms = self.get_link_atoms(\
                self.qms_edge, self.mms_edge,\
                    self.force_constants,\
                    self.equilibrium_distances_xh, \
                    self.equilibrium_distances_xy)

            #get QM systens
            self.qmsystems = self.define_QM_clusters_in_vacuum(\
                self.atoms)
            self.calculate(atoms)

    def calculation_required(self, atoms):
        """Checks if a calculation is required"""
        if ((self.positions is None) or
            (self.atoms != atoms) or
            (self.energy is None)):
            return True
        return False


    def calculate_mm(self):
        """ Calculating mm energies and forces """
        import os
        mm = self.atoms
        mm.set_calculator(self.mm_calculator)
        if (self.mm_calculator.name == 'Gromacs'):
            try:
                os.remove(self.mm_calculator.label+'.log')
            except:
                pass
        self.mm_calculator.update(mm)
        #self.mm_calculator.run()
        #self.mm_calculator.calculate(atoms=mm, properties=['energy', 'forces'])
                                     
        self.mm_energy = 0
        
        self.mm_energy += mm.get_potential_energy()
        self.forces += mm.get_forces()

    def calculate_qms(self):
        """ QM calculations on all qm systems are carried out """
        for iqm, qm in enumerate(self.qmsystems):
            qm.set_calculator(self.qm_calculators[iqm])
            self.qm_energies[iqm] = qm.get_potential_energy()
            self.qm_forces[iqm] = np.zeros((len(qm), 3))
            self.qm_forces[iqm] = qm.get_forces()
            (self.sum_qm_charge[iqm], self.qm_charges[iqm]) = \
                self.get_qm_charges(iqm,
                                    number_of_link_atoms =\
                                        len(self.qms_edge[iqm]))
            if (len(self.qms[iqm]) != len(self.qm_charges[iqm])):
                print('Problem in reading charges')
                print('len(self.qms[iqm]) %s' % str(len(self.qms[iqm])))
                print('len(self.qm_charges[iqm]) %s'
                      % str(len(self.qm_charges[iqm])))
                print('Check the output of QM program')
                print('iqm, qm %s' % str(iqm)+ ' '+ str(qm))
                print('self.qm_charges[iqm] %s' % str(self.qm_charges[iqm]))
                sys.exit()

    def calculate_single_qm(self, myqm, mycalculator):
        """ Calculate the qm energy of a single qm region
        (for X-H bond length calculations)
        """
        myqm.set_calculator(mycalculator)
        return myqm.get_potential_energy()

    def run(self, atoms):
        """Runs QMs and MM"""

        self.forces = np.zeros((len(atoms), 3))

        self.calculate_qms()

        # update QM charges to MM topology file
        self.set_qm_charges_to_mm_topology()

        #generate gromacs run file (.tpr) base on new topology
        self.mm_calculator.generate_gromacs_run_file()
        self.calculate_mm()
                        

    def calculate(self, atoms):
        """gets all energies and forces (qm, mm, qm-mm and corrections)"""

        self.run(atoms)

        self.energy = sum(self.qm_energies)+self.mm_energy
        
        #map the forces of QM systems to all atoms
        #loop over qm regions
        for qm, qm_force in zip(self.qms, self.qm_forces):
            #loop over qm atoms in a qm region
            #set forces to the all-atom set (the all atom set does not
            # have link atoms)
            for iqm_atom, qm_atom in enumerate(qm):
                self.forces[qm_atom] = self.forces[qm_atom] + \
                    qm_force[iqm_atom]
        self.get_link_atom_forces(action = 'QM')


    def get_link_atoms(self, qm_links, mm_links, \
                        force_constants,\
                        equilibrium_distances_xh, equilibrium_distances_xy):
        """
        QM atoms can be bonded to MM atoms. In this case one sets
        an extra H atom (a link atom).

        The positions of the all link H atoms in all qm regions are
        set along QM-MM and bond with length defined by:

        J. Chem. Theory Comput 2011, 7, 761-777, Eq 1
        r_XH = r_XY_current*(r_XH_from_qm_calculation /r_XY_from_forceField)

        """
        import math
        from ase import Atom

        link_hs = []

        for i_qm_region, (qm0, mm0) in enumerate (zip( qm_links, mm_links)):
            for i_link_atom, (qmatom, mmatom) in enumerate (zip(qm0, mm0)):
                dx = (self.positions[mmatom, 0] - self.positions[qmatom, 0])
                dy = (self.positions[mmatom, 1] - self.positions[qmatom, 1])
                dz = (self.positions[mmatom, 2] - self.positions[qmatom, 2])
                d = math.sqrt(dx* dx+ dy* dy+ dz* dz)
                
                unit_x = dx/ d
                unit_y = dy/ d
                unit_z = dz/ d
                xh_bond_length = \
                    d*\
                    self.equilibrium_distances_xh[i_qm_region][i_link_atom]/\
                    self.equilibrium_distances_xy[i_qm_region][i_link_atom]
            
                posh_x = self.positions[qmatom, 0] + unit_x* xh_bond_length
                posh_y = self.positions[qmatom, 1] + unit_y* xh_bond_length
                posh_z = self.positions[qmatom, 2] + unit_z* xh_bond_length
                tmp_link_h = (Atom('H', position=(posh_x, posh_y, posh_z)))
                link_h = LinkAtom(atom=tmp_link_h, \
                                      qm_region_index = i_qm_region,\
                                      link_atom_index = i_link_atom)
                link_h.set_link_atom_qm_neighbor(qmatom)
                link_h.set_link_atom_mm_neighbor(mmatom)
                link_h.set_force_constant(\
                    force_constants[i_qm_region][i_link_atom])
                link_h.set_equilibrium_distance_xh(equilibrium_distances_xh\
                                      [i_qm_region][i_link_atom])
                link_h.set_equilibrium_distance_xy(equilibrium_distances_xy\
                                      [i_qm_region][i_link_atom])


                link_hs.append(link_h)
        return (link_hs)



    def get_link_atom_forces(self, action):
        """ Add forces due to link atom to QM atom
        and to MM atom next to each link atom.

        Top Curr Chem (2007) 268: 173-290
        QM/MM Methods for Biological Systems
        Hans Martin Senn and Walter Thiel

        Eqs. 10(p192), 12(p193), 16a, 16b(p 194)

        """

        for link_atom in self.link_atoms:
            i_qm_atom = link_atom.qm_neighbor
            i_mm_atom = link_atom.mm_neighbor
            i_qm_region = link_atom.qm_region_index
            link_atom_index_in_qm = link_atom.get_link_atom_index_in_qm()

            if (action == 'QM'):
                force_of_h = self.qm_forces[i_qm_region][link_atom_index_in_qm]
            elif (action == 'MM'):
                force_of_h = link_atom.mm_force
            else:
                print('not implemented in get_link_atom_forces')
                sys.exit()

            g = link_atom.equilibrium_distance_xh/\
                link_atom.equilibrium_distance_xy
            
            self.forces[i_mm_atom, 0]  = self.forces[i_mm_atom, 0] +\
                force_of_h[0] * g
            self.forces[i_mm_atom, 1]  = self.forces[i_mm_atom, 1] +\
                force_of_h[1] * g
            self.forces[i_mm_atom, 2]  = self.forces[i_mm_atom, 2] +\
                force_of_h[2] * g

            self.forces[i_qm_atom, 0]  = self.forces[i_qm_atom, 0] +\
                force_of_h[0] * (1.0 - g)
            self.forces[i_qm_atom, 1]  = self.forces[i_qm_atom, 1] +\
                force_of_h[1] * (1.0 - g)
            self.forces[i_qm_atom, 2]  = self.forces[i_qm_atom, 2] +\
                force_of_h[2] * (1.0 - g)





    def add_energy_exclusion_group(self, indexfilename='index.ndx'):
        """
        Add energy exclusions for MM calculations.
        This is the way to block non-bonded MM (coulomb&vdW)
        interactions within a single QM region.
        """
        
        infile = open(indexfilename,'r')
        lines = infile.readlines()
        infile.close()
        
        qm_region_names = []

        for line in lines:
            if (('QM' in line) or ('Qm' in line) or ('qm' in line)):
                qm_region_names.append(line.split()[1])

        infile = open(self.mm_calculator.label+'.mdp','r')
        lines = infile.readlines()
        infile.close()
        outfile = open(self.mm_calculator.label+'.mdp','w')
        for line1 in lines:
            outfile.write(line1)
        outfile.write(';qm regions should not MM-interact with themselves \n')
        outfile.write(';but separate qm regions MM-interact with each other \n')

        outfile.write('energygrps = ')
        for name in qm_region_names:
            outfile.write(name + ' ')
        outfile.write('\n')
        outfile.write('energygrp_excl = ')
        for name in qm_region_names:
            outfile.write(name + ' ' + name + ' ')
        outfile.write('\n')
        outfile.close()

        return

    def add_exclusions(self):
        """
        Add energy exclusions for MM calculations.
        This is the way to block non-bonded MM (coulomb&vdW)
        interactions within a single QM region.
        """
        

        infile = open(self.mm_calculator.topology_filename,'r')
        lines = infile.readlines()
        infile.close()
        outfile = open(self.mm_calculator.topology_filename,'w')
        for line in lines:
            if '[ angle' in line:
                outfile.write('\n')
                outfile.write('[ exclusions ] \n')
                outfile.write(\
                    '; qm regions should not MM-interact with themselves \n')
                outfile.write(\
                    '; but separate qm regions MM-interact with each other \n')

                for qm_region in self.qms:
                    for qm_atom1 in qm_region:
                        outfile.write(str(qm_atom1 + 1) + ' ')
                        for qm_atom2 in qm_region:
                            if qm_atom1 != qm_atom2:
                                outfile.write(str(qm_atom2 + 1) + ' ')
                        outfile.write('\n')
                outfile.write('\n')
            outfile.write(line)



        outfile.close()

        return

    def get_qm_charges(self, i_current_qm, calculator='Aims',
                       number_of_link_atoms = 0):
        """
        Get partial charges on QM atoms.
        The charges at link atoms are not returned.
        """
        if calculator == 'Aims':
            infile = open('aims'+str(i_current_qm)+'.out','r')
            lines = infile.readlines()
            infile.close()
            qm_charges = []
            for line in lines:
                if ('Hirshfeld charge      ' in line):
                    qm_charges.append(float(line.split()[4]))
            sum_qm_charges = sum(qm_charges)
            #delete charges of link atoms
            if (number_of_link_atoms > 0):
                del qm_charges[-number_of_link_atoms:]
                    
        return sum_qm_charges, qm_charges

    def get_topology_lines(self, lines):
        """ Get lines including charges of atoms (ok_lines)
        also comments in these lines (comment_lines)
        and lines before and after these lines
        (lines_before and lines_after)
        """
        lines_before = []
        lines_change = []
        lines_after = []
        do_lines_before = True
        do_lines_change = False
        for line in lines:
            if (' bonds ') in line:
                do_lines_change = False
            if do_lines_before:
                lines_before.append(line)
            elif do_lines_change:
                lines_change.append(line)
            else:
                lines_after.append(line)
            
            if (' atoms ') in line:
                do_lines_before = False
                do_lines_change = True

        #kill comments and empty lines,
        #get the charge in the topology file
        comment_lines = []
        lines_ok = []
        for iline in range(len(lines_change)):
            if lines_change[iline].startswith(';'):
                comment_lines.append(lines_change[iline])
            elif not lines_change[iline].strip():
                pass
            else:
                try:
                    #new charge = float(lines_change[iline].split()[6])
                    #new charge_orig = charge_orig + charge
                    #top_charge.append(charge)
                    lines_ok.append(lines_change[iline])
                except:
                    print('error in reading gromacs topology')
                    print('line is')
                    print('%s' % lines_change[iline])
                    sys.exit()
        return lines_before, comment_lines, lines_ok, lines_after


    def set_qm_charges_to_mm_topology(self):
        """ Set qm charges to qm atoms of MM topology based on
        a QM calculation.
        1) The charges of link atoms are neglected.
        2) The charge of a qm atom next to the link atom is set to be the
        same value as in the original topology file. (trying to
        avoid the artificial polarization due to qmAtom-linkH).
        3) the total charge of the system (all QM and MM atoms) should be
        the same as in the original classical system. Therefore, all the
        QM atoms will gain/loose an equal amount of charge in the MM topology
        file.
        """

        infile = open(self.mm_calculator.topology_filename,'r')
        lines = infile.readlines()
        infile.close()

        (lines_before, comment_lines, lines_ok, lines_after) = \
            self.get_topology_lines(lines)

        #check that the atom numering is ok
        for iline in range(len(lines_ok)):
            atom_nr = iline + 1
            if int(lines_ok[iline].split()[0]) != atom_nr:
                print('2: error in reading gromacs topology')
                print('line is')
                print('%s' % lines_ok[iline])
                sys.exit()
        
        # get the total charge of non-link H atoms in the current qm system
        # The charges of edge atoms and their H neighbors
        # are taken from topology
        # (they are unchanged, it is not from QM calculations)
        for iqm, qm in enumerate(self.qms):
            charges = self.qm_charges[iqm]
            charges_ok = charges
            qm_charge_no_link_edge_mm = 0.0
            n_qm_charge_atoms = 0

            for qm_atom, charge in zip(qm, charges):
                if qm_atom not in self.constant_charge_qms:
                    qm_charge_no_link_edge_mm = \
                    qm_charge_no_link_edge_mm + charge
                    n_qm_charge_atoms = n_qm_charge_atoms + 1

            # correct the total charge to be equal the original one
            # in the topology file by
            # adding/ subtracting missing/extra charge on
            # non-edge and non-single neighbor next neib QM atoms
            change_charge = \
                ( self.classical_target_charge_sums[iqm] - \
                     qm_charge_no_link_edge_mm)/\
                     float(n_qm_charge_atoms)
            for iqmatom, qmatom in enumerate(qm):
                if qmatom not in self.constant_charge_qms:
                    charges_ok[iqmatom] = charges[iqmatom] + change_charge
            # set qm charges to the lines of gromacs topology file
            for iqmatom, qmatom in enumerate(qm):
                if qmatom not in self.constant_charge_qms:
                    lines_ok[qmatom] = \
                        lines_ok[qmatom][0:45]\
                        +str(round((charges_ok[iqmatom]),5)).rjust(11)+\
                        lines_ok[qmatom][56:70]
        # write out the new topology file
        sum_charge = 0.0
        for iline in range(len(lines_ok)):
            sum_charge = sum_charge + float(lines_ok[iline][46:56])
            comment = '; qtot '+str(round(sum_charge,4))+'\n'.ljust(12)
        
        outfile = open(self.mm_calculator.topology_filename, 'w')
        for line in lines_before:
            outfile.write(line)
        for line in comment_lines:
            outfile.write(line)
        sum_charge = 0.0
        for line in lines_ok:
            sum_charge = sum_charge + float(line[46:56])
            comment = '; qtot '+str(round(sum_charge,4)).ljust(11)+'\n'
            outfile.write(line[0:70]+comment)
        outfile.write('\n')
        for line in lines_after:
            outfile.write(line)
        outfile.close()


#------------------------------------------------------------------
#------Below the stuff needed for initializing the QM/MM system ---
#------Setting up link atoms, defining QM and MM regions ----------
#------------------------------------------------------------------




    def get_edge_qm_and_mm_atoms(self, qms, system):
        """  Get neighbors of QM atoms (MM-link-atoms) that are not in QM
        (there may be many QM regions)
        edge-QM atom can NOT be neighbored by H atom(s)
        also get edge-QM atoms
        """


        masses = system.get_masses()

        mms1 = []
        qms1 = []
        setmms1 = set([])
        setqms1 = set([])
        for qm in qms:

            link_mm_atoms = []
            link_qm_atoms = []
            for qm_atom in qm:
                indices, offsets = self.neighbor_list.get_neighbors(qm_atom)
                for neib_atom in indices:
                    if neib_atom not in qm:
                        link_mm_atoms.append(neib_atom)
            #take unique atoms of flattened list
            link_mm_atoms = list(set(link_mm_atoms))
            # Kill MM atoms that are H atoms in the neighborlist
            oklink_mm_atoms = []
            for index in link_mm_atoms:
                if masses[index] > 1.5:
                    oklink_mm_atoms.append(index)
                else:
                    print('WARNING:')
                    print('qm system cannot be bond to H atoms')
                    print('problem atom index is (numbering from 1): %s' %
                          str(index+1))
                    print('if this is water H you should consider including it')
                    print('in QM')
                    #sys.exit()


            #get indexes of QM edge atoms,
            # one qm atom can be more then one time an edge atom
            # (then this QM atom will have more than one link atoms)
            for link_mm_atom in oklink_mm_atoms:
                indices, offsets = \
                    self.neighbor_list.get_neighbors(link_mm_atom)
                for neib_atom in indices:
                    if neib_atom in qm:
                        link_qm_atoms.append(neib_atom)


            mms1.append(oklink_mm_atoms)
            qms1.append(link_qm_atoms)
            setmms1 |= set(oklink_mm_atoms)
            setqms1 |= set(link_qm_atoms)
        return mms1, qms1, setmms1, setqms1

    def get_next_neighbors(self, atom_indexes, prohibited_set):
        """  Get neighbors of all atoms in 'atom_indexes'
        that are not in 'prohibited_set'.

        'atom_indexes' is a list of list in which atom indexes belonging
        of each QM region is a separate list, that is
        [[QM1 atom_indexes], [QM2 atom_indexes], ...]
        """

        list_neibs = []
        set_list_neibs = set([])
        for current_atoms in atom_indexes:
            neibs = []
            set_current_atoms = set(current_atoms)
            for current_atom in current_atoms:
                indices, offsets = \
                    self.neighbor_list.get_neighbors(current_atom)
                setneib = set(indices)
                neibs += list(setneib - set_current_atoms-prohibited_set)
            list_neibs.append(neibs)
            set_list_neibs |= set(neibs)
        return list_neibs, set_list_neibs

    def get_constant_charge_qms(self, set_qms_edge, set_second_qms):
        """ get indices of all qm atoms whose charge in MM
        calculations is taken from the original MM-topology
        (not from the QM calculation). These atoms are edge QM atoms
        and their neighbors in QM which have only one neighbor.
        At least C(edge-qm)-H(second-edge-qm) and C(edge-qm)=O(second-edge-qm)
        """

        set_charge_exclusion = set_qms_edge

        for second_qms in set_second_qms:
            indices, offsets = self.neighbor_list.get_neighbors(second_qms)
            if len(indices)== 1:
                set_charge_exclusion.add(second_qms)
        return set_charge_exclusion


    def get_eq_distances_xy(\
        self, topfilename = 'gromos.top', force_field = 'oplsaa'):
        """
        The link atom is positioned as in
        J. Chem. Theory Comput 2011, 7, 761-777, Eq 1

        For this purpose we need the equilibrium length of each
        QM-MM covalent bond. Those are obtained here from the
        files of the force field.

        """

        import os

        print('in get_eq_distances_xy, topfilename=')
        print('%s' % topfilename)
        for qm in self.qms_edge:
            equilibrium_distance_xy = []
            for iqm in qm:
                equilibrium_distance_xy.append(0.0)
            self.equilibrium_distances_xy.append(equilibrium_distance_xy)

        #get the version of the topology file where one sees the bond
        # force constants (file is named as gromacs.top.dump)
        try:
            os.remove(self.mm_calculator.label+'.tpr.dump')
        except OSError:
            pass

        os.system('gmxdump -s '+ self.mm_calculator.label\
                      +'.tpr > ' + \
                  self.mm_calculator.label+ \
                      '.tpr.dump 2>/dev/null')
        if 'GMXDATA' in os.environ:
            gromacs_home = os.environ['GMXDATA'].split(':')[0]
        else:
            gromacs_home = '/usr/local/gromacs/share/gromacs/'

        #read the bonded force constants of this force field in order to
        #get an estimate for X-Y bond constant
        linesff = open(gromacs_home+ '/top/'+ force_field+ \
                           '.ff/ffbonded.itp', 'r').readlines()
        oklinesff = []
        start = False

        for line in linesff:
            if 'bondtypes' in line:
                start = True
            elif '[' in line:
                break
            if start and (line.strip()):
                oklinesff.append(line)

        #lines for getting oplsaa atom dual-types
        if 'opls' in force_field:
            lines_for_dual_types = open(gromacs_home+ '/top/'+ force_field+ \
                                         '.ff/ffnonbonded.itp', 'r').readlines()

        #read the types of interaction for bond stretching
        lines_tpr = open(self.mm_calculator.label+\
                             '.tpr.dump', 'r').readlines()

        #read the topology file to get QM atom type
        lines_top = open(topfilename, 'r').readlines()
        oklines_top = []
        start = False
        for line in lines_top:
            if start and ('[' in line):
                break
            if start:
                if (not line.startswith(';')) or (not line.strip()):
                    oklines_top.append(line)
            if '[ atoms' in line:
                start = True


        #get force constant and bond eq distance for all QM-MM bonds
        #
        ok_equilibrium_distances_xy = []
        ok_qmatom_types = []
        ok_mmatom_types = []
        for qm0, mm0, eqsxy in zip(
            self.qms_edge, self.mms_edge, \
                self.equilibrium_distances_xy):
            ok_eqxy = []
            ok_qmatom_type = []
            ok_mmatom_type = []
            for qmatom, mmatom, eqxy in \
                    zip(qm0, mm0, eqsxy):
                #find qm-mm bond in topology file (indexes from 0)
                # get the index for interaction
                interaction = 'empty'
                for line in lines_tpr:
                    if (' type' in line) and ('BONDS' in line):
                        if (qmatom == int(line.split()[3])) and \
                                (mmatom == int(line.split()[4])):
                            interaction = line.split()[1].lstrip('type=')
                            break
                        if (qmatom == int(line.split()[4])) and \
                                (mmatom == int(line.split()[3])):
                            interaction = line.split()[1].lstrip('type=')
                            break
                if interaction == 'empty':
                    print('QM-MM bond not found in topology')
                    print('atoms are: QM, MM: (from 1 indexing) %s' %
                          str(qmatom + 1) + '  ' + str(mmatom + 1))
                    sys.exit()
                for line in lines_tpr:
                    if ('functype['+interaction+']=BONDS') in line:
                        r_xy0 = float(line.split()[2].rstrip(','))
                #get type of the QM atom
                qmatom_type = 'empty'
                for line in oklines_top:
                    if (int(line.split()[0] ) == qmatom+ 1):
                        qmatom_type = line.split()[1]
                        #oplsaa atom type has a double name,
                        #the other one is used in file ffbonded.itp
                        break
                if (qmatom_type == 'empty'):
                    print('problem in QM atom type')
                    sys.exit()

                if 'opls' in force_field:
                    found = False
                    for line in lines_for_dual_types:
                        if (qmatom_type == line.split()[0]):
                            qmatom_type = line.split()[1]
                            found = True
                            break
                    if not found:
                        print('problem in QM atom type')
                        print('with OPLSAA force field dual atom types')
                        sys.exit()
                #get type of the true link-MM atom
                mmatom_type = 'empty'
                for line in oklines_top:
                    if (int(line.split()[0] ) == mmatom+ 1):
                        mmatom_type = line.split()[1]
                        #oplsaa atom type has a double name,
                        #the other one is used in file ffbonded.itp
                        break
                if (mmatom_type == 'empty'):
                    print('problem in MM atom type')
                    sys.exit()

                if 'opls' in force_field:
                    found = False
                    for line in lines_for_dual_types:
                        if (mmatom_type == line.split()[0]):
                            mmatom_type = line.split()[1]
                            found = True
                            break
                    if not found:
                        print('problem in MM atom type')
                        print('with OPLSAA force field dual atom types')
                        sys.exit()

                ok_qmatom_type.append(qmatom_type)
                ok_mmatom_type.append(mmatom_type)

                if (eqxy != 0.0):
                    #use eq constant given by the user
                    ok_eqxy.append(eqxy)
                else:
                    ok_eqxy.append(r_xy0)
            ok_equilibrium_distances_xy.append(ok_eqxy)
            ok_qmatom_types.append(ok_qmatom_type)
            ok_mmatom_types.append(ok_mmatom_type)
        outfile = open('qm-mm-linkAtomsInfo.txt','w')
        outfile.write(\
            '=======================================================\n')
        outfile.write('Information about QM-MM boundary(ies) \n')
        outfile.write(\
            'Created using the Atomic Simulation Environment (ASE) \n')
        outfile.write(\
            '=======================================================\n')
        qmregion_count = 0

        # ADD qm-mm-linkAtomsInfo.txt
        for qm, mm, eqs_xy, eqs_xh, qmtypes, mmtypes in zip\
                (self.qms_edge, self.mms_edge, ok_equilibrium_distances_xy,\
                     self.equilibrium_distances_xh,\
                     ok_qmatom_types, ok_mmatom_types):
            outfile.write(\
                '=======================================================\n')
            qmregion_count = qmregion_count+ 1
            outfile.write('Parameters related to QM region number '+\
                             str(qmregion_count)+'\n')
            for qmatom, mmatom,  eq_xy, eq_xh, qmtype, mmtype in zip\
                    (qm, mm, eqs_xy, eqs_xh,\
                    qmtypes, mmtypes):
                outfile.write('qm-link-atom-index (from 1): '+str(qmatom)+'\n')
                outfile.write('qm-link-atom-type: '+str(qmtype)+'\n')
                outfile.write('mm-link-atom-index (from 1): '+str(mmatom)+'\n')
                outfile.write('mm-link-atom-type: '+str(mmtype)+'\n')
                outfile.write('qm-mm(notH)-equilibrium-distance: '\
                                  +str(eq_xy)+' nm\n')
                outfile.write('qm-H-equilibrium-distance(calculated by QM): '\
                                  +str(eq_xh)+' nm\n')
        outfile.close()
        self.equilibrium_distances_xy = ok_equilibrium_distances_xy
        self.qmatom_types = ok_qmatom_types
        self.mmatom_types = ok_mmatom_types

        return

    def write_eq_distances_to_file(
        self,
        qm_links, filename='linkDATAout.txt'):
        """
        Write classical bond equilibrium lengths
        for XY (X in QM, Y in MM)
        Write QM calculated XH(link atom) bond length (X in QM, H link atom)
        """

        outfile = open(filename, 'w')
        for iqm_region, qmlink in enumerate (qm_links):
            for ilink, dummy in enumerate (qmlink):
                data = self.equilibrium_distances_xy[iqm_region][ilink]
                outfile.write(str(data)+' ')
                outfile.write('\n')
                data = self.equilibrium_distances_xh[iqm_region][ilink]
                outfile.write(str(data)+' ')
                outfile.write('\n')
                data = self.force_constants[iqm_region][ilink]
                outfile.write(str(data)+' ')
                outfile.write('\n')
                data = self.qmatom_types[iqm_region][ilink]
                outfile.write(str(data)+' ')
                outfile.write('\n')
                data = self.mmatom_types[iqm_region][ilink]
                outfile.write(str(data)+' ')
                outfile.write('\n')
        outfile.close()
        return


    def read_eq_distances_from_file(self, filename='linkDATAin.txt'):
        """
        Read classical bond equilibrium lengths
        for XY (X in QM, Y in MM) or XH (X in QM, H link atom)
        """

        myfile = open(filename, 'r')
        self.equilibrium_distances_xy = []
        self.equilibrium_distances_xh = []
        self.force_constants = []
        self.qmatom_types = []
        self.mmatom_types = []
        print('Reading X-H and other data from file: %s' % filename)

        for qm in self.qms_edge:
            equilibrium_distance_xy = []
            equilibrium_distance_xh = []
            force_constant = []
            qmatom_type = []
            mmatom_type = []
            for iqm, dum in enumerate(qm):
                line = myfile.readline()
                equilibrium_distance_xy.append(float(line.split()[0]))
                line = myfile.readline()
                equilibrium_distance_xh.append(float(line.split()[0]))
                line = myfile.readline()
                force_constant.append(float(line.split()[0]))
                line = myfile.readline()
                qmatom_type.append(line.split()[0])
                line = myfile.readline()
                mmatom_type.append(line.split()[0])

            self.equilibrium_distances_xy.append(equilibrium_distance_xy)
            self.equilibrium_distances_xh.append(equilibrium_distance_xh)
            self.force_constants.append(force_constant)
            self.qmatom_types.append(qmatom_type)
            self.mmatom_types.append(mmatom_type)
        myfile.close()
        return

    def get_eq_qm_atom_link_h_distances(self, system_tmp):
        """ get equilibrium QMatom-linkH distances
        for all linkH:s
        by  QM  """


        #import matplotlib
        #matplotlib.use('Agg')
        #import matplotlib.pyplot as plt

        from scipy.optimize import fmin

        def qm_bond_energy_function(x, system_tmp, i_qm_region):
            """ get the qm energy of a single qm system with a given
            edge-qm-atom---link-h-atom distances of that qm region
            The qm region is i_qm_region, all
            edge-qm-atom---link-h-atom distance in this qm_region are
            optimized simultaneously
            """

            for index_x, current_x in enumerate(x):
                self.equilibrium_distances_xh\
                    [i_qm_region][index_x] = current_x
            print('current X-H bond lengths [nm]')
            print('%s' % str(x))
            self.link_atoms = self.get_link_atoms(\
                self.qms_edge, self.mms_edge,\
                    self.force_constants,\
                    self.equilibrium_distances_xh, \
                    self.equilibrium_distances_xy)
            self.qmsystems = \
                self.define_QM_clusters_in_vacuum(system_tmp)
            
            #try:
            single_qm_energy = self.calculate_single_qm(\
                self.qmsystems[i_qm_region],\
                    self.qm_calculators[i_qm_region])
            #except RuntimeError:
            #    single_qm_energy = BIG_VALUE
            return single_qm_energy

        print('=====================================================')
        print('Calculating X-H bond lengths and bond force constants')
        print('by QM in one shot for each QM region.')
        print('In later calculations you can: ')
        print('cp linkDATAout.txt linkDATAin.txt')
        print("and set link_info = 'byFILE'")
        print('=====================================================')


        self.equilibrium_distances_xh = []
        self.force_constants = []
        for qm_edges in self.qms_edge:
            force_constants = []
            equilibrium_distances_xh = []
            for qm_edge in qm_edges:
                force_constants.append(0.0)
                equilibrium_distances_xh.append(0.11)
            self.force_constants.append(force_constants)
            self.equilibrium_distances_xh.append(equilibrium_distances_xh)

        #loop over qm regions. To get optimal simultaneous
        # edgeQMatom-linkH distance(s) in [nm] in that qm region
        for i_qm_region in range(len(self.qms_edge)):
            print('NOW running : ')
            print('QM region for optimising edge-linkH distances %s' %
                  str(i_qm_region))
            x = self.equilibrium_distances_xh[i_qm_region][:]
            xopt = fmin(qm_bond_energy_function, \
                            x,\
                            args=(system_tmp, i_qm_region),\
                            xtol=0.0001, ftol=0.0001)
            for index_xopt, current_xopt in enumerate(xopt):
                self.equilibrium_distances_xh\
                    [i_qm_region][index_xopt] = current_xopt
                print('i_qm_region, i_link_atom, optimal X-H bond[nm] %s' \
                      % (str(i_qm_region) + ' ' + str(index_xopt) \
                         + ' ' + str(current_xopt)))

    def define_QM_clusters_in_vacuum(self, system):
        """ Returns Each QM system as an Atoms object
        We get a list of these Atoms objects
        (in case we have many QM regions).
        """
        from ase import Atoms

        qmsystems = []
        for qm0 in self.qms:
            tmp_system = Atoms()
            for qmatom in qm0:
                tmp_system += system[qmatom]
            qmsystems.append(tmp_system)
        for link_atom in self.link_atoms:
            tmp_atom = link_atom.get_link_atom()
            qm_region = link_atom.get_link_atom_qm_region_index()
            link_atom_index_in_qm = len(qmsystems[qm_region])
            qmsystems[qm_region].append(tmp_atom)
            link_atom.set_link_atom_index_in_qm(link_atom_index_in_qm)
        return qmsystems



    def kill_top_lines_containing_only_qm_atoms(self, \
                                              intopfilename, \
                                              qms, outtopfilename):
        """
            Delete all lines in the topology file that contain only qm atoms
            in bonded sections
            (bonds, angles or dihedrals)
            and in pairs section (1-4 interactions)
            """

        # get an index of all qm atoms in all qm regions
        qm = set()
        for qm_tmp in qms:
            qm = qm.union(set(qm_tmp))

        infile = open(intopfilename,'r')
        lines = infile.readlines()
        infile.close()
        outfile = sys.stdout
        oklines = []

        accept = True
        check = ''
        for line in lines:
            if (('[ bonds' in line)):
                oklines.append(line)
                accept = False
                check = 'bond'
            elif (('[ angles' in line)):
                oklines.append(line)
                accept = False
                check = 'angle'
            elif (('[ dihedrals' in line)):
                oklines.append(line)
                accept = False
                check = 'dihedral'
            elif (('[ pairs' in line)):
                oklines.append(line)
                accept = False
                check = 'pair'
            elif ('[' in line):
                oklines.append(line)
                accept = True
                check = ''
            elif line in ['\n']:
                oklines.append(line)
                accept = True
                check = ''
            elif accept:
                oklines.append(line)
            else:
                indexes = [int(float(s)-1.0) \
                               for s in line.split() if s.isdigit()]
                indexes1 = [int(s) for s in line.split() if s.isdigit()]
                if indexes == []:# this takes comment line
                                 #after bond, angle, dihedral
                    oklines.append(line)
                elif check == 'bond':
                    bondedatoms = set(indexes[0:2])
                    #set empty bond intereaction for qm-qm bonds (type 5)
                    #(this way LJ and electrostatics is not messed up)
                    if (bondedatoms.issubset(qm)):
                        newline = str(indexes1[0]).rjust(8)+\
                            str(indexes1[1]).rjust(8)+\
                            ('5').rjust(8) + '\n'
                        oklines.append(newline)
                    else:
                        oklines.append(line)
                elif check == 'angle':
                    bondedatoms = set(indexes[0:3])
                    if (bondedatoms.issubset(qm)):
                        pass
                    else:
                        oklines.append(line)
                elif check == 'dihedral':
                    bondedatoms = set(indexes[0:4])
                    if (bondedatoms.issubset(qm)):
                        pass
                    else:
                        oklines.append(line)
                elif check == 'pair':
                    bondedatoms = set(indexes[0:2])
                    if (bondedatoms.issubset(qm)):
                        pass
                    else:
                        oklines.append(line)
        outfile = open(outtopfilename,'w')
        for line in oklines:
            outfile.write(line)
        outfile.close()

        return


    def get_classical_target_charge_sums(self, intopfilename, qms):
        """ get sum of MM charges of the charged changed by QM
        these are qm atoms that are not link-atoms or edge-qm atoms

        xxx this has a problem:
        Water is in .itp files, not in topology...
        """
        infile = open(intopfilename,'r')
        lines = infile.readlines()
        infile.close()

        (lines_before, comment_lines, ok_lines, lines_after) = \
            self.get_topology_lines(lines)


        classical_target_charge_sums = []
        for iqm, qm in enumerate(qms):
            classical_target_charge_sum = 0.0
            for line in ok_lines:
                atom_index = int(line.split()[0])-1
                if (atom_index in qm) and \
                        (not(atom_index in self.constant_charge_qms)):
                    classical_target_charge_sum = \
                        classical_target_charge_sum + \
                        float(line.split()[6])
            classical_target_charge_sums.\
                append(classical_target_charge_sum)
        return classical_target_charge_sums
