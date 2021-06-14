"""
The ASE Calculator for OpenMX <http://www.openmx-square.org>: Python interface
to the software package for nano-scale material simulations based on density
functional theories.
    Copyright (C) 2017 Charles Thomas Johnson, JaeHwan Shim and JaeJun Yu

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
import numpy as np
import os
import subprocess
import warnings

from ase.calculators.openmx.reader import rn as read_nth_to_last_value


def input_command(calc, executable_name, input_files, argument_format='%s'):
    input_files = tuple(input_files)
    command = executable_name + ' ' + argument_format % input_files
    olddir = os.getcwd()
    try:
        os.chdir(calc.directory)
        error_code = subprocess.call(command, shell=True)
    finally:
        os.chdir(olddir)
    if error_code:
        raise RuntimeError('%s returned an error: %d' %
                           (executable_name, error_code))


class DOS:

    def __init__(self, calc):
        self.calc = calc
        self.dos_dict = {}

    def read_dos(self, method='Tetrahedron', pdos=False, atom_index=1,
                 orbital='', spin_polarization=False):
        """
        function for reading DOS from the following OpenMX file extensions:
         ~.[DOS|PDOS].[Tetrahedron|Gaussian]<.atom(int).(orbital)
        :param method: the method which has been used to calcualte the density
                       of states ('Tetrahedron' or 'Gaussian')
        :param pdos: True if the pseudo-density of states have been calculated,
                     False if only the total density of states has been
                     calculated
        :param atom_index: positive integer, n. For the nth atom in the unit
                           cell as specified in the OpenMX input file
        :param orbital: '' or 's1' or 'p1', 'p2', 'p3' or 'd1', 'd2', 'd3',
                        'd4', 'd5' etc. If pdos is True then this specifies the
                        pdos from a particular orbital to read from. If '' is
                        given then the total pdos from the given atom is read.
        :param spin_polarization: if True this will read the separate pdos for
                                  up and down spin states.
        :return: None
        """
        add = False
        if not spin_polarization and self.calc['initial_magnetic_moments']:
            add = True
        p = ''
        if pdos:
            p = 'P'
        filename = self.calc.label + '.' + p + 'DOS.' + method
        if pdos:
            period = ''
            if orbital != '':
                period = '.'
            filename += '.atom' + str(atom_index) + period + orbital
        f = open(filename, 'r')
        line = '\n'
        number_of_lines = -1
        while line != '':
            line = f.readline()
            number_of_lines += 1
        f.close()
        key = ''
        atom_and_orbital = ''
        if pdos:
            key = 'p'
            atom_and_orbital = str(atom_index) + orbital
        key += 'dos'
        self.dos_dict[key + '_energies_' + atom_and_orbital] = np.ndarray(
              number_of_lines)
        if spin_polarization:
            self.dos_dict[key + atom_and_orbital + 'up'] = \
                np.ndarray(number_of_lines)
            self.dos_dict[key + atom_and_orbital + 'down'] = \
                np.ndarray(number_of_lines)
            self.dos_dict[key + '_cum_' + atom_and_orbital + 'up'] = \
                np.ndarray(number_of_lines)
            self.dos_dict[key + '_cum_' + atom_and_orbital + 'down'] = \
                np.ndarray(number_of_lines)
        else:
            self.dos_dict[key + atom_and_orbital] = np.ndarray(number_of_lines)
            self.dos_dict[key + '_cum_' + atom_and_orbital] = \
                np.ndarray(number_of_lines)
        f = open(filename, 'r')
        if spin_polarization:
            for i in range(number_of_lines):
                line = f.readline()
                self.dos_dict[key + '_energies_' + atom_and_orbital][i] = \
                    read_nth_to_last_value(line, 5)
                self.dos_dict[key + atom_and_orbital + 'up'][i] = \
                    read_nth_to_last_value(line, 4)
                self.dos_dict[key + atom_and_orbital + 'down'][i] = \
                    -float(read_nth_to_last_value(line, 3))
                self.dos_dict[key + '_cum_' + atom_and_orbital + 'up'][i] = \
                    read_nth_to_last_value(line, 2)
                self.dos_dict[key + '_cum_' + atom_and_orbital + 'down'][i] = \
                    read_nth_to_last_value(line)
        elif add:
            for i in range(number_of_lines):
                line = f.readline()
                self.dos_dict[key + '_energies_' + atom_and_orbital][i] = \
                    read_nth_to_last_value(line, 5)
                self.dos_dict[key + atom_and_orbital][i] = \
                    float(read_nth_to_last_value(line, 4)) - \
                    float(read_nth_to_last_value(line, 3))
                self.dos_dict[key + '_cum_' + atom_and_orbital][i] = \
                    float(read_nth_to_last_value(line, 2)) + \
                    float(read_nth_to_last_value(line))
        else:
            for i in range(number_of_lines):
                line = f.readline()
                self.dos_dict[key + '_energies_' + atom_and_orbital][i] = \
                    read_nth_to_last_value(line, 3)
                self.dos_dict[key + atom_and_orbital][i] = \
                    read_nth_to_last_value(line, 2)
                self.dos_dict[key + '_cum_' + atom_and_orbital][i] = \
                    read_nth_to_last_value(line)
        f.close()

    def subplot_dos(self, axis, density=True, cum=False, pdos=False,
                    atom_index=1, orbital='', spin='',
                    erange=(-25, 20), fermi_level=True):
        """
        Plots a graph of (pseudo-)density of states against energy onto a given
        axis of a subplot.
        :param axis: matplotlib.pyplot.Axes object. This allows the graph to
                     plotted on any desired axis of a plot.
        :param density: If True, the density of states will be plotted
        :param cum: If True, the cumulative (or integrated) density of states
                    will be plotted
        :param pdos: If True, the pseudo-density of states will be plotted for
                     a given atom and orbital
        :param atom_index: If pdos is True, atom_index specifies which atom's
                           PDOS to plot.
        :param orbital: If pdos is True, orbital specifies which orbital's PDOS
                        to plot.
        :param spin: If '', density of states for both spin states will be
                     combined into one plot. If 'up' or 'down', a given spin
                     state's PDOS will be plotted.
        :return: None
        """
        p = ''
        bottom_index = 0
        atom_orbital = atom_orbital_spin = ''
        if pdos:
            p = 'p'
            atom_orbital += str(atom_index) + orbital
        atom_orbital_spin += atom_orbital + spin
        key = p + 'dos'
        density_color = 'r'
        cum_color = 'b'
        if spin == 'down':
            density_color = 'c'
            cum_color = 'm'
        if density and cum:
            axis_twin = axis.twinx()
            axis.plot(self.dos_dict[key + '_energies_' + atom_orbital],
                      self.dos_dict[key + atom_orbital_spin],
                      density_color)
            axis_twin.plot(self.dos_dict[key + '_energies_' + atom_orbital],
                           self.dos_dict[key + '_cum_' + atom_orbital_spin],
                           cum_color)
            max_density = max(self.dos_dict[key + atom_orbital_spin])
            max_cum = max(self.dos_dict[key + '_cum_' + atom_orbital_spin])
            if not max_density:
                max_density = 1.
            if not max_cum:
                max_cum = 1
            axis.set_ylim(ymax=max_density)
            axis_twin.set_ylim(ymax=max_cum)
            axis.set_ylim(ymin=0.)
            axis_twin.set_ylim(ymin=0.)
            label_index = 0
            yticklabels = axis.get_yticklabels()
            if spin == 'down':
                bottom_index = len(yticklabels) - 1
            for t in yticklabels:
                if label_index == bottom_index or label_index == \
                                                  len(yticklabels) // 2:
                    t.set_color(density_color)
                else:
                    t.set_visible(False)
                label_index += 1
            label_index = 0
            yticklabels = axis_twin.get_yticklabels()
            if spin == 'down':
                bottom_index = len(yticklabels) - 1
            for t in yticklabels:
                if label_index == bottom_index or label_index == \
                                                  len(yticklabels) // 2:
                    t.set_color(cum_color)
                else:
                    t.set_visible(False)
                label_index += 1
            if spin == 'down':
                axis.set_ylim(axis.get_ylim()[::-1])
                axis_twin.set_ylim(axis_twin.get_ylim()[::-1])
        else:
            color = density_color
            if cum:
                color = cum_color
                key += '_cum_'
            key += atom_orbital_spin
            axis.plot(self.dos_dict[p + 'dos_energies_' + atom_orbital],
                      self.dos_dict[key], color)
            maximum = max(self.dos_dict[key])
            if not maximum:
                maximum = 1.
            axis.set_ylim(ymax=maximum)
            axis.set_ylim(ymin=0.)
            label_index = 0
            yticklabels = axis.get_yticklabels()
            if spin == 'down':
                bottom_index = len(yticklabels) - 1
            for t in yticklabels:
                if label_index == bottom_index or label_index == \
                                                  len(yticklabels) // 2:
                    t.set_color(color)
                else:
                    t.set_visible(False)
                label_index += 1
            if spin == 'down':
                axis.set_ylim(axis.get_ylim()[::-1])
        if fermi_level:
            axis.axvspan(erange[0], 0., color='y', alpha=0.5)

    def plot_dos(self, density=True, cum=False, pdos=False, orbital_list=None,
                 atom_index_list=None, spins=('up', 'down'), fermi_level=True,
                 spin_polarization=False, erange=(-25, 20), atoms=None,
                 method='Tetrahedron', file_format=None):
        """
        Generates a graphical figure containing possible subplots of different
        PDOSs of different atoms, orbitals and spin state combinations.
        :param density: If True, density of states will be plotted
        :param cum: If True, cumulative density of states will be plotted
        :param pdos: If True, pseudo-density of states will be plotted for
                     given atoms and orbitals
        :param atom_index_list: If pdos is True, atom_index_list specifies
                                which atoms will have their PDOS plotted.
        :param orbital_list: If pdos is True, orbital_list specifies which
                             orbitals will have their PDOS plotted.
        :param spins: If '' in spins, density of states for both spin states
                      will be combined into one graph. If 'up' or
        'down' in spins, a given spin state's PDOS graph will be plotted.
        :param spin_polarization: If spin_polarization is False then spin
                                  states will not be separated in different
                                  PDOS's.
        :param erange: range of energies to view DOS
        :return: matplotlib.figure.Figure and matplotlib.axes.Axes object
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        if not spin_polarization:
            spins = ['']
        number_of_spins = len(spins)
        if orbital_list is None:
            orbital_list = ['']
        number_of_atoms = 1
        number_of_orbitals = 1
        p = ''
        if pdos:
            p = 'P'
            if atom_index_list is None:
                atom_index_list = [i + 1 for i in range(len(atoms))]
            number_of_atoms = len(atom_index_list)
            number_of_orbitals = len(orbital_list)
        figure, axes = plt.subplots(number_of_orbitals * number_of_spins,
                                    number_of_atoms, sharex=True, sharey=False,
                                    squeeze=False)
        for i in range(number_of_orbitals):
            for s in range(number_of_spins):
                row_index = i * number_of_spins + s
                for j in range(number_of_atoms):
                    self.subplot_dos(fermi_level=fermi_level, density=density,
                                     axis=axes[row_index][j], erange=erange,
                                     atom_index=atom_index_list[j], pdos=pdos,
                                     orbital=orbital_list[i], spin=spins[s],
                                     cum=cum)
                    if j == 0 and pdos:
                        orbital = orbital_list[i]
                        if orbital == '':
                            orbital = 'All'
                        if spins[s]:
                            orbital += ' ' + spins[s]
                        axes[row_index][j].set_ylabel(orbital)
                    if row_index == 0 and pdos:
                        atom_symbol = ''
                        if atoms:
                            atom_symbol = ' (' + \
                                atoms[atom_index_list[j]].symbol + ')'
                        axes[row_index][j].set_title(
                            'Atom ' + str(atom_index_list[j]) + atom_symbol)
                    if row_index == number_of_orbitals * number_of_spins - 1:
                        axes[row_index][j].set_xlabel(
                            'Energy above Fermi Level (eV)')
        plt.xlim(xmin=erange[0], xmax=erange[1])
        if density and cum:
            figure.suptitle(self.calc.label)
            xdata = (0., 1.)
            ydata = (0., 0.)
            key_tuple = (Line2D(color='r', xdata=xdata, ydata=ydata),
                         Line2D(color='b', xdata=xdata, ydata=ydata))
            if spin_polarization:
                key_tuple = (Line2D(color='r', xdata=xdata, ydata=ydata),
                             Line2D(color='b', xdata=xdata, ydata=ydata),
                             Line2D(color='c', xdata=xdata, ydata=ydata),
                             Line2D(color='m', xdata=xdata, ydata=ydata))
            title_tuple = (p + 'DOS (eV^-1)', 'Number of States per Unit Cell')
            if spin_polarization:
                title_tuple = (p + 'DOS (eV^-1), spin up',
                               'Number of States per Unit Cell, spin up',
                               p + 'DOS (eV^-1), spin down',
                               'Number of States per Unit Cell, spin down')
            figure.legend(key_tuple, title_tuple, 'lower center')
        elif density:
            figure.suptitle(self.calc.prefix + ': ' + p + 'DOS (eV^-1)')
        elif cum:
            figure.suptitle(self.calc.prefix + ': Number of States')
        extra_margin = 0
        if density and cum and spin_polarization:
            extra_margin = 0.1
        plt.subplots_adjust(hspace=0., bottom=0.2 + extra_margin, wspace=0.29,
                            left=0.09, right=0.95)
        if file_format:
            orbitals = ''
            if pdos:
                atom_index_list = map(str, atom_index_list)
                atoms = '&'.join(atom_index_list)
                if '' in orbital_list:
                    all_index = orbital_list.index('')
                    orbital_list.remove('')
                    orbital_list.insert(all_index, 'all')
                orbitals = ''.join(orbital_list)
            plt.savefig(filename=self.calc.label + '.' + p + 'DOS.' +
                        method + '.atoms' + atoms + '.' + orbitals + '.' +
                        file_format)
        if not file_format:
            plt.show()
        return figure, axes

    def calc_dos(self, method='Tetrahedron', pdos=False, gaussian_width=0.1,
                 atom_index_list=None):
        """
        Python interface for DosMain (OpenMX's density of states calculator).
        Can automate the density of states
        calculations used in OpenMX by processing .Dos.val and .Dos.vec files.
        :param method: method to be used to calculate the density of states
                       from eigenvalues and eigenvectors.
                       ('Tetrahedron' or 'Gaussian')
        :param pdos: If True, the pseudo-density of states is calculated for a
                     given list of atoms for each orbital. If the system is
                     spin polarized, then each up and down state is also
                     calculated.
        :param gaussian_width: If the method is 'Gaussian' then gaussian_width
                               is required (eV).
        :param atom_index_list: If pdos is True, a list of atom indices are
                                required to generate the pdos of each of those
                                specified atoms.
        :return: None
        """
        method_code = '2\n'
        if method == 'Tetrahedron':
            method_code = '1\n'
        pdos_code = '1\n'
        if pdos:
            pdos_code = '2\n'
        with open(os.path.join(self.calc.directory, 'std_dos.in'), 'w') as f:
            f.write(method_code)
            if method == 'Gaussian':
                f.write(str(gaussian_width) + '\n')
            f.write(pdos_code)
            if pdos:
                atoms_code = ''
                if atom_index_list is None:
                    for i in range(len(self.calc.atoms)):
                        atoms_code += str(i + 1) + ' '
                else:
                    for i in atom_index_list:
                        atoms_code += str(i) + ' '
                atoms_code += '\n'
                f.write(atoms_code)
            f.close()
        executable_name = 'DosMain'
        input_files = (self.calc.label + '.Dos.val', self.calc.label +
                       '.Dos.vec', os.path.join(self.calc.directory,
                                                'std_dos.in'))
        argument_format = '%s %s < %s'
        input_command(self.calc, executable_name, input_files, argument_format)

    def get_dos(self, atom_index_list=None, method='Tetrahedron',
                gaussian_width=0.1, pdos=False, orbital_list=None,
                spin_polarization=None, density=True, cum=False,
                erange=(-25, 20), file_format=None, atoms=None,
                fermi_level=True):
        """
        Wraps all the density of states processing functions. Can go from
        .Dos.val and .Dos.vec files to a graphical figure showing many
        different PDOS plots against energy in one step.
        :param atom_index_list:
        :param method: method to be used to calculate the density of states
                       from eigenvalues and eigenvectors.
                       ('Tetrahedron' or 'Gaussian')
        :param gaussian_width: If the method is 'Gaussian' then gaussian_width
                               is required (eV).
        :param pdos: If True, the pseudo-density of states is calculated for a
                     given list of atoms for each orbital. If the system is
                     spin polarized, then each up and down state is also
                     calculated.
        :param orbital_list: If pdos is True, a list of atom indices are
                             required to generate the pdos of each of those
                             specified atoms.
        :param spin_polarization: If spin_polarization is False then spin
                                  states will not be separated in different
                                  PDOS's.
        :param density: If True, density of states will be plotted
        :param cum: If True, cumulative (or integrated) density of states will
                    be plotted
        :param erange: range of energies to view the DOS
        :param file_format: If not None, a file will be saved automatically in
                            that format ('pdf', 'png', 'jpeg' etc.)
        :return: matplotlib.figure.Figure object
        """
        if spin_polarization is None:
            spin_polarization = bool(self.calc['initial_magnetic_moments'])
        if spin_polarization and not self.calc['initial_magnetic_moments']:
            warnings.warn('No spin polarization calculations provided')
            spin_polarization = False
        if atom_index_list is None:
            atom_index_list = [1]
        if method == 'Tetrahedron' and self.calc['dos_kgrid'] == (1, 1, 1):
            raise ValueError('Not enough k-space grid points.')
        self.calc_dos(atom_index_list=atom_index_list, pdos=pdos,
                      method=method, gaussian_width=gaussian_width)
        if pdos:
            if orbital_list is None:
                orbital_list = ['']
            orbital_list = list(orbital_list)
            if 's' in orbital_list:
                s_index = orbital_list.index('s')
                orbital_list.remove('s')
                orbital_list.insert(s_index, 's1')
            if 'p' in orbital_list:
                p_index = orbital_list.index('p')
                orbital_list.remove('p')
                orbital_list.insert(p_index, 'p3')
                orbital_list.insert(p_index, 'p2')
                orbital_list.insert(p_index, 'p1')
            if 'd' in orbital_list:
                d_index = orbital_list.index('d')
                orbital_list.remove('d')
                orbital_list.insert(d_index, 'd5')
                orbital_list.insert(d_index, 'd4')
                orbital_list.insert(d_index, 'd3')
                orbital_list.insert(d_index, 'd2')
                orbital_list.insert(d_index, 'd1')
            if 'f' in orbital_list:
                f_index = orbital_list.index('f')
                orbital_list.remove('f')
                orbital_list.insert(f_index, 'f7')
                orbital_list.insert(f_index, 'f6')
                orbital_list.insert(f_index, 'f5')
                orbital_list.insert(f_index, 'f4')
                orbital_list.insert(f_index, 'f3')
                orbital_list.insert(f_index, 'f2')
                orbital_list.insert(f_index, 'f1')

            for atom_index in atom_index_list:
                for orbital in orbital_list:
                    self.read_dos(method=method, atom_index=atom_index,
                                  pdos=pdos, orbital=orbital,
                                  spin_polarization=spin_polarization)
        else:
            self.read_dos(method=method, spin_polarization=spin_polarization)
        return self.plot_dos(density=density, cum=cum, atoms=atoms,
                             atom_index_list=atom_index_list, pdos=pdos,
                             orbital_list=orbital_list, erange=erange,
                             spin_polarization=spin_polarization,
                             file_format=file_format, method=method,
                             fermi_level=fermi_level)
