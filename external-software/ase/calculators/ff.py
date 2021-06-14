from __future__ import division

import numpy as np

from ase.calculators.calculator import Calculator
from ase.utils import ff


class ForceField(Calculator):
    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, morses=None, bonds=None, angles=None, dihedrals=None,
                 vdws=None, coulombs=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        if (morses is None and
            bonds is None and
            angles is None and
            dihedrals is None and
            vdws is None and
                coulombs is None):
            raise ImportError("At least one of morses, bonds, angles, dihedrals,"
                              "vdws or coulombs lists must be defined!")
        if morses is None:
            self.morses = []
        else:
            self.morses = morses
        if bonds is None:
            self.bonds = []
        else:
            self.bonds = bonds
        if angles is None:
            self.angles = []
        else:
            self.angles = angles
        if dihedrals is None:
            self.dihedrals = []
        else:
            self.dihedrals = dihedrals
        if vdws is None:
            self.vdws = []
        else:
            self.vdws = vdws
        if coulombs is None:
            self.coulombs = []
        else:
            self.coulombs = coulombs

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes:
            for name in ['energy', 'forces', 'hessian']:
                self.results.pop(name, None)
        if 'energy' not in self.results:
            energy = 0.0
            for morse in self.morses:
                i, j, e = ff.get_morse_potential_value(atoms, morse)
                energy += e
            for bond in self.bonds:
                i, j, e = ff.get_bond_potential_value(atoms, bond)
                energy += e
            for angle in self.angles:
                i, j, k, e = ff.get_angle_potential_value(atoms, angle)
                energy += e
            for dihedral in self.dihedrals:
                i, j, k, l, e = ff.get_dihedral_potential_value(
                    atoms, dihedral)
                energy += e
            for vdw in self.vdws:
                i, j, e = ff.get_vdw_potential_value(atoms, vdw)
                energy += e
            for coulomb in self.coulombs:
                i, j, e = ff.get_coulomb_potential_value(atoms, coulomb)
                energy += e
            self.results['energy'] = energy
        if 'forces' not in self.results:
            forces = np.zeros(3 * len(atoms))
            for morse in self.morses:
                i, j, g = ff.get_morse_potential_gradient(atoms, morse)
                limits = get_limits([i, j])
                for gb, ge, lb, le in limits:
                    forces[gb:ge] -= g[lb:le]
            for bond in self.bonds:
                i, j, g = ff.get_bond_potential_gradient(atoms, bond)
                limits = get_limits([i, j])
                for gb, ge, lb, le in limits:
                    forces[gb:ge] -= g[lb:le]
            for angle in self.angles:
                i, j, k, g = ff.get_angle_potential_gradient(atoms, angle)
                limits = get_limits([i, j, k])
                for gb, ge, lb, le in limits:
                    forces[gb:ge] -= g[lb:le]
            for dihedral in self.dihedrals:
                i, j, k, l, g = ff.get_dihedral_potential_gradient(
                    atoms, dihedral)
                limits = get_limits([i, j, k, l])
                for gb, ge, lb, le in limits:
                    forces[gb:ge] -= g[lb:le]
            for vdw in self.vdws:
                i, j, g = ff.get_vdw_potential_gradient(atoms, vdw)
                limits = get_limits([i, j])
                for gb, ge, lb, le in limits:
                    forces[gb:ge] -= g[lb:le]
            for coulomb in self.coulombs:
                i, j, g = ff.get_coulomb_potential_gradient(atoms, coulomb)
                limits = get_limits([i, j])
                for gb, ge, lb, le in limits:
                    forces[gb:ge] -= g[lb:le]
            self.results['forces'] = np.reshape(forces, (len(atoms), 3))
        if 'hessian' not in self.results:
            hessian = np.zeros((3 * len(atoms), 3 * len(atoms)))
            for morse in self.morses:
                i, j, h = ff.get_morse_potential_hessian(atoms, morse)
                limits = get_limits([i, j])
                for gb1, ge1, lb1, le1 in limits:
                    for gb2, ge2, lb2, le2 in limits:
                        hessian[gb1:ge1, gb2:ge2] += h[lb1:le1, lb2:le2]
            for bond in self.bonds:
                i, j, h = ff.get_bond_potential_hessian(atoms, bond)
                limits = get_limits([i, j])
                for gb1, ge1, lb1, le1 in limits:
                    for gb2, ge2, lb2, le2 in limits:
                        hessian[gb1:ge1, gb2:ge2] += h[lb1:le1, lb2:le2]
            for angle in self.angles:
                i, j, k, h = ff.get_angle_potential_hessian(atoms, angle)
                limits = get_limits([i, j, k])
                for gb1, ge1, lb1, le1 in limits:
                    for gb2, ge2, lb2, le2 in limits:
                        hessian[gb1:ge1, gb2:ge2] += h[lb1:le1, lb2:le2]
            for dihedral in self.dihedrals:
                i, j, k, l, h = ff.get_dihedral_potential_hessian(
                    atoms, dihedral)
                limits = get_limits([i, j, k, l])
                for gb1, ge1, lb1, le1 in limits:
                    for gb2, ge2, lb2, le2 in limits:
                        hessian[gb1:ge1, gb2:ge2] += h[lb1:le1, lb2:le2]
            for vdw in self.vdws:
                i, j, h = ff.get_vdw_potential_hessian(atoms, vdw)
                limits = get_limits([i, j])
                for gb1, ge1, lb1, le1 in limits:
                    for gb2, ge2, lb2, le2 in limits:
                        hessian[gb1:ge1, gb2:ge2] += h[lb1:le1, lb2:le2]
            for coulomb in self.coulombs:
                i, j, h = ff.get_coulomb_potential_hessian(atoms, coulomb)
                limits = get_limits([i, j])
                for gb1, ge1, lb1, le1 in limits:
                    for gb2, ge2, lb2, le2 in limits:
                        hessian[gb1:ge1, gb2:ge2] += h[lb1:le1, lb2:le2]
            self.results['hessian'] = hessian

    def get_hessian(self, atoms=None):
        return self.get_property('hessian', atoms)


def get_limits(indices):
    gstarts = []
    gstops = []
    lstarts = []
    lstops = []
    for l, g in enumerate(indices):
        g3, l3 = 3 * g, 3 * l
        gstarts.append(g3)
        gstops.append(g3 + 3)
        lstarts.append(l3)
        lstops.append(l3 + 3)
    return zip(gstarts, gstops, lstarts, lstops)
