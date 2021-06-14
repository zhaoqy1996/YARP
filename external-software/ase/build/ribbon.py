from __future__ import print_function
from math import sqrt

import numpy as np

from ase.atoms import Atoms


def graphene_nanoribbon(n, m, type='zigzag', saturated=False, C_H=1.09,
                        C_C=1.42, vacuum=None, magnetic=False, initial_mag=1.12,
                        sheet=False, main_element='C', saturate_element='H'):
    """Create a graphene nanoribbon.

    Creates a graphene nanoribbon in the x-z plane, with the nanoribbon
    running along the z axis.

    Parameters:

    n: int
        The width of the nanoribbon.  For armchair nanoribbons, this
        n may be half-integer to repeat by half a cell.
    m: int
        The length of the nanoribbon.
    type: str
        The orientation of the ribbon.  Must be either 'zigzag'
        or 'armchair'.
    saturated: bool
        If true, hydrogen atoms are placed along the edge.
    C_H: float
        Carbon-hydrogen bond length.  Default: 1.09 Angstrom.
    C_C: float
        Carbon-carbon bond length.  Default: 1.42 Angstrom.
    vacuum: None (default) or float
        Amount of vacuum added to non-periodic directions, if present.
    magnetic: bool
        Make the edges magnetic.
    initial_mag: float
        Magnitude of magnetic moment if magnetic.
    sheet: bool
        If true, make an infinite sheet instead of a ribbon (default: False)
    """

    if m % 1 != 0:
        raise ValueError('m must be integer')
    if type == 'zigzag' and n % 1 != 0:
        raise ValueError('n must be an integer for zigzag ribbons')

    b = sqrt(3) * C_C / 4
    arm_unit = Atoms(main_element + '4',
                     pbc=(1, 0, 1),
                     cell=[4 * b, 0, 3 * C_C])
    arm_unit.positions = [[0, 0, 0],
                          [b * 2, 0, C_C / 2.],
                          [b * 2, 0, 3 * C_C / 2.],
                          [0, 0, 2 * C_C]]
    arm_unit_half = Atoms(main_element + '2',
                          pbc=(1, 0, 1),
                          cell=[2 * b, 0, 3 * C_C])
    arm_unit_half.positions = [[b * 2, 0, C_C / 2.],
                               [b * 2, 0, 3 * C_C / 2.]]
    zz_unit = Atoms(main_element + '2',
                    pbc=(1, 0, 1),
                    cell=[3 * C_C / 2.0, 0, b * 4])
    zz_unit.positions = [[0, 0, 0],
                         [C_C / 2.0, 0, b * 2]]
    atoms = Atoms()

    if type == 'zigzag':
        edge_index0 = np.arange(m) * 2
        edge_index1 = (n - 1) * m * 2 + np.arange(m) * 2 + 1

        if magnetic:
            mms = np.zeros(m * n * 2)
            for i in edge_index0:
                mms[i] = initial_mag
            for i in edge_index1:
                mms[i] = -initial_mag

        for i in range(n):
            layer = zz_unit.repeat((1, 1, m))
            layer.positions[:, 0] += 3 * C_C / 2 * i
            if i % 2 == 1:
                layer.positions[:, 2] += 2 * b
                layer[-1].position[2] -= b * 4 * m
            atoms += layer

        xmin = atoms.positions[0, 0]

        if magnetic:
            atoms.set_initial_magnetic_moments(mms)
        if saturated:
            H_atoms0 = Atoms(saturate_element + str(m))
            H_atoms0.positions = atoms[edge_index0].positions
            H_atoms0.positions[:, 0] -= C_H
            H_atoms1 = Atoms(saturate_element + str(m))
            H_atoms1.positions = atoms[edge_index1].positions
            H_atoms1.positions[:, 0] += C_H
            atoms += H_atoms0 + H_atoms1
        atoms.cell = [n * 3 * C_C / 2, 0, m * 4 * b]

    elif type == 'armchair':
        n *= 2
        n_int = int(round(n))
        if abs(n_int - n) > 1e-10:
            raise ValueError(
                'The argument n has to be half-integer for armchair ribbons.')
        n = n_int

        for i in range(n // 2):
            layer = arm_unit.repeat((1, 1, m))
            layer.positions[:, 0] -= 4 * b * i
            atoms += layer
        if n % 2:
            layer = arm_unit_half.repeat((1, 1, m))
            layer.positions[:, 0] -= 4 * b * (n // 2)
            atoms += layer

        xmin = atoms.positions[-1, 0]

        if saturated:
            if n % 2:
                arm_right_saturation = Atoms(saturate_element + '2',
                                             pbc=(1, 0, 1),
                                             cell=[2 * b, 0, 3 * C_C])
                arm_right_saturation.positions = [
                    [- sqrt(3) / 2 * C_H, 0, C_C / 2 - C_H * 0.5],
                    [- sqrt(3) / 2 * C_H, 0, 3 * C_C / 2.0 + C_H * 0.5]]
            else:
                arm_right_saturation = Atoms(saturate_element + '2',
                                             pbc=(1, 0, 1),
                                             cell=[4 * b, 0, 3 * C_C])
                arm_right_saturation.positions = [
                    [- sqrt(3) / 2 * C_H, 0, C_H * 0.5],
                    [- sqrt(3) / 2 * C_H, 0, 2 * C_C - C_H * 0.5]]
            arm_left_saturation = Atoms(saturate_element + '2', pbc=(1, 0, 1),
                                        cell=[4 * b, 0, 3 * C_C])
            arm_left_saturation.positions = [
                [b * 2 + sqrt(3) / 2 * C_H, 0, C_C / 2 - C_H * 0.5],
                [b * 2 + sqrt(3) / 2 * C_H, 0, 3 * C_C / 2.0 + C_H * 0.5]]
            arm_right_saturation.positions[:, 0] -= 4 * b * (n / 2.0 - 1)

            atoms += arm_right_saturation.repeat((1, 1, m))
            atoms += arm_left_saturation.repeat((1, 1, m))

        atoms.cell = [b * 4 * n / 2.0, 0, 3 * C_C * m]

    atoms.set_pbc([sheet, False, True])

    # The ribbon was 'built' from x=0 towards negative x.
    # Move the ribbon to positive x:
    atoms.positions[:, 0] -= xmin
    if not sheet:
        atoms.cell[0] = 0.0
    if vacuum is not None:
        atoms.center(vacuum, axis=1)
        if not sheet:
            atoms.center(vacuum, axis=0)
    return atoms
