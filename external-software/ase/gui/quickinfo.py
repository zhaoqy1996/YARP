# -*- encoding: utf-8
"Module for displaying information about the system."

from __future__ import unicode_literals

import numpy as np
from ase.gui.i18n import _
import warnings


ucellformat = """\
  {:8.3f}  {:8.3f}  {:8.3f}
  {:8.3f}  {:8.3f}  {:8.3f}
  {:8.3f}  {:8.3f}  {:8.3f}
"""


def info(gui):
    images = gui.images
    nimg = len(images)
    atoms = gui.atoms

    tokens = []
    def add(token=''):
        tokens.append(token)

    if len(atoms) < 1:
        add(_('This frame has no atoms.'))
    else:
        img = gui.frame

        if nimg == 1:
            add(_('Single image loaded.'))
        else:
            add(_('Image {} loaded (0–{}).').format(img, nimg - 1))
        add()
        add(_('Number of atoms: {}').format(len(atoms)))

        # We need to write Å³ further down, so we have no choice but to
        # use proper subscripts in the chemical formula:
        formula = atoms.get_chemical_formula()
        subscripts = dict(zip('0123456789', '₀₁₂₃₄₅₆₇₈₉'))
        pretty_formula = ''.join(subscripts.get(c, c) for c in formula)
        add(pretty_formula)

        add()
        add(_('Unit cell [Å]:'))
        add(ucellformat.format(*atoms.cell.ravel()))
        periodic = [[_('no'), _('yes')][periodic] for periodic in atoms.pbc]
        # TRANSLATORS: This has the form Periodic: no, no, yes
        add(_('Periodic: {}, {}, {}').format(*periodic))

        if nimg > 1:
            if all((atoms.cell == img.cell).all() for img in images):
                add(_('Unit cell is fixed.'))
            else:
                add(_('Unit cell varies.'))

        if atoms.number_of_lattice_vectors == 3:
            add(_('Volume: {:.3f} Å³').format(atoms.get_volume()))

        # Print electronic structure information if we have a calculator
        if atoms.calc:
            calc = atoms.calc

            def getresult(name, get_quantity):
                # ase/io/trajectory.py line 170 does this by using
                # the get_property(prop, atoms, allow_calculation=False)
                # so that is an alternative option.
                try:
                    if calc.calculation_required(atoms, [name]):
                        quantity = None
                    else:
                        quantity = get_quantity()
                except Exception as err:
                    quantity = None
                    errmsg = ('An error occured while retrieving {} '
                              'from the calculator: {}'.format(name, err))
                    warnings.warn(errmsg)
                return quantity

            # SinglePointCalculators are named after the code which
            # produced the result, so this will typically list the
            # name of a code even if they are just cached results.
            add()
            from ase.calculators.singlepoint import SinglePointCalculator
            if isinstance(calc, SinglePointCalculator):
                add(_('Calculator: {} (cached)').format(calc.name))
            else:
                add(_('Calculator: {} (attached)').format(calc.name))

            energy = getresult('energy', atoms.get_potential_energy)
            forces = getresult('forces', atoms.get_forces)
            magmom = getresult('magmom', atoms.get_magnetic_moment)

            if energy is not None:
                energy_str = _('Energy: {:.3f} eV').format(energy)
                add(energy_str)

            if forces is not None:
                maxf = np.linalg.norm(forces, axis=1).max()
                forces_str = _('Max force: {:.3f} eV/Å').format(maxf)
                add(forces_str)

            if magmom is not None:
                mag_str = _('Magmom: {:.3f} µ').format(magmom)
                add(mag_str)

    return '\n'.join(tokens)
