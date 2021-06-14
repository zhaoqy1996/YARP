# encoding: utf-8
from __future__ import unicode_literals

import os
import numpy as np

from ase.gui.i18n import _
from ase import Atoms
import ase.gui.ui as ui
from ase.data import atomic_numbers, chemical_symbols


class AddAtoms:
    def __init__(self, gui):
        self.gui = gui
        win = self.win = ui.Window(_('Add atoms'))
        win.add(_('Specify chemical symbol, formula, or filename.'))

        def set_molecule(value):
            self.entry.value = value
            self.focus()

        def choose_file():
            chooser = ui.ASEFileChooser(self.win.win)
            filename = chooser.go()
            if filename is None:  # No file selected
                return

            self.entry.value = filename

            # Load the file immediately, so we can warn now in case of error
            self.readfile(filename, format=chooser.format)

        self.entry = ui.Entry('', callback=self.add)
        win.add([_('Add:'), self.entry,
                 ui.Button(_('File ...'), callback=choose_file)])

        self._filename = None
        self._atoms_from_file = None

        from ase.collections import g2
        labels = list(sorted(g2.names))
        values = labels

        box = ui.ComboBox(labels, values, callback=set_molecule)
        win.add([_('Get molecule:'), box])
        box.value = 'H2'

        spinners = [ui.SpinBox(0.0, -1e3, 1e3, 0.1, rounding=2, width=3)
                    for __ in range(3)]

        win.add([_('Coordinates:')] + spinners)
        self.spinners = spinners
        win.add(_('Coordinates are relative to the center of the selection, '
                  'if any, else absolute.'))
        self.picky = ui.CheckButton(_('Check positions'), True)
        win.add([ui.Button(_('Add'), self.add),
                 self.picky])
        self.focus()

    def readfile(self, filename, format=None):
        if filename == self._filename:
            # We have this file already
            return self._atoms_from_file

        from ase.io import read
        try:
            atoms = read(filename)
        except Exception as err:
            ui.show_io_error(filename, err)
            atoms = None
            filename = None

        # Cache selected Atoms/filename (or None) for future calls
        self._atoms_from_file = atoms
        self._filename = filename
        return atoms

    def get_atoms(self):
        val = self.entry.value

        if val in atomic_numbers:  # Note: This means val is a symbol!
            return Atoms(val)

        if val.isdigit() and int(val) < len(chemical_symbols):
            return Atoms(numbers=[int(val)])

        from ase.collections import g2
        if val in g2.names:
            return g2[val]

        if os.path.exists(val):
            return self.readfile(val)  # May show UI error

        ui.showerror(_('Cannot add atoms'),
                     _('{} is neither atom, molecule, nor file')
                     .format(val))

        return None

    def getcoords(self):
        addcoords = np.array([spinner.value for spinner in self.spinners])

        pos = self.gui.atoms.positions
        if self.gui.images.selected[:len(pos)].any():
            pos = pos[self.gui.images.selected[:len(pos)]]
            center = pos.mean(0)
            addcoords += center

        return addcoords

    def focus(self):
        self.entry.entry.focus_set()

    def add(self):
        newatoms = self.get_atoms()
        if newatoms is None:  # Error dialog was shown
            return

        newcenter = self.getcoords()

        # Not newatoms.center() because we want the same centering method
        # used for adding atoms relative to selections (mean).
        previous_center = newatoms.positions.mean(0)
        newatoms.positions += newcenter - previous_center

        atoms = self.gui.atoms

        if len(atoms) and self.picky.value:
            from ase.geometry import get_distances
            disps, dists = get_distances(atoms.positions,
                                         newatoms.positions)
            mindist = dists.min()
            if mindist < 0.5:
                ui.showerror(_('Bad positions'),
                             _('Atom would be less than 0.5 Å from '
                               'an existing atom.  To override, '
                               'uncheck the check positions option.'))
                return

        atoms += newatoms

        if len(atoms) > self.gui.images.maxnatoms:
            self.gui.images.initialize(list(self.gui.images),
                                       self.gui.images.filenames)

        self.gui.images.selected[:] = False

        # 'selected' array may be longer than current atoms
        self.gui.images.selected[len(atoms) - len(newatoms):len(atoms)] = True
        self.gui.set_frame()
        self.gui.draw()
