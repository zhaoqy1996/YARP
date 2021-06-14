from __future__ import unicode_literals
from functools import partial

from ase.gui.i18n import _

import ase.gui.ui as ui
from ase.gui.widgets import Element
from ase.gui.utils import get_magmoms


class ModifyAtoms:
    """Presents a dialog box where the user is able to change the
    atomic type, the magnetic moment and tags of the selected atoms.
    """
    def __init__(self, gui):
        self.gui = gui
        selected = self.selection()
        if not selected.any():
            ui.error(_('No atoms selected!'))
            return

        win = ui.Window(_('Modify'))
        element = Element(callback=self.set_element)
        win.add(element)
        win.add(ui.Button(_('Change element'),
                          partial(self.set_element, element)))
        self.tag = ui.SpinBox(0, -1000, 1000, 1, self.set_tag)
        win.add([_('Tag'), self.tag])
        self.magmom = ui.SpinBox(0.0, -10, 10, 0.1, self.set_magmom)
        win.add([_('Moment'), self.magmom])

        atoms = self.gui.atoms
        Z = atoms.numbers
        if Z.ptp() == 0:
            element.Z = Z[0]

        tags = atoms.get_tags()[selected]
        if tags.ptp() == 0:
            self.tag.value = tags[0]

        magmoms = get_magmoms(atoms)[selected]
        if magmoms.round(2).ptp() == 0.0:
            self.magmom.value = round(magmoms[0], 2)

    def selection(self):
        return self.gui.images.selected[:len(self.gui.atoms)]

    def set_element(self, element):
        self.gui.atoms.numbers[self.selection()] = element.Z
        self.gui.draw()

    def set_tag(self):
        tags = self.gui.atoms.get_tags()
        tags[self.selection()] = self.tag.value
        self.gui.atoms.set_tags(tags)
        self.gui.draw()

    def set_magmom(self):
        magmoms = get_magmoms(self.gui.atoms)
        magmoms[self.selection()] = self.magmom.value
        self.gui.atoms.set_initial_magnetic_moments(magmoms)
        self.gui.draw()
