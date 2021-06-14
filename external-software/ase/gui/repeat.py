from __future__ import unicode_literals
import numpy as np

import ase.gui.ui as ui
from ase.gui.i18n import _


class Repeat:
    def __init__(self, gui):
        win = ui.Window(_('Repeat'))
        win.add(_('Repeat atoms:'))
        self.repeat = [ui.SpinBox(r, 1, 9, 1, self.change)
                       for r in gui.images.repeat]
        win.add(self.repeat)
        win.add(ui.Button(_('Set unit cell'), self.set_unit_cell))

        for sb, vec in zip(self.repeat, gui.atoms.cell):
            if not vec.any():
                sb.active = False

        self.gui = gui

    def change(self):
        repeat = [int(r.value) for r in self.repeat]
        self.gui.images.repeat_images(repeat)
        self.gui.set_frame()

    def set_unit_cell(self):
        self.gui.images.repeat_unit_cell()
        for r in self.repeat:
            r.value = 1
        self.gui.set_frame()

    def set_unit_cell0(self):
        self.gui.images.A *= self.gui.images.repeat.reshape((3, 1))
        self.gui.images.E *= self.gui.images.repeat.prod()
        self.gui.images.repeat = np.ones(3, int)
        for r in self.repeat:
            r.value = 1
        self.gui.set_frame()
