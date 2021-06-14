from __future__ import unicode_literals
from ase.gui.i18n import _

import ase.gui.ui as ui
from ase.utils import rotate, irotate


class Rotate:
    update = True

    def __init__(self, gui):
        self.gui = gui
        win = ui.Window(_('Rotate'))
        win.add(_('Rotation angles:'))
        self.rotate = [ui.SpinBox(42.0, -360, 360, 1, self.change)
                       for i in '123']
        win.add(self.rotate)
        win.add(ui.Button(_('Update'), self.update_angles))
        win.add(_('Note:\nYou can rotate freely\n'
                  'with the mouse, by holding\n'
                  'down mouse button 2.'))
        self.update_angles()

    def change(self):
        x, y, z = [float(a.value) for a in self.rotate]
        self.gui.axes = rotate('%fx,%fy,%fz' % (x, y, z))
        self.gui.set_frame()

    def update_angles(self):
        angles = irotate(self.gui.axes)
        for r, a in zip(self.rotate, angles):
            r.value = a
