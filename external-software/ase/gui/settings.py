from __future__ import unicode_literals
from ase.gui.i18n import _

import ase.gui.ui as ui


class Settings:
    def __init__(self, gui):
        self.gui = gui
        win = ui.Window(_('Settings'))

        # Constraints
        win.add(_('Constraints:'))
        win.add([ui.Button(_('Constrain'), self.constrain_selected),
                 '/',
                 ui.Button(_('release'), self.release_selected),
                 _(' selected atoms')])
        win.add(ui.Button(_('Constrain immobile atoms'), self.immobile))
        win.add(ui.Button(_('Clear all constraints'), self.clear_constraints))

        # Visibility
        win.add(_('Visibility:'))
        win.add([ui.Button(_('Hide'), self.hide_selected),
                 '/',
                 ui.Button(_('show'), self.show_selected),
                 _(' selected atoms')])
        win.add(ui.Button(_('View all atoms'), self.view_all))

        # Miscellaneous
        win.add(_('Miscellaneous:'))
        self.scale = ui.SpinBox(self.gui.images.atom_scale,
                                0.2, 2.0, 0.1, self.scale_radii)
        win.add([_('Scale atomic radii:'), self.scale])
        self.force_vector_scale = ui.SpinBox(
            self.gui.force_vector_scale,
            0.0, 1e32, 0.1,
            rounding=2,
            callback=self.scale_force_vectors
        )
        win.add([_('Scale force vectors:'), self.force_vector_scale])
        self.velocity_vector_scale = ui.SpinBox(
            self.gui.velocity_vector_scale,
            0.0, 1e32, 0.1,
            rounding=2,
            callback=self.scale_velocity_vectors
        )
        win.add([_('Scale velocity vectors:'), self.velocity_vector_scale])

    def scale_radii(self):
        self.gui.images.atom_scale = self.scale.value
        self.gui.draw()
        return True

    def scale_force_vectors(self):
        self.gui.force_vector_scale = float(self.force_vector_scale.value)
        self.gui.draw()
        return True

    def scale_velocity_vectors(self):
        self.gui.velocity_vector_scale = float(self.velocity_vector_scale.value)
        self.gui.draw()
        return True

    def hide_selected(self):
        self.gui.images.visible[self.gui.images.selected] = False
        self.gui.draw()

    def show_selected(self):
        self.gui.images.visible[self.gui.images.selected] = True
        self.gui.draw()

    def view_all(self):
        self.gui.images.visible[:] = True
        self.gui.draw()

    def constrain_selected(self):
        self.gui.images.set_dynamic(self.gui.images.selected, False)
        self.gui.draw()

    def release_selected(self):
        self.gui.images.set_dynamic(self.gui.images.selected, True)
        self.gui.draw()

    def immobile(self):
        # wtf? XXX detect non-moving atoms somehow
        #self.gui.images.set_dynamic()
        self.gui.draw()

    def clear_constraints(self):
        # This clears *all* constraints.  But when we constrain, we
        # only add FixAtoms....
        for atoms in self.gui.images:
            atoms.constraints = []
        self.gui.draw()
