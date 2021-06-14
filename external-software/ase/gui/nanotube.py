# encoding: utf-8
"""Window for setting up Carbon nanotubes and similar tubes.
"""

from __future__ import unicode_literals

import ase.gui.ui as ui
from ase.build import nanotube
from ase.gui.i18n import _
from ase.gui.widgets import Element, pybutton


introtext = _("""\
Set up a Carbon nanotube by specifying the (n,m) roll-up vector.
Please note that m <= n.

Nanotubes of other elements can be made by specifying the element
and bond length.\
""")

py_template = """\
from ase.build import nanotube
atoms = nanotube({n}, {m}, length={length}, bond={bl:.3f}, symbol='{symb}')
"""

label_template = _('{natoms} atoms, diameter: {diameter:.3f} Å, '
                   'total length: {total_length:.3f} Å')


class SetupNanotube:
    """Window for setting up a (Carbon) nanotube."""
    def __init__(self, gui):
        self.element = Element('C', self.make)
        self.bondlength = ui.SpinBox(1.42, 0.0, 10.0, 0.01, self.make)
        self.n = ui.SpinBox(5, 1, 100, 1, self.make)
        self.m = ui.SpinBox(5, 0, 100, 1, self.make)
        self.length = ui.SpinBox(1, 1, 100, 1, self.make)
        self.description = ui.Label('')

        win = self.win = ui.Window(_('Nanotube'))
        win.add(ui.Text(introtext))
        win.add(self.element)
        win.add([_('Bond length: '),
                 self.bondlength,
                 _(u'Å')])
        win.add(_('Select roll-up vector (n,m) and tube length:'))
        win.add(['n:', self.n,
                 'm:', self.m,
                 _('Length:'), self.length])
        win.add(self.description)
        win.add([pybutton(_('Creating a nanoparticle.'), self.make),
                 ui.Button(_('Apply'), self.apply),
                 ui.Button(_('OK'), self.ok)])

        self.gui = gui
        self.atoms = None

    def make(self, element=None):
        symbol = self.element.symbol
        if symbol is None:
            self.atoms = None
            self.python = None
            self.description.text = ''
            return

        n = self.n.value
        m = self.m.value
        length = self.length.value
        bl = self.bondlength.value
        self.atoms = nanotube(n, m, length=length, bond=bl, symbol=symbol)
        label = label_template.format(
            natoms=len(self.atoms),
            total_length=self.atoms.cell[2, 2],
            diameter=self.atoms.cell[0, 0] / 2)
        self.description.text = label
        return py_template.format(n=n, m=m, length=length, symb=symbol, bl=bl)

    def apply(self):
        self.make()
        if self.atoms is not None:
            self.gui.new_atoms(self.atoms)
            return True
        else:
            ui.error(_('No valid atoms.'),
                     _('You have not (yet) specified a consistent '
                       'set of parameters.'))
            return False

    def ok(self, *args):
        if self.apply():
            self.win.close()
