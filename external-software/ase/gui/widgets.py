from __future__ import unicode_literals
from ase.gui.i18n import _

import ase.data
import ase.gui.ui as ui

from ase import Atoms
from ase.collections import g2


class Element(list):
    def __init__(self, symbol='', callback=None, allow_molecule=False):
        list.__init__(self,
                      [_('Element:'),
                       ui.Entry(symbol, 10 if allow_molecule else 3,
                                self.enter),
                       ui.Button(_('Help'), self.show_help),
                       ui.Label('', 'red')])
        self.callback = callback
        self.allow_molecule = allow_molecule

    def grab_focus(self):
        self[1].entry.focus_set()

    def show_help(self):
        names = []
        import re
        for name in g2.names:
            if not re.match('^[A-Z][a-z]?$', name):  # Not single atoms
                names.append(name)

        # This infobox is indescribably ugly because of the
        # ridiculously large font size used by Tkinter.  Ouch!
        msg = _('Enter a chemical symbol or the name of a molecule '
                'from the G2 testset:\n'
                '{}'.format(', '.join(names)))
        ui.showinfo('Info', msg)

    @property
    def Z(self):
        assert not self.allow_molecule
        atoms = self.get_atoms()
        if atoms is None:
            return None
        assert len(atoms) == 1
        return atoms.numbers[0]

    @property
    def symbol(self):
        Z = self.Z
        return None if Z is None else ase.data.chemical_symbols[Z]

    # Used by tests...
    @symbol.setter
    def symbol(self, value):
        self[1].value = value

    def get_atoms(self):
        val = self._get()
        if val is not None:
            self[2].text = ''
        return val

    def _get(self):
        txt = self[1].value

        if not txt:
            self.error(_('No element specified!'))
            return None

        if txt.isdigit():
            txt = int(txt)
            try:
                txt = ase.data.chemical_symbols[txt]
            except KeyError:
                self.error()
                return None

        if txt in ase.data.atomic_numbers:
            return Atoms(txt)

        if self.allow_molecule and g2.has(txt):
            return g2[txt]

        self.error()

    def enter(self):
        self.callback(self)

    def error(self, text=_('ERROR: Invalid element!')):
        self[2].text = text


def pybutton(title, callback):
    """A button for displaying Python code.

    When pressed, it opens a window displaying some Python code, or an error
    message if no Python code is ready.
    """
    return ui.Button('Python', pywindow, title, callback)


def pywindow(title, callback):
    code = callback()
    if code is None:
        ui.error(
            _('No Python code'),
            _('You have not (yet) specified a consistent set of parameters.'))
    else:
        win = ui.Window(title)
        win.add(ui.Text(code))
