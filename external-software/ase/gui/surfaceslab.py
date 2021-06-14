# encoding: utf-8
'''surfaceslab.py - Window for setting up surfaces
'''
from __future__ import division, unicode_literals
from ase.gui.i18n import _, ngettext

import ase.gui.ui as ui
import ase.build as build
from ase.data import reference_states
from ase.gui.widgets import Element, pybutton

introtext = _("""\
  Use this dialog to create surface slabs.  Select the element by
writing the chemical symbol or the atomic number in the box.  Then
select the desired surface structure.  Note that some structures can
be created with an othogonal or a non-orthogonal unit cell, in these
cases the non-orthogonal unit cell will contain fewer atoms.

  If the structure matches the experimental crystal structure, you can
look up the lattice constant, otherwise you have to specify it
yourself.""")

# Name, structure, orthogonal, function
surfaces = [(_('FCC(100)'), _('fcc'), 'ortho', build.fcc100),
            (_('FCC(110)'), _('fcc'), 'ortho', build.fcc110),
            (_('FCC(111)'), _('fcc'), 'both', build.fcc111),
            (_('FCC(211)'), _('fcc'), 'ortho', build.fcc211),
            (_('BCC(100)'), _('bcc'), 'ortho', build.bcc100),
            (_('BCC(110)'), _('bcc'), 'both', build.bcc110),
            (_('BCC(111)'), _('bcc'), 'both', build.bcc111),
            (_('HCP(0001)'), _('hcp'), 'both', build.hcp0001),
            (_('HCP(10-10)'), _('hcp'), 'ortho', build.hcp10m10),
            (_('DIAMOND(100)'), _('diamond'), 'ortho', build.diamond100),
            (_('DIAMOND(111)'), _('diamond'), 'non-ortho', build.diamond111)]

structures, crystal, orthogonal, functions = zip(*surfaces)

py_template = """
from ase.build import {func}

atoms = {func}(symbol='{symbol}', size={size},
    a={a}, {c}vacuum={vacuum}, orthogonal={ortho})
"""


class SetupSurfaceSlab:
    '''Window for setting up a surface.'''
    def __init__(self, gui):
        self.element = Element('', self.apply)
        self.structure = ui.ComboBox(structures, structures,
                                     self.structure_changed)
        self.structure_warn = ui.Label('', 'red')
        self.orthogonal = ui.CheckButton('', True, self.make)
        self.lattice_a = ui.SpinBox(3.2, 0.0, 10.0, 0.001, self.make)
        self.retrieve = ui.Button(_('Get from database'),
                                  self.structure_changed)
        self.lattice_c = ui.SpinBox(None, 0.0, 10.0, 0.001, self.make)
        self.x = ui.SpinBox(1, 1, 30, 1, self.make)
        self.x_warn = ui.Label('', 'red')
        self.y = ui.SpinBox(1, 1, 30, 1, self.make)
        self.y_warn = ui.Label('', 'red')
        self.z = ui.SpinBox(1, 1, 30, 1, self.make)
        self.vacuum_check = ui.CheckButton('', False, self.vacuum_checked)
        self.vacuum = ui.SpinBox(5, 0, 40, 0.01, self.make)
        self.description = ui.Label('')

        win = self.win = ui.Window(_('Surface'))
        win.add(ui.Text(introtext))
        win.add(self.element)
        win.add([_('Structure:'), self.structure, self.structure_warn])
        win.add([_('Orthogonal cell:'), self.orthogonal])
        win.add([_('Lattice constant:')])
        win.add([_('\ta'), self.lattice_a, (u'Å'), self.retrieve])
        win.add([_('\tc'), self.lattice_c, (u'Å')])
        win.add([_('Size:')])
        win.add([_('\tx: '), self.x, _(' unit cells'), self.x_warn])
        win.add([_('\ty: '), self.y, _(' unit cells'), self.y_warn])
        win.add([_('\tz: '), self.z, _(' unit cells')])
        win.add([_('Vacuum: '), self.vacuum_check, self.vacuum, (u'Å')])
        win.add(self.description)
        # TRANSLATORS: This is a title of a window.
        win.add([pybutton(_('Creating a surface.'), self.make),
                 ui.Button(_('Apply'), self.apply),
                 ui.Button(_('OK'), self.ok)])

        self.element.grab_focus()
        self.gui = gui
        self.atoms = None
        self.lattice_c.active = False
        self.vacuum.active = False
        self.structure_changed()

    def vacuum_checked(self, *args):
        if self.vacuum_check.var.get():
            self.vacuum.active = True
        else:
            self.vacuum.active = False
        self.make()

    def get_lattice(self, *args):
        if self.element.symbol is None:
            return
        ref = reference_states[self.element.Z]
        symmetry = "unknown"
        for struct in surfaces:
            if struct[0] == self.structure.value:
                symmetry = struct[1]
        if ref['symmetry'] != symmetry:
            # TRANSLATORS: E.g. "... assume fcc crystal structure for Au"
            self.structure_warn.text = (_('Error: Reference values assume {} '
                                          'crystal structure for {}!').
                                        format(ref['symmetry'],
                                               self.element.symbol))
        else:
            if symmetry == 'fcc' or symmetry == 'bcc' or symmetry == 'diamond':
                self.lattice_a.value = ref['a']
            elif symmetry == 'hcp':
                self.lattice_a.value = ref['a']
                self.lattice_c.value = ref['a'] * ref['c/a']
        self.make()

    def structure_changed(self, *args):
        for surface in surfaces:
            if surface[0] == self.structure.value:
                if surface[2] == 'ortho':
                    self.orthogonal.var.set(True)
                    self.orthogonal.check['state'] = ['disabled']
                elif surface[2] == 'non-ortho':
                    self.orthogonal.var.set(False)
                    self.orthogonal.check['state'] = ['disabled']
                else:
                    self.orthogonal.check['state'] = ['normal']

                if surface[1] == _('hcp'):
                    self.lattice_c.active = True
                    self.lattice_c.value = round(self.lattice_a.value *
                                                 ((8.0/3.0) ** (0.5)), 3)
                else:
                    self.lattice_c.active = False
                    self.lattice_c.value = 'None'
        self.get_lattice()

    def make(self, *args):
        symbol = self.element.symbol
        self.atoms = None
        self.description.text = ''
        self.python = None
        self.x_warn.text = ''
        self.y_warn.text = ''
        if symbol is None:
            return

        x = self.x.value
        y = self.y.value
        z = self.z.value
        size = (x, y, z)
        a = self.lattice_a.value
        c = self.lattice_c.value
        vacuum = self.vacuum.value
        if not self.vacuum_check.var.get():
            vacuum = None
        ortho = self.orthogonal.var.get()

        ortho_warn_even = _('Please enter an even value for orthogonal cell')

        struct = self.structure.value
        if struct == _('BCC(111)') and (not (y % 2 == 0) and ortho):
            self.y_warn.text = ortho_warn_even
            return
        if struct == _('BCC(110)') and (not (y % 2 == 0) and ortho):
            self.y_warn.text = ortho_warn_even
            return
        if struct == _('FCC(111)') and (not (y % 2 == 0) and ortho):
            self.y_warn.text = ortho_warn_even
            return
        if struct == _('FCC(211)') and (not (x % 3 == 0) and ortho):
            self.x_warn.text = _('Please enter a value divisible by 3'
                                 ' for orthogonal cell')
            return
        if struct == _('HCP(0001)') and (not (y % 2 == 0) and ortho):
            self.y_warn.text = ortho_warn_even
            return
        if struct == _('HCP(10-10)') and (not (y % 2 == 0) and ortho):
            self.y_warn.text = ortho_warn_even
            return

        for surface in surfaces:
            if surface[0] == struct:
                c_py = ""
                if surface[1] == _('hcp'):
                    self.atoms = surface[3](symbol, size, a, c, vacuum, ortho)
                    c_py = "{}, ".format(c)
                else:
                    self.atoms = surface[3](symbol, size, a, vacuum, ortho)

                if vacuum is not None:
                    vacuumtext =_(' Vacuum: {} Å.').format(vacuum)
                else:
                    vacuumtext = ''

                natoms = len(self.atoms)
                label = ngettext(
                    # TRANSLATORS: e.g. "Au fcc100 surface with 2 atoms."
                    # or "Au fcc100 surface with 2 atoms. Vacuum: 5 Å."
                    '{symbol} {surf} surface with one atom.{vacuum}',
                    '{symbol} {surf} surface with {natoms} atoms.{vacuum}',
                    natoms).format(symbol=symbol,
                                   surf=surface[3].__name__,
                                   natoms=natoms,
                                   vacuum=vacuumtext)

                self.description.text = label
                return py_template.format(func=surface[3].__name__, a=a,
                                          c=c_py, symbol=symbol, size=size,
                                          ortho=ortho, vacuum=vacuum)

    def apply(self, *args):
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
