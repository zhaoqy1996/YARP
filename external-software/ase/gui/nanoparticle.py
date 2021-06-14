# encoding: utf-8
"""nanoparticle.py - Window for setting up crystalline nanoparticles.
"""
from __future__ import division, unicode_literals
from copy import copy
from ase.gui.i18n import _

import numpy as np

import ase
import ase.data
import ase.gui.ui as ui

# Delayed imports:
# ase.cluster.data

from ase.cluster.cubic import FaceCenteredCubic, BodyCenteredCubic, SimpleCubic
from ase.cluster.hexagonal import HexagonalClosedPacked, Graphite
from ase.cluster import wulff_construction
from ase.gui.widgets import Element, pybutton


introtext = _("""\
Create a nanoparticle either by specifying the number of layers, or using the
Wulff construction.  Please press the [Help] button for instructions on how to
specify the directions.
WARNING: The Wulff construction currently only works with cubic crystals!
""")

helptext = _("""
The nanoparticle module sets up a nano-particle or a cluster with a given
crystal structure.

1) Select the element, the crystal structure and the lattice constant(s).
   The [Get structure] button will find the data for a given element.

2) Choose if you want to specify the number of layers in each direction, or if
   you want to use the Wulff construction.  In the latter case, you must
   specify surface energies in each direction, and the size of the cluster.

How to specify the directions:
------------------------------

First time a direction appears, it is interpreted as the entire family of
directions, i.e. (0,0,1) also covers (1,0,0), (-1,0,0) etc.  If one of these
directions is specified again, the second specification overrules that specific
direction.  For this reason, the order matters and you can rearrange the
directions with the [Up] and [Down] keys.  You can also add a new direction,
remember to press [Add] or it will not be included.

Example: (1,0,0) (1,1,1), (0,0,1) would specify the {100} family of directions,
the {111} family and then the (001) direction, overruling the value given for
the whole family of directions.
""")

py_template_layers = """
import ase
%(import)s

surfaces = %(surfaces)s
layers = %(layers)s
lc = %(latconst)s
atoms = %(factory)s('%(element)s', surfaces, layers, latticeconstant=lc)

# OPTIONAL: Cast to ase.Atoms object, discarding extra information:
# atoms = ase.Atoms(atoms)
"""

py_template_wulff = """
import ase
from ase.cluster import wulff_construction

surfaces = %(surfaces)s
esurf = %(energies)s
lc = %(latconst)s
size = %(natoms)s  # Number of atoms
atoms = wulff_construction('%(element)s', surfaces, esurf,
                           size, '%(structure)s',
                           rounding='%(rounding)s', latticeconstant=lc)

# OPTIONAL: Cast to ase.Atoms object, discarding extra information:
# atoms = ase.Atoms(atoms)
"""


class SetupNanoparticle:
    "Window for setting up a nanoparticle."
    # Structures:  Abbreviation, name,
    # 4-index (boolean), two lattice const (bool), factory
    structure_data = (('fcc', _('Face centered cubic (fcc)'),
                       False, False, FaceCenteredCubic),
                      ('bcc', _('Body centered cubic (bcc)'),
                       False, False, BodyCenteredCubic),
                      ('sc', _('Simple cubic (sc)'),
                       False, False, SimpleCubic),
                      ('hcp', _('Hexagonal closed-packed (hcp)'),
                       True, True, HexagonalClosedPacked),
                      ('graphite', _('Graphite'),
                       True, True, Graphite))
    # NB:  HCP is broken!

    # A list of import statements for the Python window.
    import_names = {
        'fcc': 'from ase.cluster.cubic import FaceCenteredCubic',
        'bcc': 'from ase.cluster.cubic import BodyCenteredCubic',
        'sc': 'from ase.cluster.cubic import SimpleCubic',
        'hcp': 'from ase.cluster.hexagonal import HexagonalClosedPacked',
        'graphite': 'from ase.cluster.hexagonal import Graphite'}

    # Default layer specifications for the different structures.
    default_layers = {'fcc': [((1, 0, 0), 6),
                              ((1, 1, 0), 9),
                              ((1, 1, 1), 5)],
                      'bcc': [((1, 0, 0), 6),
                              ((1, 1, 0), 9),
                              ((1, 1, 1), 5)],
                      'sc': [((1, 0, 0), 6),
                             ((1, 1, 0), 9),
                             ((1, 1, 1), 5)],
                      'hcp': [((0, 0, 0, 1), 5),
                              ((1, 0, -1, 0), 5)],
                      'graphite': [((0, 0, 0, 1), 5),
                                   ((1, 0, -1, 0), 5)]}

    def __init__(self, gui):
        self.atoms = None
        self.no_update = True
        self.old_structure = 'undefined'

        win = self.win = ui.Window(_('Nanoparticle'))
        win.add(ui.Text(introtext))

        self.element = Element('', self.apply)
        lattice_button = ui.Button(_('Get structure'),
                                   self.set_structure_data)
        self.elementinfo = ui.Label(' ')
        win.add(self.element)
        win.add(self.elementinfo)
        win.add(lattice_button)

        # The structure and lattice constant
        labels = []
        values = []
        self.needs_4index = {}
        self.needs_2lat = {}
        self.factory = {}
        for abbrev, name, n4, c, factory in self.structure_data:
            labels.append(name)
            values.append(abbrev)
            self.needs_4index[abbrev] = n4
            self.needs_2lat[abbrev] = c
            self.factory[abbrev] = factory
        self.structure = ui.ComboBox(labels, values, self.update_structure)
        win.add([_('Structure:'), self.structure])
        self.fourindex = self.needs_4index[values[0]]

        self.a = ui.SpinBox(3.0, 0.0, 1000.0, 0.01, self.update)
        self.c = ui.SpinBox(3.0, 0.0, 1000.0, 0.01, self.update)
        win.add([_('Lattice constant:  a ='), self.a, ' c =', self.c])

        # Choose specification method
        self.method = ui.ComboBox(
            [_('Layer specification'), _('Wulff construction')],
            ['layers', 'wulff'],
            self.update_gui_method)
        win.add([_('Method: '), self.method])

        self.layerlabel = ui.Label('Missing text')  # Filled in later
        win.add(self.layerlabel)
        self.direction_table_rows = ui.Rows()
        win.add(self.direction_table_rows)
        self.default_direction_table()

        win.add(_('Add new direction:'))
        self.new_direction_and_size_rows = ui.Rows()
        win.add(self.new_direction_and_size_rows)
        self.update_new_direction_and_size_stuff()

        # Information
        win.add(_('Information about the created cluster:'))
        self.info = [_('Number of atoms: '),
                     ui.Label('-'),
                     _('   Approx. diameter: '),
                     ui.Label('-')]
        win.add(self.info)

        # Finalize setup
        self.update_structure('fcc')
        self.update_gui_method()
        self.no_update = False

        self.auto = ui.CheckButton(_('Automatic Apply'))
        win.add(self.auto)

        win.add([pybutton(_('Creating a nanoparticle.'), self.makeatoms),
                 ui.helpbutton(helptext),
                 ui.Button(_('Apply'), self.apply),
                 ui.Button(_('OK'), self.ok)])

        self.gui = gui
        self.smaller_button = None
        self.largeer_button = None

        self.element.grab_focus()

    def default_direction_table(self):
        'Set default directions and values for the current crystal structure.'
        self.direction_table = []
        struct = self.structure.value
        for direction, layers in self.default_layers[struct]:
            self.direction_table.append((direction, layers, 1.0))

    def update_direction_table(self):
        self.direction_table_rows.clear()
        for direction, layers, energy in self.direction_table:
            self.add_direction(direction, layers, energy)
        self.update()

    def add_direction(self, direction, layers, energy):
        i = len(self.direction_table_rows)

        if self.method.value == 'wulff':
            spin = ui.SpinBox(energy, 0.0, 1000.0, 0.1, self.update)
        else:
            spin = ui.SpinBox(layers, 1, 100, 1, self.update)

        up = ui.Button(_('Up'), self.row_swap_next, i - 1)
        down = ui.Button(_('Down'), self.row_swap_next, i)
        delete = ui.Button(_('Delete'), self.row_delete, i)

        self.direction_table_rows.add([str(direction) + ':',
                                       spin, up, down, delete])
        up.active = i > 0
        down.active = False
        delete.active = i > 0

        if i > 0:
            down, delete = self.direction_table_rows[-2][3:]
            down.active = True
            delete.active = True

    def update_new_direction_and_size_stuff(self):
        if self.needs_4index[self.structure.value]:
            n = 4
        else:
            n = 3

        rows = self.new_direction_and_size_rows

        rows.clear()

        self.new_direction = row = ['(']
        for i in range(n):
            if i > 0:
                row.append(',')
            row.append(ui.SpinBox(0, -100, 100, 1))
        row.append('):')

        if self.method.value == 'wulff':
            row.append(ui.SpinBox(1.0, 0.0, 1000.0, 0.1))
        else:
            row.append(ui.SpinBox(5, 1, 100, 1))

        row.append(ui.Button(_('Add'), self.row_add))

        rows.add(row)

        if self.method.value == 'wulff':
            # Extra widgets for the Wulff construction
            self.size_radio = ui.RadioButtons(
                [_('Number of atoms'), _('Diameter')],
                ['natoms', 'diameter'],
                self.update_gui_size)
            self.size_natoms = ui.SpinBox(100, 1, 100000, 1,
                                          self.update_size_natoms)
            self.size_diameter = ui.SpinBox(5.0, 0, 100.0, 0.1,
                                            self.update_size_diameter)
            self.round_radio = ui.RadioButtons(
                [_('above  '), _('below  '), _('closest  ')],
                ['above', 'below', 'closest'],
                callback=self.update)
            self.smaller_button = ui.Button(_('Smaller'), self.wulff_smaller)
            self.larger_button = ui.Button(_('Larger'), self.wulff_larger)
            rows.add(_('Choose size using:'))
            rows.add(self.size_radio)
            rows.add([_('atoms'), self.size_natoms,
                      _(u'Å³'), self.size_diameter])
            rows.add(
                _('Rounding: If exact size is not possible, choose the size:'))
            rows.add(self.round_radio)
            rows.add([self.smaller_button, self.larger_button])
            self.update_gui_size()
        else:
            self.smaller_button = None
            self.larger_button = None

    def update_structure(self, s):
        'Called when the user changes the structure.'
        # s = self.structure.value
        if s != self.old_structure:
            old4 = self.fourindex
            self.fourindex = self.needs_4index[s]
            if self.fourindex != old4:
                # The table of directions is invalid.
                self.default_direction_table()
            self.old_structure = s
            self.c.active = self.needs_2lat[s]

        self.update()

    def update_gui_method(self, *args):
        'Switch between layer specification and Wulff construction.'
        self.update_direction_table()
        self.update_new_direction_and_size_stuff()
        if self.method.value == 'wulff':
            self.layerlabel.text = _(
                'Surface energies (as energy/area, NOT per atom):')
        else:
            self.layerlabel.text = _('Number of layers:')

        self.update()

    def wulff_smaller(self, widget=None):
        'Make a smaller Wulff construction.'
        n = len(self.atoms)
        self.size_radio.value = 'natoms'
        self.size_natoms.value = n - 1
        self.round_radio.value = 'below'
        self.apply()

    def wulff_larger(self, widget=None):
        'Make a larger Wulff construction.'
        n = len(self.atoms)
        self.size_radio.value = 'natoms'
        self.size_natoms.value = n + 1
        self.round_radio.value = 'above'
        self.apply()

    def row_add(self, widget=None):
        'Add a row to the list of directions.'
        if self.fourindex:
            n = 4
        else:
            n = 3
        idx = tuple(a.value for a in self.new_direction[1:1 + 2 * n:2])
        if not any(idx):
            ui.error(_('At least one index must be non-zero'), '')
            return
        if n == 4 and sum(idx) != 0:
            ui.error(_('Invalid hexagonal indices',
                       'The sum of the first three numbers must be zero'))
            return
        new = [idx, 5, 1.0]
        if self.method.value == 'wulff':
            new[1] = self.new_direction[-2].value
        else:
            new[2] = self.new_direction[-2].value
        self.direction_table.append(new)
        self.add_direction(*new)
        self.update()

    def row_delete(self, row):
        del self.direction_table[row]
        self.update_direction_table()

    def row_swap_next(self, row):
        dt = self.direction_table
        dt[row], dt[row + 1] = dt[row + 1], dt[row]
        self.update_direction_table()

    def update_gui_size(self, widget=None):
        'Update gui when the cluster size specification changes.'
        self.size_natoms.active = self.size_radio.value == 'natoms'
        self.size_diameter.active = self.size_radio.value == 'diameter'

    def update_size_natoms(self, widget=None):
        at_vol = self.get_atomic_volume()
        dia = 2.0 * (3 * self.size_natoms.value * at_vol /
                     (4 * np.pi))**(1 / 3)
        self.size_diameter.value = dia
        self.update()

    def update_size_diameter(self, widget=None, update=True):
        if self.size_diameter.active:
            at_vol = self.get_atomic_volume()
            n = round(np.pi / 6 * self.size_diameter.value**3 / at_vol)
            self.size_natoms.value = int(n)
            if update:
                self.update()

    def update(self, *args):
        if self.no_update:
            return
        self.element.Z  # Check
        if self.auto.value:
            self.makeatoms()
            if self.atoms is not None:
                self.gui.new_atoms(self.atoms)
        else:
            self.clearatoms()
        self.makeinfo()

    def set_structure_data(self, *args):
        'Called when the user presses [Get structure].'
        z = self.element.Z
        if z is None:
            return
        ref = ase.data.reference_states[z]
        if ref is None:
            structure = None
        else:
            structure = ref['symmetry']

        if ref is None or structure not in [s[0]
                                            for s in self.structure_data]:
            ui.error(_('Unsupported or unknown structure'),
                     _('Element = {0}, structure = {1}')
                     .format(self.element.symbol, structure))
            return

        self.structure.value = structure

        a = ref['a']
        self.a.value = a
        self.fourindex = self.needs_4index[structure]
        if self.fourindex:
            try:
                c = ref['c']
            except KeyError:
                c = ref['c/a'] * a
            self.c.value = c

    def makeatoms(self, *args):
        'Make the atoms according to the current specification.'
        symbol = self.element.symbol
        if symbol is None:
            self.clearatoms()
            self.makeinfo()
            return False
        struct = self.structure.value
        if self.needs_2lat[struct]:
            # a and c lattice constants
            lc = {'a': self.a.value,
                  'c': self.c.value}
            lc_str = str(lc)
        else:
            lc = self.a.value
            lc_str = '%.5f' % (lc,)
        if self.method.value == 'wulff':
            # Wulff construction
            surfaces = [x[0] for x in self.direction_table]
            surfaceenergies = [x[1].value
                               for x in self.direction_table_rows.rows]
            self.update_size_diameter(update=False)
            rounding = self.round_radio.value
            self.atoms = wulff_construction(symbol,
                                            surfaces,
                                            surfaceenergies,
                                            self.size_natoms.value,
                                            self.factory[struct],
                                            rounding, lc)
            python = py_template_wulff % {'element': symbol,
                                          'surfaces': str(surfaces),
                                          'energies': str(surfaceenergies),
                                          'latconst': lc_str,
                                          'natoms': self.size_natoms.value,
                                          'structure': struct,
                                          'rounding': rounding}
        else:
            # Layer-by-layer specification
            surfaces = [x[0] for x in self.direction_table]
            layers = [x[1].value for x in self.direction_table_rows.rows]
            self.atoms = self.factory[struct](symbol,
                                              copy(surfaces),
                                              layers, latticeconstant=lc)
            imp = self.import_names[struct]
            python = py_template_layers % {'import': imp,
                                           'element': symbol,
                                           'surfaces': str(surfaces),
                                           'layers': str(layers),
                                           'latconst': lc_str,
                                           'factory': imp.split()[-1]}
        self.makeinfo()

        return python

    def clearatoms(self):
        self.atoms = None

    def get_atomic_volume(self):
        s = self.structure.value
        a = self.a.value
        c = self.c.value
        if s == 'fcc':
            return a**3 / 4
        elif s == 'bcc':
            return a**3 / 2
        elif s == 'sc':
            return a**3
        elif s == 'hcp':
            return np.sqrt(3.0) / 2 * a * a * c / 2
        elif s == 'graphite':
            return np.sqrt(3.0) / 2 * a * a * c / 4

    def makeinfo(self):
        """Fill in information field about the atoms.

        Also turns the Wulff construction buttons [Larger] and
        [Smaller] on and off.
        """
        if self.atoms is None:
            self.info[1].text = '-'
            self.info[3].text = '-'
        else:
            at_vol = self.get_atomic_volume()
            dia = 2 * (3 * len(self.atoms) * at_vol / (4 * np.pi))**(1 / 3)
            self.info[1].text = str(len(self.atoms))
            self.info[3].text = u'{0:.1f} Å'.format(dia)

        if self.method.value == 'wulff':
            if self.smaller_button is not None:
                self.smaller_button.active = self.atoms is not None
                self.larger_button.active = self.atoms is not None

    def apply(self, callbackarg=None):
        self.makeatoms()
        if self.atoms is not None:
            self.gui.new_atoms(self.atoms)
            return True
        else:
            ui.error(_('No valid atoms.'),
                     _('You have not (yet) specified a consistent set of '
                       'parameters.'))
            return False

    def ok(self):
        if self.apply():
            self.win.close()
