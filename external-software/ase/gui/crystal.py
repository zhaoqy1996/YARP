# encoding: utf-8
"""crystal.py - Window for setting up arbitrary crystal lattices
"""

from __future__ import unicode_literals
from ase.gui.i18n import _

import ase.gui.ui as ui
from ase.gui.status import formula
from ase.spacegroup import crystal, Spacegroup

import ase

pack = error = cancel_apply_ok = PyButton = SetupWindow = 42

introtext = _("""\
  Use this dialog to create crystal lattices. First select the structure,
  either from a set of common crystal structures, or by space group description.
  Then add all other lattice parameters.

  If an experimental crystal structure is available for an atom, you can
  look up the crystal type and lattice constant, otherwise you have to specify it
  yourself.  """)

py_template = """
from ase.spacegroup import crystal

atoms = crystal(spacegroup=%(spacegroup)d,
                symbols=%(symbols)s,
                basis=%(basis)s,
                cellpar=%(cellpar)s)
"""
label_template = _(
    """ %(natoms)i atoms: %(symbols)s, Volume: %(volume).3f A<sup>3</sup>""")

# all predefined crystals go into tuples here:
# (selection name, spacegroup, group_active, [repeats], [a,b,c,alpha,beta,gamma],[lattice constraints],[constraints_active],basis)
crystal_definitions = [
    ('Spacegroup', 1, True, [1, 1, 1], [3.0, 3.0, 3.0, 90.0, 90.0, 90.0],
     [0, 0, 0, 0, 0, 0], [True, True, True, True, True, True],
     [['', '', '', '']]),
    ('fcc', 225, False, [1, 1, 1], [3.0, 3.0, 3.0, 90.0, 90.0, 90.0],
     [0, 1, 1, 3, 3, 3], [False, False, False, False, False, False],
     [['', '', '', '']]),
    ('bcc', 229, False, [1, 1, 1], [3.0, 3.0, 3.0, 90.0, 90.0, 90.0],
     [0, 1, 1, 3, 3, 3], [False, False, False, False, False, False],
     [['', '', '', '']]), (
         'diamond', 227, False, [1, 1, 1], [3.0, 3.0, 3.0, 90.0, 90.0, 90.0],
         [0, 1, 1, 3, 3, 3], [False, False, False, False, False, False],
         [['', '', '', '']]),
    ('hcp', 194, False, [1, 1, 1], [3.0, 3.0, 3.0, 90.0, 90.0, 120.0],
     [0, 1, 0, 3, 3, 3], [False, False, False, False, False, False],
     [['', '1./3.', '2./3.', '3./4.']]), (
         'graphite', 186, False, [1, 1, 1], [3.0, 3.0, 3.0, 90.0, 90.0, 120.0],
         [0, 1, 0, 3, 3, 3], [False, False, False, False, False, False],
         [['', '0', '0', '0'], ['', '1./3.', '2./3.', '0']]),
    ('rocksalt', 225, False, [1, 1, 1], [3.0, 3.0, 3.0, 90.0, 90.0, 90.0],
     [0, 1, 1, 3, 3, 3], [False, False, False, False, False, False],
     [['', '0', '0', '0'], ['', '0.5', '0.5', '0.5']]), (
         'rutile', 136, False, [1, 1, 1], [3.0, 3.0, 3.0, 90.0, 90.0, 90.0],
         [0, 1, 0, 3, 3, 3], [False, False, False, False, False, False],
         [['', '0', '0', '0'], ['O', '0.3', '0.3', '0']])
]


class SetupBulkCrystal:
    """Window for setting up a surface."""

    def __init__(self, gui):
        SetupWindow.__init__(self)
        self.set_title(_("Create Bulk Crystal by Spacegroup"))
        self.atoms = None
        vbox = ui.VBox()
        self.packtext(vbox, introtext)
        self.structinfo = ui.combo_box_new_text()
        self.structures = {}
        for c in crystal_definitions:
            self.structinfo.append_text(c[0])
            self.structures[c[0]] = c
        self.structinfo.set_active(0)
        self.structinfo.connect("changed", self.set_lattice_type)
        self.spacegroup = ui.Entry(max=14)
        self.spacegroup.set_text('P 1')
        self.elementinfo = ui.Label("")
        self.spacegroupinfo = ui.Label(_('Number: 1'))
        pack(vbox, [
            ui.Label(_("Lattice: ")), self.structinfo,
            ui.Label(_("\tSpace group: ")), self.spacegroup, ui.Label('  '),
            self.spacegroupinfo, ui.Label('  '), self.elementinfo
        ])
        pack(vbox, [ui.Label("")])
        self.size = [ui.Adjustment(1, 1, 100, 1) for i in range(3)]
        buttons = [ui.SpinButton(s, 0, 0) for s in self.size]
        pack(vbox, [
            ui.Label(_("Size: x: ")), buttons[0], ui.Label(_("  y: ")),
            buttons[1], ui.Label(_("  z: ")), buttons[2],
            ui.Label(_(" unit cells"))
        ])
        pack(vbox, [ui.Label("")])
        self.lattice_lengths = [
            ui.Adjustment(3.0, 0.0, 1000.0, 0.01) for i in range(3)
        ]
        self.lattice_angles = [
            ui.Adjustment(90.0, 0.0, 180.0, 1) for i in range(3)
        ]
        self.lattice_lbuts = [
            ui.SpinButton(self.lattice_lengths[i], 0, 0) for i in range(3)
        ]
        self.lattice_abuts = [
            ui.SpinButton(self.lattice_angles[i], 0, 0) for i in range(3)
        ]
        for i in self.lattice_lbuts:
            i.set_digits(5)
        for i in self.lattice_abuts:
            i.set_digits(3)
        self.lattice_lequals = [ui.combo_box_new_text() for i in range(3)]
        self.lattice_aequals = [ui.combo_box_new_text() for i in range(3)]
        self.lattice_lequals[0].append_text(_('free'))
        self.lattice_lequals[0].append_text(_('equals b'))
        self.lattice_lequals[0].append_text(_('equals c'))
        self.lattice_lequals[0].append_text(_('fixed'))
        self.lattice_lequals[1].append_text(_('free'))
        self.lattice_lequals[1].append_text(_('equals a'))
        self.lattice_lequals[1].append_text(_('equals c'))
        self.lattice_lequals[1].append_text(_('fixed'))
        self.lattice_lequals[2].append_text(_('free'))
        self.lattice_lequals[2].append_text(_('equals a'))
        self.lattice_lequals[2].append_text(_('equals b'))
        self.lattice_lequals[2].append_text(_('fixed'))
        self.lattice_aequals[0].append_text(_('free'))
        self.lattice_aequals[0].append_text(_('equals beta'))
        self.lattice_aequals[0].append_text(_('equals gamma'))
        self.lattice_aequals[0].append_text(_('fixed'))
        self.lattice_aequals[1].append_text(_('free'))
        self.lattice_aequals[1].append_text(_('equals alpha'))
        self.lattice_aequals[1].append_text(_('equals gamma'))
        self.lattice_aequals[1].append_text(_('fixed'))
        self.lattice_aequals[2].append_text(_('free'))
        self.lattice_aequals[2].append_text(_('equals alpha'))
        self.lattice_aequals[2].append_text(_('equals beta'))
        self.lattice_aequals[2].append_text(_('fixed'))
        for i in range(3):
            self.lattice_lequals[i].set_active(0)
            self.lattice_aequals[i].set_active(0)
        pack(vbox, [ui.Label(_('Lattice parameters'))])
        pack(vbox, [
            ui.Label(_('\t\ta:\t')), self.lattice_lbuts[0], ui.Label('  '),
            self.lattice_lequals[0], ui.Label(_('\talpha:\t')),
            self.lattice_abuts[0], ui.Label('  '), self.lattice_aequals[0]
        ])
        pack(vbox, [
            ui.Label(_('\t\tb:\t')), self.lattice_lbuts[1], ui.Label('  '),
            self.lattice_lequals[1], ui.Label(_('\tbeta:\t')),
            self.lattice_abuts[1], ui.Label('  '), self.lattice_aequals[1]
        ])
        pack(vbox, [
            ui.Label(_('\t\tc:\t')), self.lattice_lbuts[2], ui.Label('  '),
            self.lattice_lequals[2], ui.Label(_('\tgamma:\t')),
            self.lattice_abuts[2], ui.Label('  '), self.lattice_aequals[2]
        ])
        self.get_data = ui.Button(_("Get from database"))
        self.get_data.connect("clicked", self.get_from_database)
        self.get_data.set_sensitive(False)
        pack(vbox, [ui.Label("     "), self.get_data])
        pack(vbox, [ui.Label("")])
        pack(vbox, [ui.Label(_("Basis: "))])
        self.elements = [[
            ui.Entry(max=3), ui.Entry(max=8), ui.Entry(max=8), ui.Entry(max=8),
            True
        ]]
        self.element = self.elements[0][0]
        add_atom = ui.Button(stock='Add')
        add_atom.connect("clicked", self.add_basis_atom)
        add_atom.connect("activate", self.add_basis_atom)
        pack(vbox, [
            ui.Label(_('  Element:\t')), self.elements[0][0],
            ui.Label(_('\tx: ')), self.elements[0][1], ui.Label(_('  y: ')),
            self.elements[0][2], ui.Label(_('  z: ')), self.elements[0][3],
            ui.Label('\t'), add_atom
        ])
        self.vbox_basis = ui.VBox()
        swin = ui.ScrolledWindow()
        swin.set_border_width(0)
        swin.set_policy(ui.POLICY_AUTOMATIC, ui.POLICY_AUTOMATIC)
        vbox.pack_start(swin, True, True, 0)
        swin.add_with_viewport(self.vbox_basis)
        self.vbox_basis.get_parent().set_shadow_type(ui.SHADOW_NONE)
        self.vbox_basis.get_parent().set_size_request(-1, 100)
        swin.show()

        pack(self.vbox_basis, [ui.Label('')])
        pack(vbox, [self.vbox_basis])
        self.vbox_basis.show()
        pack(vbox, [ui.Label("")])
        self.status = ui.Label("")
        pack(vbox, [self.status])
        pack(vbox, [ui.Label("")])
        self.pybut = PyButton(_("Creating a crystal."))
        self.pybut.connect('clicked', self.update)

        clear = ui.Button(stock='Clear')
        clear.connect("clicked", self.clear)
        buts = cancel_apply_ok(
            cancel=lambda widget: self.destroy(), apply=self.apply, ok=self.ok)
        pack(vbox, [self.pybut, clear, buts], end=True, bottom=True)
        self.structinfo.connect("changed", self.update)
        self.spacegroup.connect("activate", self.update)
        for s in self.size:
            s.connect("value-changed", self.update)
        for el in self.elements:
            if el[-1]:
                for i in el[:-1]:
                    i.connect("activate", self.update)
                    i.connect("changed", self.update)
        for i in range(3):
            self.lattice_lbuts[i].connect("value-changed", self.update)
            self.lattice_abuts[i].connect("value-changed", self.update)
            self.lattice_lequals[i].connect("changed", self.update)
            self.lattice_aequals[i].connect("changed", self.update)
        self.clearing_in_process = False
        self.gui = gui
        self.add(vbox)
        vbox.show()
        self.show()

    def update(self, *args):
        """ all changes of physical constants are handled here, atoms are set up"""
        if self.clearing_in_process:
            return True
        self.update_element()
        a_equals = self.lattice_lequals[0].get_active()
        b_equals = self.lattice_lequals[1].get_active()
        c_equals = self.lattice_lequals[2].get_active()
        alpha_equals = self.lattice_aequals[0].get_active()
        beta_equals = self.lattice_aequals[1].get_active()
        gamma_equals = self.lattice_aequals[2].get_active()
        sym = self.spacegroup.get_text()
        valid = True
        try:
            no = int(sym)
            spg = Spacegroup(no).symbol
            self.spacegroupinfo.set_label(_('Symbol: %s') % str(spg))
            spg = no
        except:
            try:
                no = Spacegroup(sym).no
                self.spacegroupinfo.set_label(_('Number: %s') % str(no))
                spg = no
            except:
                self.spacegroupinfo.set_label(_('Invalid Spacegroup!'))
                valid = False

        if a_equals == 0:
            self.lattice_lbuts[0].set_sensitive(True)
        elif a_equals == 1:
            self.lattice_lbuts[0].set_sensitive(False)
            self.lattice_lbuts[0].set_value(self.lattice_lbuts[1].get_value())
        elif a_equals == 2:
            self.lattice_lbuts[0].set_sensitive(False)
            self.lattice_lbuts[0].set_value(self.lattice_lbuts[2].get_value())
        else:
            self.lattice_lbuts[0].set_sensitive(False)
        if b_equals == 0:
            self.lattice_lbuts[1].set_sensitive(True)
        elif b_equals == 1:
            self.lattice_lbuts[1].set_sensitive(False)
            self.lattice_lbuts[1].set_value(self.lattice_lbuts[0].get_value())
        elif b_equals == 2:
            self.lattice_lbuts[1].set_sensitive(False)
            self.lattice_lbuts[1].set_value(self.lattice_lbuts[2].get_value())
        else:
            self.lattice_lbuts[1].set_sensitive(False)
        if c_equals == 0:
            self.lattice_lbuts[2].set_sensitive(True)
        elif c_equals == 1:
            self.lattice_lbuts[2].set_sensitive(False)
            self.lattice_lbuts[2].set_value(self.lattice_lbuts[0].get_value())
        elif c_equals == 2:
            self.lattice_lbuts[2].set_sensitive(False)
            self.lattice_lbuts[2].set_value(self.lattice_lbuts[1].get_value())
        else:
            self.lattice_lbuts[2].set_sensitive(False)
        if alpha_equals == 0:
            self.lattice_abuts[0].set_sensitive(True)
        elif alpha_equals == 1:
            self.lattice_abuts[0].set_sensitive(False)
            self.lattice_abuts[0].set_value(self.lattice_abuts[1].get_value())
        elif alpha_equals == 2:
            self.lattice_abuts[0].set_sensitive(False)
            self.lattice_abuts[0].set_value(self.lattice_abuts[2].get_value())
        else:
            self.lattice_abuts[0].set_sensitive(False)
        if beta_equals == 0:
            self.lattice_abuts[1].set_sensitive(True)
        elif beta_equals == 1:
            self.lattice_abuts[1].set_sensitive(False)
            self.lattice_abuts[1].set_value(self.lattice_abuts[0].get_value())
        elif beta_equals == 2:
            self.lattice_abuts[1].set_sensitive(False)
            self.lattice_abuts[1].set_value(self.lattice_abuts[2].get_value())
        else:
            self.lattice_abuts[1].set_sensitive(False)
        if gamma_equals == 0:
            self.lattice_abuts[2].set_sensitive(True)
        elif gamma_equals == 1:
            self.lattice_abuts[2].set_sensitive(False)
            self.lattice_abuts[2].set_value(self.lattice_abuts[0].get_value())
        elif gamma_equals == 2:
            self.lattice_abuts[2].set_sensitive(False)
            self.lattice_abuts[2].set_value(self.lattice_abuts[1].get_value())
        else:
            self.lattice_abuts[2].set_sensitive(False)

        valid = len(self.elements[0][0].get_text()) and valid
        self.get_data.set_sensitive(valid and self.get_n_elements() == 1 and
                                    self.update_element())
        self.atoms = None
        if valid:
            basis_count = -1
            for el in self.elements:
                if el[-1]:
                    basis_count += 1
            if basis_count:
                symbol_str = '['
                basis_str = "["
                symbol = []
                basis = []
            else:
                symbol_str = ''
                basis_str = ''
                basis = None
            for el in self.elements:
                if el[-1]:
                    symbol_str += "'" + el[0].get_text() + "'"
                    if basis_count:
                        symbol_str += ','
                        symbol += [el[0].get_text()]
                        exec('basis += [[float(' + el[1].get_text(
                        ) + '),float(' + el[2].get_text() + '),float(' + el[3]
                             .get_text() + ')]]')
                    else:
                        symbol = el[0].get_text()
                        exec('basis = [[float(' + el[1].get_text() + '),float('
                             + el[2].get_text() + '),float(' + el[3].get_text(
                             ) + ')]]')
                    basis_str += '[' + el[1].get_text() + ',' + el[2].get_text(
                    ) + ',' + el[3].get_text() + '],'
            basis_str = basis_str[:-1]
            if basis_count:
                symbol_str = symbol_str[:-1] + ']'
                basis_str += ']'
            size_str = '(' + str(int(self.size[0].get_value())) + ',' + str(
                int(self.size[1].get_value())) + ',' + str(
                    int(self.size[2].get_value())) + ')'
            size = (int(self.size[0].get_value()),
                    int(self.size[1].get_value()),
                    int(self.size[2].get_value()))
            cellpar_str = ''
            cellpar = []
            for i in self.lattice_lbuts:
                cellpar_str += str(i.get_value()) + ','
                cellpar += [i.get_value()]
            for i in self.lattice_abuts:
                cellpar_str += str(i.get_value()) + ','
                cellpar += [i.get_value()]
            cellpar_str = '[' + cellpar_str[:-1] + ']'
            args = {
                'symbols': symbol,
                'basis': basis,
                'size': size,
                'spacegroup': spg,
                'cellpar': cellpar
            }
            args_str = {
                'symbols': symbol_str,
                'basis': basis_str,
                'size': size_str,
                'spacegroup': spg,
                'cellpar': cellpar_str
            }
            self.pybut.python = py_template % args_str
            try:
                self.atoms = crystal(**args)
                label = label_template % {
                    'natoms': len(self.atoms),
                    'symbols': formula(self.atoms.get_atomic_numbers()),
                    'volume': self.atoms.get_volume()
                }
                self.status.set_label(label)
            except:
                self.atoms = None
                self.status.set_markup(
                    _("Please specify a consistent set of atoms."))
        else:
            self.atoms = None
            self.status.set_markup(
                _("Please specify a consistent set of atoms."))

    def apply(self, *args):
        """ create gui atoms from currently active atoms"""
        self.update()
        if self.atoms is not None:
            self.gui.new_atoms(self.atoms)
            return True
        else:
            error(
                _('No valid atoms.'),
                _('You have not (yet) specified a consistent set of '
                  'parameters.'))
            return False

    def ok(self, *args):
        if self.apply():
            self.destroy()

    def add_basis_atom(self, *args):
        """ add an atom to the customizable basis """
        n = len(self.elements)
        self.elements += [[
            ui.Entry(max=3), ui.Entry(max=8), ui.Entry(max=8), ui.Entry(max=8),
            ui.Label('\t\t\t'), ui.Label('\tx: '), ui.Label('  y: '),
            ui.Label('  z: '), ui.Label(' '), ui.Button('Delete'), True
        ]]
        self.elements[n][-2].connect("clicked", self.delete_basis_atom,
                                     {'n': n})
        pack(self.vbox_basis, [
            self.elements[n][4], self.elements[n][0], self.elements[n][5],
            self.elements[n][1], self.elements[n][6], self.elements[n][2],
            self.elements[n][7], self.elements[n][3], self.elements[n][8],
            self.elements[n][9]
        ])
        self.update()

    def delete_basis_atom(self, button, index, *args):
        """ delete atom index from customizable basis"""
        n = index['n']
        self.elements[n][-1] = False
        for i in range(10):
            self.elements[n][i].destroy()
        self.update()

    def get_n_elements(self):
        """ counts how many basis atoms are actually active """
        n = 0
        for el in self.elements:
            if el[-1]:
                n += 1
        return n

    def clear(self, *args):
        """ reset to original state """
        self.clearing_in_process = True
        self.clear_lattice()
        self.structinfo.set_active(0)
        self.set_lattice_type()
        self.clearing_in_process = False
        self.update()

    def clear_lattice(self, *args):
        """ delete all custom settings """
        self.atoms = None
        if len(self.elements) > 1:
            for n, el in enumerate(self.elements[1:]):
                self.elements[n + 1][-1] = False
                for i in range(10):
                    self.elements[n + 1][i].destroy()
        for i in range(4):
            self.elements[0][i].set_text("")
        self.spacegroup.set_sensitive(True)
        for i in self.lattice_lbuts:
            i.set_sensitive(True)
        for i in self.lattice_abuts:
            i.set_sensitive(True)
        for i in range(3):
            self.lattice_lequals[i].set_sensitive(True)
            self.lattice_aequals[i].set_sensitive(True)
            self.lattice_lequals[i].set_active(0)
            self.lattice_aequals[i].set_active(0)
        for s in self.size:
            s.set_value(1)

    def set_lattice_type(self, *args):
        """ set defaults from original """
        self.clearing_in_process = True
        self.clear_lattice()
        lattice = crystal_definitions[self.structinfo.get_active()]
        self.spacegroup.set_text(str(lattice[1]))
        self.spacegroup.set_sensitive(lattice[2])
        for s, i in zip(self.size, lattice[3]):
            s.set_value(i)
        self.lattice_lbuts[0].set_value(lattice[4][0])
        self.lattice_lbuts[1].set_value(lattice[4][1])
        self.lattice_lbuts[2].set_value(lattice[4][2])
        self.lattice_abuts[0].set_value(lattice[4][3])
        self.lattice_abuts[1].set_value(lattice[4][4])
        self.lattice_abuts[2].set_value(lattice[4][5])
        self.lattice_lequals[0].set_active(lattice[5][0])
        self.lattice_lequals[1].set_active(lattice[5][1])
        self.lattice_lequals[2].set_active(lattice[5][2])
        self.lattice_aequals[0].set_active(lattice[5][3])
        self.lattice_aequals[1].set_active(lattice[5][4])
        self.lattice_aequals[2].set_active(lattice[5][5])
        self.lattice_lequals[0].set_sensitive(lattice[6][0])
        self.lattice_lequals[1].set_sensitive(lattice[6][1])
        self.lattice_lequals[2].set_sensitive(lattice[6][2])
        self.lattice_aequals[0].set_sensitive(lattice[6][3])
        self.lattice_aequals[1].set_sensitive(lattice[6][4])
        self.lattice_aequals[2].set_sensitive(lattice[6][5])
        for n, at in enumerate(lattice[7]):
            l = 0
            if n > 0:
                l = len(self.elements)
                self.add_basis_atom()
            for i, s in enumerate(at):
                self.elements[l][i].set_text(s)
        self.clearing_in_process = False
        self.update()

    def get_from_database(self, *args):
        element = self.elements[0][0].get_text()
        z = ase.data.atomic_numbers[self.legal_element]
        ref = ase.data.reference_states[z]
        lattice = ref['symmetry']
        index = 0
        while index < len(crystal_definitions) and crystal_definitions[index][
                0] != lattice:
            index += 1
        if index == len(crystal_definitions) or not self.legal_element:
            error(_("Can't find lattice definition!"))
            return False
        self.structinfo.set_active(index)
        self.lattice_lbuts[0].set_value(ref['a'])
        if lattice == 'hcp':
            self.lattice_lbuts[2].set_value(ref['c/a'] * ref['a'])
        self.elements[0][0].set_text(element)
        if lattice in ['fcc', 'bcc', 'diamond']:
            self.elements[0][1].set_text('0')
            self.elements[0][2].set_text('0')
            self.elements[0][3].set_text('0')
