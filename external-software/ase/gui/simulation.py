"Base class for simulation windows"

from __future__ import unicode_literals
from ase.gui.i18n import _

import ase.gui.ui as ui
#from ase import Atoms
#from ase.constraints import FixAtoms

raise NotImplementedError('Module not ported to tkinter')

pack = error = 42


class Simulation:
    def __init__(self, gui):
        ui.Window.__init__(self)
        self.gui = gui

    def packtext(self, vbox, text, label=None):
        "Pack an text frame into the window."
        pack(vbox, ui.Label(""))
        txtframe = ui.Frame(label)
        txtlbl = ui.Label(text)
        txtframe.add(txtlbl)
        txtlbl.show()
        pack(vbox, txtframe)
        pack(vbox, ui.Label(""))

    def packimageselection(self,
                           outerbox,
                           txt1=_(" (rerun simulation)"),
                           txt2=_(" (continue simulation)")):
        "Make the frame for selecting starting config if more than one."
        self.startframe = ui.Frame(_("Select starting configuration:"))
        pack(outerbox, [self.startframe])
        vbox = ui.VBox()
        self.startframe.add(vbox)
        vbox.show()
        self.numconfig_format = _("There are currently %i "
                                  "configurations loaded.")
        self.numconfig_label = ui.Label("")
        pack(vbox, [self.numconfig_label])
        lbl = ui.Label(
            _("Choose which one to use as the "
              "initial configuration"))
        pack(vbox, [lbl])
        self.start_radio_first = ui.RadioButton(
            None, _("The first configuration %s.") % txt1)
        pack(vbox, [self.start_radio_first])
        self.start_radio_nth = ui.RadioButton(self.start_radio_first,
                                              _("Configuration number "))
        self.start_nth_adj = ui.Adjustment(0, 0, 1, 1)
        self.start_nth_spin = ui.SpinButton(self.start_nth_adj, 0, 0)
        self.start_nth_spin.set_sensitive(False)
        pack(vbox, [self.start_radio_nth, self.start_nth_spin])
        self.start_radio_last = ui.RadioButton(
            self.start_radio_first, _("The last configuration %s.") % txt2)
        self.start_radio_last.set_active(True)
        pack(vbox, self.start_radio_last)
        self.start_radio_nth.connect("toggled", self.start_radio_nth_toggled)
        self.setupimageselection()

    def start_radio_nth_toggled(self, widget):
        self.start_nth_spin.set_sensitive(self.start_radio_nth.get_active())

    def setupimageselection(self):
        "Decide if the start image selection frame should be shown."
        n = len(self.gui.images)
        if n <= 1:
            self.startframe.hide()
        else:
            self.startframe.show()
            if self.start_nth_adj.value >= n:
                self.start_nth_adj.value = n - 1
            self.start_nth_adj.upper = n - 1
            self.numconfig_label.set_text(self.numconfig_format % (n, ))

    def getimagenumber(self):
        "Get the image number selected in the start image frame."
        nmax = len(self.gui.images)
        if nmax <= 1:
            return 0
        elif self.start_radio_first.get_active():
            return 0
        elif self.start_radio_nth.get_active():
            return self.start_nth_adj.value
        else:
            assert self.start_radio_last.get_active()
            return nmax - 1

    def makebutbox(self, vbox, helptext=None):
        self.buttons = ui.HButtonBox()
        runbut = ui.Button(_("Run"))
        runbut.connect('clicked', self.run)
        closebut = ui.Button('Close')
        closebut.connect('clicked', lambda x: self.destroy())
        for w in (runbut, closebut):
            self.buttons.pack_start(w, 0, 0)
            w.show()
        if helptext:
            helpbut = [help(helptext)]
        else:
            helpbut = []
        pack(vbox, helpbut + [self.buttons], end=True, bottom=True)

    def setup_atoms(self):
        self.atoms = self.get_atoms()
        if self.atoms is None:
            return False
        try:
            self.calculator = self.gui.simulation['calc']
        except KeyError:
            error(_("No calculator: Use Calculate/Set Calculator on the menu."))
            return False
        self.atoms.set_calculator(self.calculator())
        return True

    def get_atoms(self):
        "Make an atoms object from the active image"
        images = self.gui.images
        atoms = images[self.getimagenumber()]
        natoms = len(atoms) // images.repeat.prod()
        if natoms < 1:
            error(_("No atoms present"))
            return None
        return atoms[:natoms]

    def begin(self, **kwargs):
        if 'progress' in self.gui.simulation:
            self.gui.simulation['progress'].begin(**kwargs)

    def end(self):
        if 'progress' in self.gui.simulation:
            self.gui.simulation['progress'].end()

    def prepare_store_atoms(self):
        "Informs the gui that the next configuration should be the first."
        self.gui.prepare_new_atoms()
        self.count_steps = 0

    def store_atoms(self):
        "Observes the minimization and stores the atoms in the gui."
        self.gui.append_atoms(self.atoms)
        self.count_steps += 1
