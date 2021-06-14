# encoding: utf-8

"Module for performing energy minimization."

import ase.gui.ui as ui
from ase.gui.simulation import Simulation
import ase
import ase.optimize


pack = _ = AseGuiCancelException = 42


class MinimizeMixin:
    minimizers = ('BFGS', 'BFGSLineSearch', 'LBFGS', 'LBFGSLineSearch',
                  'MDMin', 'FIRE')

    def make_minimize_gui(self, box):
        self.algo = ui.combo_box_new_text()
        for m in self.minimizers:
            self.algo.append_text(m)
        self.algo.set_active(0)
        self.algo.connect('changed', self.min_algo_specific)
        pack(box, [ui.Label(_("Algorithm: ")), self.algo])

        self.fmax = ui.Adjustment(0.05, 0.00, 10.0, 0.01)
        self.fmax_spin = ui.SpinButton(self.fmax, 0, 3)
        lbl = ui.Label()
        lbl.set_markup(_("Convergence criterion: F<sub>max</sub> = "))
        pack(box, [lbl, self.fmax_spin])

        self.steps = ui.Adjustment(100, 1, 1000000, 1)
        self.steps_spin = ui.SpinButton(self.steps, 0, 0)
        pack(box, [ui.Label(_("Max. number of steps: ")), self.steps_spin])

        # Special stuff for MDMin
        lbl = ui.Label(_("Pseudo time step: "))
        self.mdmin_dt = ui.Adjustment(0.05, 0.0, 10.0, 0.01)
        spin = ui.SpinButton(self.mdmin_dt, 0, 3)
        self.mdmin_widgets = [lbl, spin]
        pack(box, self.mdmin_widgets)
        self.min_algo_specific()

    def min_algo_specific(self, *args):
        "SHow or hide algorithm-specific widgets."
        minimizer = self.minimizers[self.algo.get_active()]
        for w in self.mdmin_widgets:
            if minimizer == 'MDMin':
                w.show()
            else:
                w.hide()


class Minimize(Simulation, MinimizeMixin):
    "Window for performing energy minimization."

    def __init__(self, gui):
        Simulation.__init__(self, gui)
        self.set_title(_("Energy minimization"))

        vbox = ui.VBox()
        self.packtext(vbox,
                      _("Minimize the energy with respect to the positions."))
        self.packimageselection(vbox)
        pack(vbox, ui.Label(""))

        self.make_minimize_gui(vbox)

        pack(vbox, ui.Label(""))
        self.status_label = ui.Label("")
        pack(vbox, [self.status_label])
        self.makebutbox(vbox)
        vbox.show()
        self.add(vbox)
        self.show()
        self.gui.register_vulnerable(self)

    def run(self, *args):
        "User has pressed [Run]: run the minimization."
        if not self.setup_atoms():
            return
        fmax = self.fmax.value
        steps = self.steps.value
        mininame = self.minimizers[self.algo.get_active()]
        self.begin(mode="min", algo=mininame, fmax=fmax, steps=steps)
        algo = getattr(ase.optimize, mininame)
        try:
            logger_func = self.gui.simulation['progress'].get_logger_stream
        except (KeyError, AttributeError):
            logger = None
        else:
            logger = logger_func()  # Don't catch errors in the function.

        # Display status message
        self.status_label.set_text(_("Running ..."))
        self.status_label.modify_fg(ui.STATE_NORMAL,
                                    '#AA0000')
        while ui.events_pending():
            ui.main_iteration()

        self.prepare_store_atoms()
        if mininame == "MDMin":
            minimizer = algo(self.atoms, logfile=logger,
                             dt=self.mdmin_dt.value)
        else:
            minimizer = algo(self.atoms, logfile=logger)
        minimizer.attach(self.store_atoms)
        try:
            minimizer.run(fmax=fmax, steps=steps)
        except AseGuiCancelException:
            # Update display to reflect cancellation of simulation.
            self.status_label.set_text(_("Minimization CANCELLED after "
                                         "%i steps.")
                                       % (self.count_steps,))
            self.status_label.modify_fg(ui.STATE_NORMAL,
                                        '#AA4000')
        except MemoryError:
            self.status_label.set_text(_("Out of memory, consider using "
                                         "LBFGS instead"))
            self.status_label.modify_fg(ui.STATE_NORMAL,
                                        '#AA4000')

        else:
            # Update display to reflect successful end of simulation.
            self.status_label.set_text(_("Minimization completed in %i steps.")
                                       % (self.count_steps,))
            self.status_label.modify_fg(ui.STATE_NORMAL,
                                        '#007700')

        self.end()
        if self.count_steps:
            # Notify other windows that atoms have changed.
            # This also notifies this window!
            self.gui.notify_vulnerable()

        # Open movie window and energy graph
        # XXX disabled 2018-10-19.  --askhl
        #if self.gui.images.nimages > 1:
        #    self.gui.movie()
        #    assert not np.isnan(self.gui.images.E[0])
        #    if not self.gui.plot_graphs_newatoms():
        #        expr = 'i, e - E[-1]'
        #        self.gui.plot_graphs(expr=expr)

    def notify_atoms_changed(self):
        "When atoms have changed, check for the number of images."
        self.setupimageselection()
