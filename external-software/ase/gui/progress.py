from __future__ import print_function
# encoding: utf-8

import ase.gui.ui as ui
import numpy as np
import sys
import re
import time

_ = pack = AseGuiCancelException = 42


class DummyProgressIndicator:
    def begin(self, **kwargs):
        pass

    def end(self):
        pass


class DefaultProgressIndicator:
    "Window for reporting progress."
    waittime = 3  # time (in sec) after which a progress bar appears
    updatetime = 0.1  # min time (in sec) between updates of the progress bars

    def __init__(self):
        ui.Window.__init__(self)
        self.set_title(_("Progress"))
        self.globalbox = ui.VBox()
        self.nextupdate = 0
        self.fmax_max = 1.0

        # Scaling deformation progress frame
        self.scalebox = ui.VBox()
        self.scaleframe = ui.Frame(_("Scaling deformation:"))
        vbox = ui.VBox()
        self.scaleframe.add(vbox)
        pack(self.scalebox, [self.scaleframe])
        pack(self.scalebox, ui.Label(""))

        self.label_scale_stepno_format = _("Step number %s of %s.")
        self.label_scale_stepno = ui.Label(self.label_scale_stepno_format %
                                           ("-", "-"))
        pack(vbox, [self.label_scale_stepno])
        self.scale_progress = ui.ProgressBar()
        self.scale_progress.modify_bg(ui.STATE_PRELIGHT, '#00AA00')
        pack(vbox, [self.scale_progress])

        vbox.show()
        self.scaleframe.show()
        self.globalbox.pack_start(self.scalebox)

        # Minimization progress frame
        self.minbox = ui.VBox()  # Box containing frame and spacing
        self.minframe = ui.Frame(_("Energy minimization:"))
        vbox = ui.VBox()  # Box containing the frames content.
        self.minframe.add(vbox)
        pack(self.minbox, [self.minframe])
        pack(self.minbox, ui.Label(""))

        self.label_min_stepno = ui.Label("-")
        pack(vbox, [ui.Label(_("Step number: ")), self.label_min_stepno])
        lbl = ui.Label()
        lbl.set_markup(_("F<sub>max</sub>: "))
        self.minimize_progress = ui.ProgressBar()
        pack(vbox, [lbl, self.minimize_progress])
        self.label_min_fmax = ui.Label("-")
        lbl = ui.Label()
        lbl.set_markup(_("Convergence criterion: F<sub>max</sub> = "))
        pack(vbox, [lbl, self.label_min_fmax])
        self.label_min_maxsteps = ui.Label("-")
        pack(vbox,
             [ui.Label(_("Max. number of steps: ")), self.label_min_maxsteps])

        vbox.show()
        self.minframe.show()
        self.globalbox.pack_start(self.minbox)
        self.globalbox.show()
        self.add(self.globalbox)

        # Make the cancel button
        self.cancelbut = ui.Button('Cancel')
        self.cancelbut.connect('clicked', self.cancel)
        pack(self.globalbox, [self.cancelbut], end=True, bottom=True)

    def begin(self,
              mode=None,
              algo=None,
              fmax=None,
              steps=None,
              scalesteps=None):
        self.mode = mode
        # Hide all mode-specific boxes
        self.scalebox.hide()
        self.minbox.hide()
        # Activate any relevant box
        if mode == "scale" or mode == "scale/min":
            self.scalesteps = int(scalesteps)
            self.scalebox.show()
            self.set_scale_progress(0, init=True)
        if mode == "min" or mode == "scale/min":
            # It is a minimization.
            self.minbox.show()
            self.label_min_stepno.set_text("-")
            self.label_min_fmax.set_text("%.3f" % (fmax, ))
            self.label_min_maxsteps.set_text(str(int(steps)))
            self.minimize_progress.set_fraction(0)
            self.minimize_progress.set_text(_("unknown"))
        # Record starting time
        self.starttime = time.time()
        self.active = None  # Becoming active
        self.raisecancelexception = False

    def end(self):
        self.hide()
        self.active = False

    def activity(self):
        "Register that activity occurred."
        if self.active is None and time.time(
        ) > self.starttime + self.waittime:
            # This has taken so long that a progress bar is needed.
            self.show()
            self.active = True
        # Allow GTK to update display
        if self.active:
            while ui.events_pending():
                ui.main_iteration()
        if self.raisecancelexception:
            self.cancelbut.set_sensitive(True)
            raise AseGuiCancelException

    def cancel(self, widget):
        print("CANCEL pressed.")
        # We cannot raise the exception here, as this function is
        # called by the GTK main loop.
        self.raisecancelexception = True
        self.cancelbut.set_sensitive(False)

    def set_scale_progress(self, step, init=False):
        "Set the step number in scaling deformation."
        self.label_scale_stepno.set_text(self.label_scale_stepno_format %
                                         (step, self.scalesteps))
        percent = 1.0 * step / self.scalesteps
        self.scale_progress.set_fraction(percent)
        self.scale_progress.set_text("%i%%" % (round(100 * percent), ))
        if not init:
            self.activity()

    def logger_write(self, line):
        if time.time() > self.nextupdate:
            if self.mode == "min" or self.mode == "scale/min":
                # Update the minimization progress bar.
                w = line.split()
                fmax = float(w[-1])
                step = w[1]
                if fmax > self.fmax_max:
                    self.fmax_max = np.ceil(fmax)
                self.minimize_progress.set_fraction(fmax / self.fmax_max)
                self.minimize_progress.set_text(w[-1])
                self.label_min_stepno.set_text(step)
            else:
                raise RuntimeError(
                    "ProgressIndicator.logger_write called unexpectedly")
            self.activity()
            self.nextupdate = time.time() + self.updatetime

    def get_logger_stream(self):
        return LoggerStream(self)


class GpawProgressIndicator(DefaultProgressIndicator):
    "Window for reporting GPAW progress."

    def __init__(self):
        DefaultProgressIndicator.__init__(self)

        # GPAW progress frame
        self.gpawframe = ui.Frame("GPAW progress:")
        vbox = self.gpawvbox = ui.VBox()
        self.gpawframe.add(vbox)
        self.table = ui.Table(1, 2)
        self.tablerows = 0
        pack(vbox, self.table)
        self.status = ui.Label("-")
        self.tablepack([ui.Label(_("Status: ")), self.status])
        self.iteration = ui.Label("-")
        self.tablepack([ui.Label(_("Iteration: ")), self.iteration])
        self.tablepack([ui.Label("")])
        lbl = ui.Label()
        lbl.set_markup(_("log<sub>10</sub>(change):"))
        self.tablepack([ui.Label(""), lbl])
        self.wfs_progress = ui.ProgressBar()
        self.tablepack([ui.Label(_("Wave functions: ")), self.wfs_progress])
        self.dens_progress = ui.ProgressBar()
        self.tablepack([ui.Label(_("Density: ")), self.dens_progress])
        self.energy_progress = ui.ProgressBar()
        self.tablepack([ui.Label(_("Energy: ")), self.energy_progress])
        self.tablepack([ui.Label("")])
        self.versionlabel = ui.Label("")
        self.tablepack([ui.Label(_("GPAW version: ")), self.versionlabel])
        self.natomslabel = ui.Label("")
        self.tablepack([ui.Label(_("Number of atoms: ")), self.natomslabel])
        self.memorylabel = ui.Label(_("N/A"))
        self.tablepack([ui.Label(_("Memory estimate: ")), self.memorylabel])
        self.globalbox.pack_start(self.gpawframe)
        self.gpawframe.show()

        vbox.show()
        self.active = False

    def tablepack(self, widgets):
        self.tablerows += 1
        self.table.resize(self.tablerows, 2)
        for i, w in enumerate(widgets):
            self.table.attach(w, i, i + 1, self.tablerows - 1, self.tablerows)
            if hasattr(w, "set_alignment"):
                w.set_alignment(0, 0.5)
            w.show()

    def begin(self, **kwargs):
        DefaultProgressIndicator.begin(self, **kwargs)
        # Set GPAW specific stuff.
        self.active = True
        self.oldenergy = None
        self.poscount = None
        self.reset_gpaw_bars()
        # With GPAW, all calculations are slow: Show progress window
        # immediately.
        self.show()
        while ui.events_pending():
            ui.main_iteration()

    def reset_gpaw_bars(self):
        for lbl in (self.status, self.iteration):
            lbl.set_text("-")
        for bar in (self.wfs_progress, self.dens_progress,
                    self.energy_progress):
            bar.set_fraction(0.0)
            bar.set_text(_("No info"))

    def gpaw_write(self, txt):
        # if not self.active:
        #    self.begin()
        sys.stdout.write(txt)
        versearch = re.search(r"\|[ |_.]+([0-9]+\.[0-9]+\.[0-9]+)", txt)
        if versearch:
            # Starting a gpaw calculation.
            self.versionlabel.set_text(versearch.group(1))
            self.status.set_text(_("Initializing"))
        elif txt.startswith(_("Positions:")):
            # Start counting atoms
            self.poscount = True
            self.reset_gpaw_bars()
            self.status.set_text(_("Starting calculation"))
            self.oldenergy = None
        elif txt.strip() == "":
            # Stop counting atoms
            self.poscount = False
        elif self.poscount:
            # Count atoms.
            w = txt.split()
            assert (len(w) == 5)
            self.natoms = int(w[0]) + 1
            self.natomslabel.set_text(str(self.natoms))
        elif txt.startswith("iter:"):
            # Found iteration line.
            wfs = txt[self.wfs_idx:self.density_idx].strip()
            dens = txt[self.density_idx:self.energy_idx].strip()
            energy = txt[self.energy_idx:self.fermi_idx].strip()
            if wfs:
                p = fraction(float(wfs), -9.0)
                self.wfs_progress.set_fraction(p)
                self.wfs_progress.set_text(wfs)
            if dens:
                p = fraction(float(dens), -4.0)
                self.dens_progress.set_fraction(p)
                self.dens_progress.set_text(dens)
            if energy:
                if self.oldenergy is None:
                    self.oldenergy = float(energy)
                else:
                    de = abs(self.oldenergy - float(energy))
                    self.oldenergy = float(energy)
                    if de > 1e-10:
                        de = np.log10(de / self.natoms)
                        p = fraction(de, -3.0)
                        self.energy_progress.set_fraction(p)
                        self.energy_progress.set_text("%.1f" % de)
                    else:
                        self.energy_progress.set_fraction(1)
                        self.energy_progress.set_text(_("unchanged"))
            words = txt.split()
            self.iteration.set_text(words[1])
        elif (-1 < txt.find("WFS") < txt.find("Density") < txt.find("Energy") <
              txt.find("Fermi")):
            # Found header of convergence table
            self.wfs_idx = txt.find("WFS")
            self.density_idx = txt.find("Density")
            self.energy_idx = txt.find("Energy")
            self.fermi_idx = txt.find("Fermi")
            self.status.set_text(_("Self-consistency loop"))
            self.iteration.set_text("0")
        elif txt.find("Converged After") != -1:
            # SCF loop has converged.
            words = txt.split()
            self.status.set_text(_("Calculating forces"))
            self.iteration.set_text(words[2] + _(" (converged)"))
        elif -1 < txt.find("Calculator") < txt.find("MiB"):
            # Memory estimate
            words = txt.split()
            self.memorylabel.set_text(words[1] + " " + words[2])
        self.activity()

    def get_gpaw_stream(self):
        return GpawStream(self)


class LoggerStream:
    "A file-like object feeding minimizer logs to GpawProgressWindow."

    def __init__(self, progresswindow):
        self.window = progresswindow

    def write(self, txt):
        self.window.logger_write(txt)

    def flush(self):
        pass


class GpawStream:
    "A file-like object feeding GPAWs txt file to GpawProgressWindow."

    def __init__(self, progresswindow):
        self.window = progresswindow

    def write(self, txt):
        if txt == "":
            return
        endline = txt[-1] == '\n'
        if endline:
            txt = txt[:-1]
        lines = txt.split("\n")
        if endline:
            for l in lines:
                self.window.gpaw_write(l + '\n')
        else:
            for l in lines[:-1]:
                self.window.gpaw_write(l + '\n')
            self.window.gpaw_write(lines[-1])

    def flush(self):
        pass


def fraction(value, maximum):
    p = value / maximum
    if p < 0.0:
        return 0.0
    elif p > 1.0:
        return 1.0
    else:
        return p
