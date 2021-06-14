from __future__ import unicode_literals, division

import numpy as np

import ase.gui.ui as ui
from ase.gui.i18n import _


class Movie:
    def __init__(self, gui):
        self.win = win = ui.Window(_('Movie'), self.close)
        win.add(_('Image number:'))
        self.frame_number = ui.Scale(gui.frame + 1, 1,
                                     len(gui.images),
                                     callback=self.new_frame)
        win.add(self.frame_number)

        win.add([ui.Button(_('First'), self.click, -1, True),
                 ui.Button(_('Back'), self.click, -1),
                 ui.Button(_('Forward'), self.click, 1),
                 ui.Button(_('Last'), self.click, 1, True)])

        play = ui.Button(_('Play'), self.play)
        stop = ui.Button(_('Stop'), self.stop)

        # TRANSLATORS: This function plays an animation forwards and backwards
        # alternatingly, e.g. for displaying vibrational movement
        self.rock = ui.CheckButton(_('Rock'))

        win.add([play, stop, self.rock])

        if len(gui.images) > 150:
            skipdefault = len(gui.images) // 150
            tdefault = min(max(len(gui.images) / (skipdefault * 5.0),
                               1.0), 30)
        else:
            skipdefault = 0
            tdefault = min(max(len(gui.images) / 5.0, 1.0), 30)
        self.time = ui.SpinBox(tdefault, 1.0, 99, 0.1)
        self.skip = ui.SpinBox(skipdefault, 0, 99, 1)
        win.add([_(' Frame rate: '), self.time, _(' Skip frames: '),
                 self.skip])

        self.gui = gui
        self.direction = 1
        self.timer = None
        gui.register_vulnerable(self)

    def notify_atoms_changed(self):
        """Called by gui object when the atoms have changed."""
        self.close()

    def close(self):
        self.stop()
        self.win.close()

    def click(self, step, firstlast=False):
        if firstlast and step < 0:
            i = 0
        elif firstlast:
            i = len(self.gui.images) - 1
        else:
            i = max(0, min(len(self.gui.images) - 1, self.gui.frame + step))

        self.frame_number.value = i + 1
        if firstlast:
            self.direction = np.sign(-step)
        else:
            self.direction = np.sign(step)

    def new_frame(self, value):
        self.gui.set_frame(value - 1)

    def play(self):
        self.stop()
        t = 1 / self.time.value
        self.timer = self.gui.window.after(t, self.step)

    def stop(self):
        if self.timer is not None:
            self.timer.cancel()

    def step(self):
        i = self.gui.frame
        nimages = len(self.gui.images)
        delta = int(self.skip.value + 1)

        if self.rock.value:
            if i <= self.skip.value:
                self.direction = 1
            elif i >= nimages - delta:
                self.direction = -1
            i += self.direction * delta
        else:
            i = (i + self.direction * delta + nimages) % nimages

        self.frame_number.value = i + 1
        self.play()
