from __future__ import unicode_literals, division

import os
import pickle
import subprocess
import sys
import tempfile
import weakref
from functools import partial
from ase.gui.i18n import _
from time import time

import numpy as np

from ase import __version__
import ase.gui.ui as ui
from ase.gui.crystal import SetupBulkCrystal
from ase.gui.defaults import read_defaults
from ase.gui.graphene import SetupGraphene
from ase.gui.images import Images
from ase.gui.nanoparticle import SetupNanoparticle
from ase.gui.nanotube import SetupNanotube
from ase.gui.save import save_dialog
from ase.gui.settings import Settings
from ase.gui.status import Status
from ase.gui.surfaceslab import SetupSurfaceSlab
from ase.gui.view import View


class GUI(View, Status):
    ARROWKEY_SCAN = 0
    ARROWKEY_MOVE = 1
    ARROWKEY_ROTATE = 2

    def __init__(self, images=None,
                 rotations='',
                 show_bonds=False, expr=None):

        if not isinstance(images, Images):
            images = Images(images)

        self.images = images

        self.config = read_defaults()
        if show_bonds:
            self.config['show_bonds'] = True

        menu = self.get_menu_data()

        self.window = ui.ASEGUIWindow(close=self.exit, menu=menu,
                                      config=self.config, scroll=self.scroll,
                                      scroll_event=self.scroll_event,
                                      press=self.press, move=self.move,
                                      release=self.release,
                                      resize=self.resize)

        View.__init__(self, rotations)
        Status.__init__(self)

        self.subprocesses = []  # list of external processes
        self.movie_window = None
        self.vulnerable_windows = []
        self.simulation = {}  # Used by modules on Calculate menu.
        self.module_state = {}  # Used by modules to store their state.

        self.arrowkey_mode = self.ARROWKEY_SCAN
        self.move_atoms_mask = None

        self.set_frame(len(self.images) - 1, focus=True)

        # Used to move the structure with the mouse
        self.prev_pos = None
        self.last_scroll_time = time()
        self.orig_scale = self.scale

        if len(self.images) > 1:
            self.movie()

        if expr is None:
            expr = self.config['gui_graphs_string']

        if expr is not None and expr != '' and len(self.images) > 1:
            self.plot_graphs(expr=expr, ignore_if_nan=True)

    @property
    def moving(self):
        return self.arrowkey_mode != self.ARROWKEY_SCAN

    def run(self, test=None):
        if test:
            self.window.test(test)
        else:
            self.window.run()

    def toggle_move_mode(self, key=None):
        self.toggle_arrowkey_mode(self.ARROWKEY_MOVE)

    def toggle_rotate_mode(self, key=None):
        self.toggle_arrowkey_mode(self.ARROWKEY_ROTATE)

    def toggle_arrowkey_mode(self, mode):
        # If not currently in given mode, activate it.
        # Else, deactivate it (go back to SCAN mode)
        assert mode != self.ARROWKEY_SCAN

        if self.arrowkey_mode == mode:
            self.arrowkey_mode = self.ARROWKEY_SCAN
            self.move_atoms_mask = None
        else:
            self.arrowkey_mode = mode
            self.move_atoms_mask = self.images.selected.copy()

        self.draw()

    def step(self, key):
        d = {'Home': -10000000,
             'Page-Up': -1,
             'Page-Down': 1,
             'End': 10000000}[key]
        i = max(0, min(len(self.images) - 1, self.frame + d))
        self.set_frame(i)
        if self.movie_window is not None:
            self.movie_window.frame_number.value = i + 1

    def _do_zoom(self, x):
        """Utility method for zooming"""
        self.scale *= x
        self.draw()

    def zoom(self, key):
        """Zoom in/out on keypress or clicking menu item"""
        x = {'+': 1.2, '-': 1 / 1.2}[key]
        self._do_zoom(x)

    def scroll_event(self, event):
        """Zoom in/out when using mouse wheel"""
        SHIFT = event.modifier == 'shift'
        x = 1.0
        if event.button == 4 or event.delta > 0:
            x = 1.0 + (1 - SHIFT) * 0.2 + SHIFT * 0.01
        elif event.button == 5 or event.delta < 0:
            x = 1.0 / (1.0 + (1 - SHIFT) * 0.2 + SHIFT * 0.01)
        self._do_zoom(x)

    def settings(self):
        return Settings(self)

    def scroll(self, event):
        CTRL = event.modifier == 'ctrl'

        # Bug: Simultaneous CTRL + shift is the same as just CTRL.
        # Therefore movement in Z direction does not support the
        # shift modifier.
        dxdydz = {'up': (0, 1 - CTRL, CTRL),
                  'down': (0, -1 + CTRL, -CTRL),
                  'right': (1, 0, 0),
                  'left': (-1, 0, 0)}.get(event.key, None)

        # Get scroll direction using shift + right mouse button
        # event.type == '6' is mouse motion, see:
        # http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/event-types.html
        if event.type == '6':
            cur_pos = np.array([event.x, -event.y])
            # Continue scroll if button has not been released
            if self.prev_pos is None or time() - self.last_scroll_time > .5:
                self.prev_pos = cur_pos
                self.last_scroll_time = time()
            else:
                dxdy = cur_pos - self.prev_pos
                dxdydz = np.append(dxdy, [0])
                self.prev_pos = cur_pos
                self.last_scroll_time = time()

        if dxdydz is None:
            return

        vec = 0.1 * np.dot(self.axes, dxdydz)
        if event.modifier == 'shift':
            vec *= 0.1

        if self.arrowkey_mode == self.ARROWKEY_MOVE:
            self.atoms.positions[self.move_atoms_mask[:len(self.atoms)]] += vec
            self.set_frame()
        elif self.arrowkey_mode == self.ARROWKEY_ROTATE:
            # For now we use atoms.rotate having the simplest interface.
            # (Better to use something more minimalistic, obviously.)
            mask = self.move_atoms_mask[:len(self.atoms)]
            center = self.atoms.positions[mask].mean(axis=0)
            tmp_atoms = self.atoms[mask]
            tmp_atoms.positions -= center
            tmp_atoms.rotate(50 * np.linalg.norm(vec), vec)
            self.atoms.positions[mask] = tmp_atoms.positions + center
            self.set_frame()
        else:
            # The displacement vector is scaled
            # so that the cursor follows the structure
            # Scale by a third works for some reason
            scale = self.orig_scale / (3 * self.scale)
            self.center -= vec * scale

            # dx * 0.1 * self.axes[:, 0] - dy * 0.1 * self.axes[:, 1])

            self.draw()

    def delete_selected_atoms(self, widget=None, data=None):
        import ase.gui.ui as ui
        nselected = sum(self.images.selected)
        if nselected and ui.ask_question('Delete atoms',
                                         'Delete selected atoms?'):
            mask = self.images.selected[:len(self.atoms)]
            del self.atoms[mask]

            # Will remove selection in other images, too
            self.images.selected[:] = False
            self.set_frame()
            self.draw()

    def execute(self):
        from ase.gui.execute import Execute
        Execute(self)

    def constraints_window(self):
        from ase.gui.constraints import Constraints
        Constraints(self)

    def select_all(self, key=None):
        self.images.selected[:] = True
        self.draw()

    def invert_selection(self, key=None):
        self.images.selected[:] = ~self.images.selected
        self.draw()

    def select_constrained_atoms(self, key=None):
        self.images.selected[:] = ~self.images.get_dynamic(self.atoms)
        self.draw()

    def select_immobile_atoms(self, key=None):
        if len(self.images) > 1:
            R0 = self.images[0].positions
            for atoms in self.images[1:]:
                R = atoms.positions
                self.images.selected[:] = ~(np.abs(R - R0) > 1.0e-10).any(1)
        self.draw()

    def movie(self):
        from ase.gui.movie import Movie
        self.movie_window = Movie(self)

    def plot_graphs(self, key=None, expr=None, ignore_if_nan=False):
        from ase.gui.graphs import Graphs
        g = Graphs(self)
        if expr is not None:
            g.plot(expr=expr, ignore_if_nan=ignore_if_nan)

    def pipe(self, task, data):
        process = subprocess.Popen([sys.executable, '-m', 'ase.gui.pipe'],
                                   stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE)
        pickle.dump((task, data), process.stdin)
        process.stdin.close()
        # Either process writes a line, or it crashes and line becomes ''
        line = process.stdout.readline().decode('utf8').strip()

        if line != 'GUI:OK':
            if line == '':  # Subprocess probably crashed
                line = _('Failure in subprocess')
            self.bad_plot(line)
        else:
            self.subprocesses.append(process)

    def bad_plot(self, err, msg=''):
        ui.error(_('Plotting failed'), '\n'.join([str(err), msg]).strip())

    def neb(self):
        from ase.neb import NEBtools
        try:
            nebtools = NEBtools(self.images)
            fit = nebtools.get_fit()
        except Exception as err:
            self.bad_plot(err, _('Images must have energies and forces, '
                                 'and atoms must not be stationary.'))
        else:
            self.pipe('neb', fit)

    def bulk_modulus(self):
        try:
            v = [abs(np.linalg.det(atoms.cell)) for atoms in self.images]
            e = [self.images.get_energy(a) for a in self.images]
            from ase.eos import EquationOfState
            eos = EquationOfState(v, e)
            plotdata = eos.getplotdata()
        except Exception as err:
            self.bad_plot(err, _('Images must have energies '
                                 'and varying cell.'))
        else:
            self.pipe('eos', plotdata)

    def reciprocal(self):
        if self.atoms.number_of_lattice_vectors != 3:
            self.bad_plot(_('Requires 3D cell.'))
            return

        kwargs = dict(cell=self.atoms.cell, vectors=True)
        self.pipe('reciprocal', kwargs)

    def open(self, button=None, filename=None):
        chooser = ui.ASEFileChooser(self.window.win)

        filename = filename or chooser.go()
        format = chooser.format
        if filename:
            try:
                self.images.read([filename], slice(None), format)
            except Exception as err:
                ui.show_io_error(filename, err)
                return  # Hmm.  Is self.images in a consistent state?
            self.set_frame(len(self.images) - 1, focus=True)

    def modify_atoms(self, key=None):
        from ase.gui.modify import ModifyAtoms
        ModifyAtoms(self)

    def add_atoms(self, key=None):
        from ase.gui.add import AddAtoms
        AddAtoms(self)

    def cell_editor(self, key=None):
        from ase.gui.celleditor import CellEditor
        CellEditor(self)

    def quick_info_window(self, key=None):
        from ase.gui.quickinfo import info
        ui.Window(_('Quick Info')).add(info(self))

    def bulk_window(self):
        SetupBulkCrystal(self)

    def surface_window(self):
        SetupSurfaceSlab(self)

    def nanoparticle_window(self):
        return SetupNanoparticle(self)

    def graphene_window(self, menuitem):
        SetupGraphene(self)

    def nanotube_window(self):
        return SetupNanotube(self)

    def new_atoms(self, atoms, init_magmom=False):
        "Set a new atoms object."
        rpt = getattr(self.images, 'repeat', None)
        self.images.repeat_images(np.ones(3, int))
        self.images.initialize([atoms], init_magmom=init_magmom)
        self.frame = 0  # Prevent crashes
        self.images.repeat_images(rpt)
        self.set_frame(frame=0, focus=True)
        self.notify_vulnerable()

    def notify_vulnerable(self):
        """Notify windows that would break when new_atoms is called.

        The notified windows may adapt to the new atoms.  If that is not
        possible, they should delete themselves.
        """
        new_vul = []  # Keep weakrefs to objects that still exist.
        for wref in self.vulnerable_windows:
            ref = wref()
            if ref is not None:
                new_vul.append(wref)
                ref.notify_atoms_changed()
        self.vulnerable_windows = new_vul

    def register_vulnerable(self, obj):
        """Register windows that are vulnerable to changing the images.

        Some windows will break if the atoms (and in particular the
        number of images) are changed.  They can register themselves
        and be closed when that happens.
        """
        self.vulnerable_windows.append(weakref.ref(obj))

    def exit(self, event=None):
        for process in self.subprocesses:
            process.terminate()
        self.window.close()

    def new(self, key=None):
        os.system('ase gui &')

    def save(self, key=None):
        return save_dialog(self)

    def external_viewer(self, name):
        command = {'xmakemol': 'xmakemol -f',
                   'rasmol': 'rasmol -xyz'}.get(name, name)
        fd, filename = tempfile.mkstemp('.xyz', 'ase.gui-')
        os.close(fd)
        self.images.write(filename)
        os.system('(%s %s &); (sleep 60; rm %s) &' %
                  (command, filename, filename))

    def get_menu_data(self):
        M = ui.MenuItem
        return [
            (_('_File'),
             [M(_('_Open'), self.open, 'Ctrl+O'),
              M(_('_New'), self.new, 'Ctrl+N'),
              M(_('_Save'), self.save, 'Ctrl+S'),
              M('---'),
              M(_('_Quit'), self.exit, 'Ctrl+Q')]),

            (_('_Edit'),
             [M(_('Select _all'), self.select_all),
              M(_('_Invert selection'), self.invert_selection),
              M(_('Select _constrained atoms'), self.select_constrained_atoms),
              M(_('Select _immobile atoms'), self.select_immobile_atoms),
              # M('---'),
              # M(_('_Copy'), self.copy_atoms, 'Ctrl+C'),
              # M(_('_Paste'), self.paste_atoms, 'Ctrl+V'),
              M('---'),
              M(_('Hide selected atoms'), self.hide_selected),
              M(_('Show selected atoms'), self.show_selected),
              M('---'),
              M(_('_Modify'), self.modify_atoms, 'Ctrl+Y'),
              M(_('_Add atoms'), self.add_atoms, 'Ctrl+A'),
              M(_('_Delete selected atoms'), self.delete_selected_atoms,
                'Backspace'),
              M(_('Edit _cell'), self.cell_editor, 'Ctrl+E'),
              M('---'),
              M(_('_First image'), self.step, 'Home'),
              M(_('_Previous image'), self.step, 'Page-Up'),
              M(_('_Next image'), self.step, 'Page-Down'),
              M(_('_Last image'), self.step, 'End')]),

            (_('_View'),
             [M(_('Show _unit cell'), self.toggle_show_unit_cell, 'Ctrl+U',
                value=self.config['show_unit_cell']),
              M(_('Show _axes'), self.toggle_show_axes,
                value=self.config['show_axes']),
              M(_('Show _bonds'), self.toggle_show_bonds, 'Ctrl+B',
                value=self.config['show_bonds']),
              M(_('Show _velocities'), self.toggle_show_velocities, 'Ctrl+G',
                value=False),
              M(_('Show _forces'), self.toggle_show_forces, 'Ctrl+F',
                value=False),
              M(_('Show _Labels'), self.show_labels,
                choices=[_('_None'),
                         _('Atom _Index'),
                         _('_Magnetic Moments'),  # XXX check if exist
                         _('_Element Symbol'),
                         _('_Initial Charges'),  # XXX check if exist
                         ]),
              M('---'),
              M(_('Quick Info ...'), self.quick_info_window, 'Ctrl+I'),
              M(_('Repeat ...'), self.repeat_window, 'R'),
              M(_('Rotate ...'), self.rotate_window),
              M(_('Colors ...'), self.colors_window, 'C'),
              # TRANSLATORS: verb
              M(_('Focus'), self.focus, 'F'),
              M(_('Zoom in'), self.zoom, '+'),
              M(_('Zoom out'), self.zoom, '-'),
              M(_('Change View'),
                submenu=[
                    M(_('Reset View'), self.reset_view, '='),
                    M(_('xy-plane'), self.set_view, 'Z'),
                    M(_('yz-plane'), self.set_view, 'X'),
                    M(_('zx-plane'), self.set_view, 'Y'),
                    M(_('yx-plane'), self.set_view, 'Alt+Z'),
                    M(_('zy-plane'), self.set_view, 'Alt+X'),
                    M(_('xz-plane'), self.set_view, 'Alt+Y'),
                    M(_('a2,a3-plane'), self.set_view, '1'),
                    M(_('a3,a1-plane'), self.set_view, '2'),
                    M(_('a1,a2-plane'), self.set_view, '3'),
                    M(_('a3,a2-plane'), self.set_view, 'Alt+1'),
                    M(_('a1,a3-plane'), self.set_view, 'Alt+2'),
                    M(_('a2,a1-plane'), self.set_view, 'Alt+3')]),
              M(_('Settings ...'), self.settings),
              M('---'),
              M(_('VMD'), partial(self.external_viewer, 'vmd')),
              M(_('RasMol'), partial(self.external_viewer, 'rasmol')),
              M(_('xmakemol'), partial(self.external_viewer, 'xmakemol')),
              M(_('avogadro'), partial(self.external_viewer, 'avogadro'))]),

            (_('_Tools'),
             [M(_('Graphs ...'), self.plot_graphs),
              M(_('Movie ...'), self.movie),
              M(_('Expert mode ...'), self.execute, disabled=True),
              M(_('Constraints ...'), self.constraints_window),
              M(_('Render scene ...'), self.render_window),
              M(_('_Move atoms'), self.toggle_move_mode, 'Ctrl+M'),
              M(_('_Rotate atoms'), self.toggle_rotate_mode, 'Ctrl+R'),
              M(_('NE_B'), self.neb),
              M(_('B_ulk Modulus'), self.bulk_modulus),
              M(_('Reciprocal space ...'), self.reciprocal)]),

            # TRANSLATORS: Set up (i.e. build) surfaces, nanoparticles, ...
            (_('_Setup'),
             [M(_('_Bulk Crystal'), self.bulk_window, disabled=True),
              M(_('_Surface slab'), self.surface_window, disabled=False),
              M(_('_Nanoparticle'),
                self.nanoparticle_window),
              M(_('Nano_tube'), self.nanotube_window),
              M(_('Graphene'), self.graphene_window, disabled=True)]),

            # (_('_Calculate'),
            # [M(_('Set _Calculator'), self.calculator_window, disabled=True),
            #  M(_('_Energy and Forces'), self.energy_window, disabled=True),
            #  M(_('Energy Minimization'), self.energy_minimize_window,
            #    disabled=True)]),

            (_('_Help'),
             [M(_('_About'), partial(ui.about, 'ASE-GUI',
                                     version=__version__,
                                     webpage='https://wiki.fysik.dtu.dk/'
                                     'ase/ase/gui/gui.html')),
              M(_('Webpage ...'), webpage)])]

    def repeat_poll(self, callback, ms, ensure_update=True):
        """Invoke callback(gui=self) every ms milliseconds.

        This is useful for polling a resource for updates to load them
        into the GUI.  The GUI display will be hence be updated after
        each call; pass ensure_update=False to circumvent this.

        Polling stops if the callback function raises StopIteration.

        Example to run a movie manually, then quit::

            from ase.collections import g2
            from ase.gui.gui import GUI

            names = iter(g2.names)

            def main(gui):
                try:
                    name = next(names)
                except StopIteration:
                    gui.window.win.quit()
                else:
                    atoms = g2[name]
                    gui.images.initialize([atoms])

            gui = GUI()
            gui.repeat_poll(main, 30)
            gui.run()"""

        def callbackwrapper():
            try:
                callback(gui=self)
            except StopIteration:
                pass
            finally:
                # Reinsert self so we get called again:
                self.window.win.after(ms, callbackwrapper)

            if ensure_update:
                self.set_frame()
                self.draw()

        self.window.win.after(ms, callbackwrapper)

def webpage():
    import webbrowser
    webbrowser.open('https://wiki.fysik.dtu.dk/ase/ase/gui/gui.html')
