# -*- encoding: utf-8 -*-
from __future__ import print_function, unicode_literals
from ase.gui.i18n import _
import ase.gui.ui as ui
from ase.io.pov import write_pov
from os import system
import numpy as np

pack = error = Help = 42

class Render:
    texture_list = ['ase2', 'ase3', 'glass', 'simple', 'pale',
                    'intermediate', 'vmd', 'jmol']
    cameras = ['orthographic', 'perspective', 'ultra_wide_angle']

    def __init__(self, gui):
        self.gui = gui
        self.win = win = ui.Window(_('Render current view in povray ... '))
        win.add(ui.Label(_("Rendering %d atoms.") % len(self.gui.atoms)))

        guiwidth, guiheight = self.get_guisize()
        self.width_widget = ui.SpinBox(guiwidth, start=1, end=9999, step=1)
        self.height_widget = ui.SpinBox(guiheight, start=1, end=9999, step=1)
        win.add([ui.Label(_('Size')), self.width_widget,
                 ui.Label('⨯'), self.height_widget])

        self.linewidth_widget = ui.SpinBox(0.07, start=0.01, end=9.99,
                                           step=0.01)
        win.add([ui.Label(_('Line width')), self.linewidth_widget,
                 ui.Label(_('Ångström'))])

        self.constraints_widget = ui.CheckButton(_("Render constraints"))
        self.cell_widget = ui.CheckButton(_("Render unit cell"), value=True)
        win.add([self.cell_widget, self.constraints_widget])

        formula = gui.atoms.get_chemical_formula(mode='hill')
        self.basename_widget = ui.Entry(width=30, value=formula,
                                        callback=self.update_outputname)
        win.add([ui.Label(_('Output basename: ')), self.basename_widget])
        self.outputname_widget = ui.Label()
        win.add([ui.Label(_('Output filename: ')), self.outputname_widget])
        self.update_outputname()

        self.texture_widget = ui.ComboBox(labels=self.texture_list,
                                          values=self.texture_list)
        win.add([ui.Label(_('Atomic texture set:')),
                 self.texture_widget])
        # complicated texture stuff

        self.camera_widget = ui.ComboBox(labels=self.cameras,
                                         values=self.cameras)
        self.camera_distance_widget = ui.SpinBox(50.0, -99.0, 99.0, 1.0)
        win.add([ui.Label(_('Camera type: ')), self.camera_widget])
        win.add([ui.Label(_('Camera distance')), self.camera_distance_widget])

        # render current frame/all frames
        self.frames_widget = ui.RadioButtons([_('Render current frame'),
                                              _('Render all frames')])
        win.add(self.frames_widget)
        if len(gui.images) == 1:
            self.frames_widget.buttons[1].widget.configure(state='disabled')

        self.run_povray_widget = ui.CheckButton(_('Run povray'), True)
        self.keep_files_widget = ui.CheckButton(_('Keep povray files'), False)
        self.show_output_widget = ui.CheckButton(_('Show output window'), True)
        self.transparent = ui.CheckButton(_("Transparent background"), True)
        win.add(self.transparent)
        win.add([self.run_povray_widget, self.keep_files_widget,
                 self.show_output_widget])
        win.add(ui.Button(_('Render'), self.ok))

    def get_guisize(self):
        win = self.gui.window.win
        return win.winfo_width(), win.winfo_height()

    def ok(self, *args):
        print("Rendering with povray:")
        guiwidth, guiheight = self.get_guisize()
        width = self.width_widget.value
        height = self.height_widget.value
        # (Do width/height become inconsistent upon gui resize?  Not critical)
        scale = self.gui.scale * height / guiheight
        bbox = np.empty(4)
        size = np.array([width, height]) / scale
        bbox[0:2] = np.dot(self.gui.center, self.gui.axes[:, :2]) - size / 2
        bbox[2:] = bbox[:2] + size
        povray_settings = {
            'run_povray': self.run_povray_widget.value,
            'bbox': bbox,
            'rotation': self.gui.axes,
            'show_unit_cell': self.cell_widget.value,
            'display': self.show_output_widget.value,
            'transparent': self.transparent.value,
            'camera_type': self.camera_widget.value,
            'camera_dist': self.camera_distance_widget.value,
            'canvas_width': width,
            'celllinewidth': self.linewidth_widget.value,
            'exportconstraints': self.constraints_widget.value,
        }
        multiframe = bool(self.frames_widget.value)
        if multiframe:
            assert len(self.gui.images) > 1

        if multiframe:
            frames = range(len(self.gui.images))
        else:
            frames = [self.gui.frame]

        initial_frame = self.gui.frame
        for frame in frames:
            self.gui.set_frame(frame)
            povray_settings['textures'] = self.get_textures()
            povray_settings['colors'] = self.gui.get_colors(rgb=True)
            atoms = self.gui.images.get_atoms(frame)
            filename = self.update_outputname()
            print(" | Writing files for image", filename, "...")
            write_pov(
                filename, atoms, radii=self.gui.get_covalent_radii(),
                **povray_settings)
            if not self.keep_files_widget.value:
                print(" | Deleting temporary file ", filename)
                system("rm " + filename)
                filename = filename[:-4] + '.ini'
                print(" | Deleting temporary file ", filename)
                system("rm " + filename)
        self.gui.set_frame(initial_frame)
        self.update_outputname()

    def update_outputname(self):
        tokens = [self.basename_widget.value]
        movielen = len(self.gui.images)
        if movielen > 1:
            ndigits = len(str(movielen))
            token = ('{:0' + str(ndigits) + 'd}').format(self.gui.frame)
            tokens.append(token)
        tokens.append('pov')
        fname = '.'.join(tokens)
        self.outputname_widget.text = fname
        return fname
        #if self.movie.get_active():
        #    while len(movie_index) + len(str(self.iframe)) < len(
        #            str(self.nimages)):
        #        movie_index += '0'
        #    movie_index = '.' + movie_index + str(self.iframe)
        #name = self.basename.get_text() + movie_index + '.pov'
        #self.outputname.set_text(name)

    def get_textures(self):
        return [self.texture_widget.value] * len(self.gui.atoms)
        #natoms = len(self.gui.atoms)
        #textures = natoms * [
            #self.texture_list[0]  #self.default_texture.get_active()]
        #]
        #for mat in self.materials:
        #    sel = mat[1]
        #    t = self.finish_list[mat[2].get_active()]
        #    if mat[0]:
        #        for n, val in enumerate(sel):
        #            if val:
        #                textures[n] = t
        #return textures
