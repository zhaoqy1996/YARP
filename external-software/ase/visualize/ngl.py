# coding: utf-8

from ase import Atoms


class NGLDisplay:
    """Structure display class

    Provides basic structure/trajectory display
    in the notebook and optional gui which can be used to enhance its
    usability.  It is also possible to extend the functionality of the
    particular instance of the viewer by adding further widgets
    manipulating the structure.
    """
    def __init__(self, atoms, xsize=500, ysize=500):
        import nglview
        import nglview.color

        from ipywidgets import Dropdown, FloatSlider, IntSlider, HBox, VBox
        self.atoms = atoms
        if isinstance(atoms[0], Atoms):
            # Assume this is a trajectory or struct list
            self.view = nglview.show_asetraj(atoms)
            self.frm = IntSlider(value=0, min=0, max=len(atoms) - 1)
            self.frm.observe(self._update_frame)
            self.struct = atoms[0]
        else:
            # Assume this is just a single structure
            self.view = nglview.show_ase(atoms)
            self.struct = atoms
            self.frm = None

        self.colors = {}
        self.view._remote_call('setSize', target='Widget',
                               args=['%dpx' % (xsize,), '%dpx' % (ysize,)])
        self.view.add_unitcell()
        self.view.add_spacefill()
        self.view.remove_ball_and_stick()
        self.view.camera = 'orthographic'
        self.view.parameters = { "clipDist": 0 }

        self.view.center()

        self.asel = Dropdown(options=['All'] +
                             list(set(self.struct.get_chemical_symbols())),
                             value='All', description='Show')

        self.csel = Dropdown(options=nglview.color.COLOR_SCHEMES,
                             value=' ', description='Color scheme')

        self.rad = FloatSlider(value=0.8, min=0.0, max=1.5, step=0.01,
                               description='Ball size')

        self.asel.observe(self._select_atom)
        self.csel.observe(self._update_repr)
        self.rad.observe(self._update_repr)

        self.view.update_spacefill(radiusType='covalent',
                                   scale=0.8,
                                   color_scheme=self.csel.value,
                                   color_scale='rainbow')

        wdg = [self.asel, self.csel, self.rad]
        if self.frm:
            wdg.append(self.frm)

        self.gui = HBox([self.view, VBox(wdg)])
        # Make useful shortcuts for the user of the class
        self.gui.view = self.view
        self.gui.control_box = self.gui.children[1]
        self.gui.custom_colors = self.custom_colors

    def _update_repr(self, chg=None):
        self.view.update_spacefill(radiusType='covalent',
                                   scale=self.rad.value,
                                   color_scheme=self.csel.value,
                                   color_scale='rainbow')

    def _update_frame(self, chg=None):
        self.view.frame = self.frm.value
        return

    def _select_atom(self, chg=None):
        sel = self.asel.value
        self.view.remove_spacefill()
        for e in set(self.struct.get_chemical_symbols()):
            if (sel == 'All' or e == sel):
                if e in self.colors:
                    self.view.add_spacefill(selection='#' + e,
                                            color=self.colors[e])
                else:
                    self.view.add_spacefill(selection='#' + e)
        self._update_repr()

    def custom_colors(self, clr=None):
        """
        Define custom colors for some atoms. Pass a dictionary of the form
        {'Fe':'red', 'Au':'yellow'} to the function.
        To reset the map to default call the method without parameters.
        """
        if clr:
            self.colors = clr
        else:
            self.colors = {}
        self._select_atom()


def view_ngl(atoms, w=500, h=500):
    """
    Returns the nglviewer + some control widgets in the VBox ipywidget.
    The viewer supports any Atoms objectand any sequence of Atoms objects.
    The returned object has two shortcuts members:

    .view:
        nglviewer ipywidget for direct interaction
    .control_box:
        VBox ipywidget containing view control widgets
    """
    return NGLDisplay(atoms, w, h).gui
