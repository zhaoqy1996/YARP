import time
from distutils.version import LooseVersion
from ase.io.utils import generate_writer_variables, make_patch_list


class EPS:
    def __init__(self, atoms,
                 rotation='', show_unit_cell=0, radii=None,
                 bbox=None, colors=None, scale=20, maxwidth=500):
        """Encapsulated PostScript writer.

        show_unit_cell: int
            0: Don't show unit cell (default).  1: Show unit cell.
            2: Show unit cell and make sure all of it is visible.
        """
        generate_writer_variables(
            self, atoms, rotation=rotation,
            show_unit_cell=show_unit_cell,
            radii=radii, bbox=bbox, colors=colors, scale=scale,
            maxwidth=maxwidth)

    def write(self, filename):
        self.filename = filename
        self.write_header()
        self.write_body()
        self.write_trailer()

    def write_header(self):
        import matplotlib
        if LooseVersion(matplotlib.__version__) <= '0.8':
            raise RuntimeError('Your version of matplotlib (%s) is too old' %
                               matplotlib.__version__)

        from matplotlib.backends.backend_ps import RendererPS, psDefs

        self.fd = open(self.filename, 'w')
        self.fd.write('%!PS-Adobe-3.0 EPSF-3.0\n')
        self.fd.write('%%Creator: G2\n')
        self.fd.write('%%CreationDate: %s\n' % time.ctime(time.time()))
        self.fd.write('%%Orientation: portrait\n')
        bbox = (0, 0, self.w, self.h)
        self.fd.write('%%%%BoundingBox: %d %d %d %d\n' % bbox)
        self.fd.write('%%EndComments\n')

        Ndict = len(psDefs)
        self.fd.write('%%BeginProlog\n')
        self.fd.write('/mpldict %d dict def\n' % Ndict)
        self.fd.write('mpldict begin\n')
        for d in psDefs:
            d = d.strip()
            for l in d.split('\n'):
                self.fd.write(l.strip() + '\n')
        self.fd.write('%%EndProlog\n')

        self.fd.write('mpldict begin\n')
        self.fd.write('%d %d 0 0 clipbox\n' % (self.w, self.h))

        self.renderer = RendererPS(self.w, self.h, self.fd)

    def write_body(self):
        patch_list = make_patch_list(self)
        for patch in patch_list:
            patch.draw(self.renderer)

    def write_trailer(self):
        self.fd.write('end\n')
        self.fd.write('showpage\n')
        self.fd.close()


def write_eps(filename, atoms, **parameters):
    if isinstance(atoms, list):
        assert len(atoms) == 1
        atoms = atoms[0]
    EPS(atoms, **parameters).write(filename)
