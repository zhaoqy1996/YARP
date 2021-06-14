from distutils.version import LooseVersion

import numpy as np

from ase.io.eps import EPS


class PNG(EPS):
    def write_header(self):
        from matplotlib.backends.backend_agg import RendererAgg

        try:
            from matplotlib.transforms import Value
        except ImportError:
            dpi = 72
        else:
            dpi = Value(72)

        self.renderer = RendererAgg(self.w, self.h, dpi)

    def write_trailer(self):
        renderer = self.renderer
        if hasattr(renderer._renderer, 'write_png'):
            # Old version of matplotlib:
            renderer._renderer.write_png(self.filename)
        else:
            from matplotlib import _png
            # buffer_rgba does not accept arguments from version 1.2.0
            # https://github.com/matplotlib/matplotlib/commit/f4fee350f9f
            import matplotlib
            if LooseVersion(matplotlib.__version__) < '1.2.0':
                _png.write_png(renderer.buffer_rgba(0, 0),
                               renderer.width, renderer.height,
                               self.filename, 72)
            else:
                x = renderer.buffer_rgba()
                try:
                    _png.write_png(x, self.w, self.h, self.filename, 72)
                except (TypeError, ValueError):
                    x = np.frombuffer(x, np.uint8).reshape(
                        (int(self.h), int(self.w), 4))
                    _png.write_png(x, self.filename, 72)


def write_png(filename, atoms, **parameters):
    PNG(atoms, **parameters).write(filename)
