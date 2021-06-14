import numpy as np

from ase.dft.kpoints import labels_from_kpts
from ase.io.jsonio import encode, decode
from ase.parallel import paropen


def get_band_structure(atoms=None, calc=None):
    """Create band structure object from Atoms or calculator."""
    atoms = atoms if atoms is not None else calc.atoms
    calc = calc if calc is not None else atoms.calc

    kpts = calc.get_ibz_k_points()

    energies = []
    for s in range(calc.get_number_of_spins()):
        energies.append([calc.get_eigenvalues(kpt=k, spin=s)
                         for k in range(len(kpts))])
    energies = np.array(energies)

    return BandStructure(cell=atoms.cell,
                         kpts=kpts,
                         energies=energies,
                         reference=calc.get_fermi_level())


class BandStructurePlot:
    def __init__(self, bs):
        self.bs = bs
        self.ax = None
        self.xcoords = None
        self.show_legend = False

    def plot(self, ax=None, spin=None, emin=-10, emax=5, filename=None,
             show=None, ylabel=None, colors=None, label=None,
             spin_labels=['spin up', 'spin down'], loc=None, **plotkwargs):
        """Plot band-structure.

        spin: int or None
            Spin channel.  Default behaviour is to plot both spin up and down
            for spin-polarized calculations.
        emin,emax: float
            Maximum energy above reference.
        filename: str
            Write image to a file.
        ax: Axes
            MatPlotLib Axes object.  Will be created if not supplied.
        show: bool
            Show the image.
        """

        if self.ax is None:
            ax = self.prepare_plot(ax, emin, emax, ylabel)

        if spin is None:
            e_skn = self.bs.energies
        else:
            e_skn = self.bs.energies[spin, np.newaxis]

        if colors is None:
            if len(e_skn) == 1:
                colors = 'g'
            else:
                colors = 'yb'

        nspins = len(e_skn)

        for spin, e_kn in enumerate(e_skn):
            color = colors[spin]
            kwargs = dict(color=color)
            kwargs.update(plotkwargs)
            if nspins == 2:
                if label:
                    lbl = label + ' ' + spin_labels[spin]
                else:
                    lbl = spin_labels[spin]
            else:
                lbl = label
            ax.plot(self.xcoords, e_kn[:, 0], label=lbl, **kwargs)
            for e_k in e_kn.T[1:]:
                ax.plot(self.xcoords, e_k, **kwargs)

        self.show_legend = label is not None or nspins == 2
        self.finish_plot(filename, show, loc)

        return ax

    def plot_with_colors(self, ax=None, emin=-10, emax=5, filename=None,
                         show=None, energies=None, colors=None,
                         ylabel=None, clabel='$s_z$', cmin=-1.0, cmax=1.0,
                         sortcolors=False, loc=None, s=2):
        """Plot band-structure with colors."""

        import matplotlib.pyplot as plt

        if self.ax is None:
            ax = self.prepare_plot(ax, emin, emax, ylabel)

        shape = energies.shape
        xcoords = np.vstack([self.xcoords] * shape[1])
        if sortcolors:
            perm = colors.argsort(axis=None)
            energies = energies.ravel()[perm].reshape(shape)
            colors = colors.ravel()[perm].reshape(shape)
            xcoords = xcoords.ravel()[perm].reshape(shape)

        for e_k, c_k, x_k in zip(energies, colors, xcoords):
            things = ax.scatter(x_k, e_k, c=c_k, s=s,
                                vmin=cmin, vmax=cmax)

        cbar = plt.colorbar(things)
        cbar.set_label(clabel)

        self.finish_plot(filename, show, loc)

        return ax

    def prepare_plot(self, ax=None, emin=-10, emax=5, ylabel=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.figure().add_subplot(111)

        def pretty(kpt):
            if kpt == 'G':
                kpt = r'$\Gamma$'
            elif len(kpt) == 2:
                kpt = kpt[0] + '$_' + kpt[1] + '$'
            return kpt

        emin += self.bs.reference
        emax += self.bs.reference

        self.xcoords, label_xcoords, orig_labels = self.bs.get_labels()

        labels = [pretty(name) for name in orig_labels]
        i = 1
        while i < len(labels):
            if label_xcoords[i - 1] == label_xcoords[i]:
                labels[i - 1] = labels[i - 1][:-1] + ',' + labels[i][1:]
                labels[i] = ''
            i += 1

        for x in label_xcoords[1:-1]:
            ax.axvline(x, color='0.5')

        ylabel = ylabel if ylabel is not None else 'energies [eV]'

        ax.set_xticks(label_xcoords)
        ax.set_xticklabels(labels)
        ax.axis(xmin=0, xmax=self.xcoords[-1], ymin=emin, ymax=emax)
        ax.set_ylabel(ylabel)
        ax.axhline(self.bs.reference, color='k', ls=':')
        self.ax = ax
        return ax

    def finish_plot(self, filename, show, loc):
        import matplotlib.pyplot as plt

        if self.show_legend:
            leg = plt.legend(loc=loc)
            leg.get_frame().set_alpha(1)

        if filename:
            plt.savefig(filename)

        if show is None:
            show = not filename

        if show:
            plt.show()


class BandStructure:
    def __init__(self, cell, kpts, energies, reference=0.0):
        """Create band structure object from energies and k-points."""
        assert cell.shape == (3, 3)
        self.cell = cell
        assert kpts.shape[1] == 3
        self.kpts = kpts
        self.energies = np.asarray(energies)
        self.reference = reference

    def get_labels(self):
        return labels_from_kpts(self.kpts, self.cell)

    def todict(self):
        return dict((key, getattr(self, key))
                    for key in
                    ['cell', 'kpts', 'energies', 'reference'])

    def write(self, filename):
        """Write to json file."""
        with paropen(filename, 'w') as f:
            f.write(encode(self))

    @staticmethod
    def read(filename):
        """Read from json file."""
        with open(filename, 'r') as f:
            dct = decode(f.read())
        return BandStructure(**dct)

    def plot(self, *args, **kwargs):
        bsp = BandStructurePlot(self)
        # Maybe return bsp?  But for now run the plot, for compatibility
        return bsp.plot(*args, **kwargs)
