import matplotlib.pyplot as plt
from ase.test.testsuite import NotAvailable
from ase.gui.ui import tk

try:
    plt.figure()
except (tk.TclError, RuntimeError) as err:
    # "RuntimeError: Invalid DISPLAY variable" may happen in conda tests
    raise NotAvailable(err)

from ase.visualize.plot import plot_atoms
from ase.lattice.cubic import FaceCenteredCubic

slab = FaceCenteredCubic('Au', size=(2, 2, 2))

fig, ax = plt.subplots()
plot_atoms(slab, ax, radii=0.5, rotation=('10x,10y,10z'))

assert len(ax.patches) == len(slab)
print(ax)
