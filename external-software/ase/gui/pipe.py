from __future__ import print_function
import pickle
import sys


def main():
    import matplotlib.pyplot as plt
    stdin = sys.stdin
    if sys.version_info[0] == 3:
        stdin = stdin.buffer
    task, data = pickle.load(stdin)
    if task == 'eos':
        from ase.eos import plot
        plot(*data)
    elif task == 'neb':
        from ase.neb import plot_band_from_fit
        plot_band_from_fit(*data)
    elif task == 'reciprocal':
        from ase.dft.bz import bz3d_plot
        bz3d_plot(**data)
    elif task == 'graph':
        from ase.gui.graphs import make_plot
        make_plot(show=False, *data)
    else:
        print('Invalid task {}'.format(task))
        sys.exit(17)

    # Magic string to tell GUI that things went okay:
    print('GUI:OK')
    sys.stdout.close()

    plt.show()

if __name__ == '__main__':
    main()
