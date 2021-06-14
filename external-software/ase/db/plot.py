from ase.io.png import write_png
from ase.utils import basestring


def atoms2png(atoms, filename):
    if atoms:
        size = atoms.positions.ptp(0)
        i = size.argmin()
        rotation = ['-90y', '90x', ''][i]
        size[i] = 0.0
        scale = min(50, 200 / max(1, size.max()))
    else:
        scale = 20
        rotation = ''
    write_png(filename, atoms, show_unit_cell=1,
              rotation=rotation, scale=scale)


def dct2plot(dct, name, filename=None, show=True):
    """Create a plot from a dict.

    Example::

        d = {'a': [0, 1, 2],
             'b': [1.2, 1.1, 1.0],
             'abplot': {'title': 'Example',
                        'data': [{'x': 'a',
                                  'y': 'b',
                                  'label': 'label1',
                                  'style': 'o-g'}],
                        'xlabel': 'blah-blah [eV]'}}
        dct2plot(d, 'plot')

    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    styles = ['k-', 'r-', 'g-', 'b-']
    plot = dct[name]
    lines = []
    labels = []
    for d in plot['data']:
        x = d['x']
        if isinstance(x, basestring):
            x = dct[x]
        y = d['y']
        if isinstance(y, basestring):
            y = dct[y]
        style = d.get('style')
        if not style:
            style = styles.pop()
        lines.append(plt.plot(x, y, style)[0])
        labels.append(d['label'])
    plt.legend(lines, labels)
    if isinstance(plot['xlabel'], basestring):
        plt.xlabel(plot['xlabel'])
    else:
        x, labels = plot['xlabel']
        plt.xticks(x, labels)
        plt.xlim(x[0], x[-1])
    plt.ylabel(plot['ylabel'])
    if 'ylim' in plot:
        plt.ylim(*plot['ylim'])
    plt.title(plot['title'])
    try:
        plt.tight_layout()
    except AttributeError:
        pass
    if show:
        plt.show()
    if filename:
        plt.savefig(filename)
        plt.close(fig)
