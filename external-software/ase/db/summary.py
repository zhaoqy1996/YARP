from __future__ import print_function

from ase.db.core import float_to_time_string, now
from ase.geometry import cell_to_cellpar
from ase.utils import formula_metal

# Predefined blocks:
ATOMS = {'type': 'atoms'}
UNITCELL = {'type': 'cell'}


def create_table(row,  # type: AtomsRow
                 header,  # type: List[str]
                 keys,  # type: List[str]
                 key_descriptions,  # type: Dict[str, Tuple[str, str, str]]
                 digits=3  # type: int
                 ):  # -> Dict[str, Any]
    """Create table-dict from row."""
    table = []
    for key in keys:
        if key == 'age':
            age = float_to_time_string(now() - row.ctime, True)
            table.append(('Age', age))
            continue
        value = row.get(key)
        if value is not None:
            if isinstance(value, float):
                value = '{:.{}f}'.format(value, digits)
            elif not isinstance(value, str):
                value = str(value)
            desc, unit = key_descriptions.get(key, ['', key, ''])[1:]
            if unit:
                value += ' ' + unit
            table.append((desc, value))
    return {'type': 'table',
            'header': header,
            'rows': table}


def default_layout(row,  # type: AtomsRow
                   key_descriptions,  # type: Dict[str, Tuple[str, str, str]]
                   prefix  # type: str
                   ):  # -> List[Tuple[str, List[List[Dict[str, Any]]]]]
    """Default page layout.

    "Basic properties" section and the rest in a "miscellaneous" section.
    """
    keys = ['id',
            'energy', 'fmax', 'smax',
            'mass',
            'age']
    table = create_table(row, ['Key', 'Value'], keys, key_descriptions)
    misc = miscellaneous_section(row, key_descriptions, exclude=keys)
    layout = [('Basic properties', [[ATOMS, UNITCELL],
                                    [table]]),
              misc]
    return layout


def miscellaneous_section(row, key_descriptions, exclude):
    """Helper function for adding a "miscellaneous" section.

    Create table with all keys except those in exclude.
    """
    misckeys = (set(key_descriptions) |
                set(row.key_value_pairs)) - set(exclude)
    misc = create_table(row, ['Items', ''], sorted(misckeys), key_descriptions)
    return ('Miscellaneous', [[misc]])


class Summary:
    def __init__(self, row, meta={}, subscript=None, prefix=''):
        self.row = row

        self.cell = [['{:.3f}'.format(a) for a in axis] for axis in row.cell]
        par = ['{:.3f}'.format(x) for x in cell_to_cellpar(row.cell)]
        self.lengths = par[:3]
        self.angles = par[3:]

        self.stress = row.get('stress')
        if self.stress is not None:
            self.stress = ', '.join('{0:.3f}'.format(s) for s in self.stress)

        self.formula = formula_metal(row.numbers)
        if subscript:
            self.formula = subscript.sub(r'<sub>\1</sub>', self.formula)

        kd = meta.get('key_descriptions', {})
        create_layout = meta.get('layout') or default_layout
        self.layout = create_layout(row, kd, prefix)

        self.dipole = row.get('dipole')
        if self.dipole is not None:
            self.dipole = ', '.join('{0:.3f}'.format(d) for d in self.dipole)

        self.data = row.get('data')
        if self.data:
            self.data = ', '.join(self.data.keys())

        self.constraints = row.get('constraints')
        if self.constraints:
            self.constraints = ', '.join(c.__class__.__name__
                                         for c in self.constraints)

    def write(self):
        print(self.formula + ':')
        for headline, columns in self.layout:
            blocks = columns[0]
            if len(columns) == 2:
                blocks += columns[1]
            print((' ' + headline + ' ').center(78, '='))
            for block in blocks:
                if block['type'] == 'table':
                    rows = block['rows']
                    if not rows:
                        print()
                        continue
                    rows = [block['header']] + rows
                    widths = [max(len(row[n]) for row in rows)
                              for n in range(len(rows[0]))]
                    for row in rows:
                        print('|'.join('{:{}}'.format(word, width)
                                       for word, width in zip(row, widths)))
                    print()
                elif block['type'] == 'figure':
                    print(block['filename'])
                    print()
                elif block['type'] == 'cell':
                    print('Unit cell in Ang:')
                    print('axis|periodic|          x|          y|          z')
                    c = 1
                    fmt = '   {0}|     {1}|{2[0]:>11}|{2[1]:>11}|{2[2]:>11}'
                    for p, axis in zip(self.row.pbc, self.cell):
                        print(fmt.format(c, [' no', 'yes'][p], axis))
                        c += 1
                    print('Lengths: {:>10}{:>10}{:>10}'
                          .format(*self.lengths))
                    print('Angles:  {:>10}{:>10}{:>10}\n'
                          .format(*self.angles))

        if self.stress:
            print('Stress tensor (xx, yy, zz, zy, zx, yx) in eV/Ang^3:')
            print('   ', self.stress, '\n')

        if self.dipole:
            print('Dipole moment in e*Ang: ({})\n'.format(self.dipole))

        if self.constraints:
            print('Constraints:', self.constraints, '\n')

        if self.data:
            print('Data:', self.data, '\n')


def convert_old_layout(page):
    def layout(row, kd, prefix):
        def fix(block):
            if isinstance(block, tuple):
                title, keys = block
                return create_table(row, title, keys, kd)
            return block

        return [(title, [[fix(block) for block in column]
                         for column in columns])
                for title, columns in page]
    return layout
