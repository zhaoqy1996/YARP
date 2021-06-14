from __future__ import unicode_literals
import argparse
import os

import numpy as np

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import molecule
from ase.gui.i18n import _
from ase.test import NotAvailable

try:
    import ase.gui.ui as ui
except ImportError:
    raise NotAvailable

from ase.gui.gui import GUI
from ase.gui.save import save_dialog


if not os.environ.get('DISPLAY'):
    raise NotAvailable


class Error:
    """Fake window for testing puposes."""
    has_been_called = False

    def __call__(self, title, text=None):
        self.text = text or title
        self.has_been_called = True

    def called(self, text=None):
        """Check that an oops-window was opened with correct title."""
        if not self.has_been_called:
            return False

        self.has_been_called = False  # ready for next call

        return text is None or text == self.text


ui.error = Error()

alltests = []


def test(f):
    """Decorator for marking tests."""
    alltests.append(f.__name__)
    return f


@test
def nanotube(gui):
    nt = gui.nanotube_window()
    nt.apply()
    nt.element[1].value = '?'
    nt.apply()
    assert ui.error.called(
        _('You have not (yet) specified a consistent set of parameters.'))

    nt.element[1].value = 'C'
    nt.ok()
    assert len(gui.images[0]) == 20


@test
def nanopartickle(gui):
    n = gui.nanoparticle_window()
    n.element.symbol = 'Cu'
    n.apply()
    n.set_structure_data()
    assert len(gui.images[0]) == 675
    n.method.value = 'wulff'
    n.update_gui_method()
    n.apply()


@test
def color(gui):
    a = Atoms('C10', magmoms=np.linspace(1, -1, 10))
    a.positions[:] = np.linspace(0, 9, 10)[:, None]
    a.calc = SinglePointCalculator(a, forces=a.positions)
    gui.new_atoms(a)
    c = gui.colors_window()
    c.toggle('force')
    text = c.toggle('magmom')
    activebuttons = [button.active for button in c.radio.buttons]
    assert activebuttons == [1, 0, 1, 0, 0, 1, 1], activebuttons
    assert text.rsplit('[', 1)[1].startswith('-1.000000,1.000000]')


@test
def settings(gui):
    gui.new_atoms(molecule('H2O'))
    s = gui.settings()
    s.scale.value = 1.9
    s.scale_radii()


@test
def rotate(gui):
    gui.window['toggle-show-bonds'] = True
    gui.new_atoms(molecule('H2O'))
    gui.rotate_window()


@test
def open_and_save(gui):
    mol = molecule('H2O')
    for i in range(3):
        mol.write('h2o.json')
    gui.open(filename='h2o.json')
    save_dialog(gui, 'h2o.cif@-1')

@test
def test_fracocc(gui):
    from ase.test.fio.cif import content
    with open('./fracocc.cif', 'w') as f:
        f.write(content)
    gui.open(filename='fracocc.cif')



p = argparse.ArgumentParser()
p.add_argument('tests', nargs='*')
p.add_argument('-p', '--pause', action='store_true')

if __name__ == '__main__':
    args = p.parse_args()
else:
    # We are running inside the test framework: ignore sys.args
    args = p.parse_args([])

for name in args.tests or alltests:
    for n in alltests:
        if n.startswith(name):
            name = n
            break
    else:
        1 / 0
    print(name)
    test = globals()[name]
    gui = GUI()

    def f():
        test(gui)
        if not args.pause:
            gui.exit()
    gui.run(test=f)




import os
from functools import partial

from ase.test import NotAvailable

try:
    import ase.gui.ui as ui
except ImportError:
    raise NotAvailable


if not os.environ.get('DISPLAY'):
    raise NotAvailable


def window():

    def hello(event=None):
        print('hello', event)

    menu = [('Hi', [ui.MenuItem('_Hello', hello, 'Ctrl+H')]),
            ('Hell_o', [ui.MenuItem('ABC', hello, choices='ABC')])]
    win = ui.MainWindow('Test', menu=menu)

    win.add(ui.Label('Hello'))
    win.add(ui.Button('Hello', hello))

    r = ui.Rows([ui.Label(x * 7) for x in 'abcd'])
    win.add(r)
    r.add('11111\n2222\n333\n44\n5')

    def abc(x):
        print(x, r.rows)

    cb = ui.ComboBox(['Aa', 'Bb', 'Cc'], callback=abc)
    win.add(cb)

    rb = ui.RadioButtons(['A', 'B', 'C'], 'ABC', abc)
    win.add(rb)

    b = ui.CheckButton('Hello')

    def hi():
        print(b.value, rb.value, cb.value)
        del r[2]
        r.add('-------------')

    win.add([b, ui.Button('Hi', hi)])

    return win


def run():
    win = window()
    win.test(partial(test, win))


def test(win):
    win.things[1].callback()
    win.things[1].callback()
    win.close()

run()
