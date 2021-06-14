from ase.calculators.test import make_test_dft_calculation
from ase.dft.stm import STM

atoms = make_test_dft_calculation()
stm = STM(atoms, [0, 1, 2])
c = stm.get_averaged_current(-1.0, 4.5)
x, y, h = stm.scan(-1.0, c)
stm.write('stm.pckl')
x, y, h2 = STM('stm.pckl').scan(-1, c)
assert abs(h - h2).max() == 0

stm = STM(atoms, use_density=True)
c = stm.get_averaged_current(-1, 4.5)
x, y, I = stm.scan2(-1.0, 1.0)
stm.write('stm2.pckl')
x, y, I2 = STM('stm2.pckl').scan2(-1, 1)
assert abs(I - I2).max() == 0

stm = STM(atoms, use_density=True)
c = stm.get_averaged_current(42, 4.5)
x, y = stm.linescan(42, c, [0, 0], [2, 2])
assert abs(x[-1] - 2 * 2**0.5) < 1e-13
assert abs(y[-1] - y[0]) < 1e-13
