from ase.build import bulk
a1 = bulk('ZnS', 'wurtzite', a=3.0, u=0.23) * (1, 2, 1)
a2 = bulk('ZnS', 'wurtzite', a=3.0, u=0.23, orthorhombic=True)
a1.cell = a2.cell
a1.wrap()
assert abs(a1.positions - a2.positions).max() < 1e-14
