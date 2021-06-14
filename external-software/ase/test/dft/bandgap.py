from ase.dft.bandgap import bandgap


class Test:
    def get_ibz_k_points(self):
        return [(0, 0, 0)]

    def get_number_of_spins(self):
        return 2

    def get_eigenvalues(self, kpt, spin):
        return [-10, spin, spin + 2.0]

    def get_fermi_level(self):
        return 0.5


gaps = [2, 11, 1, 2, 11, 1]

calc = Test()
for direct in [0, 1]:
    for spin in [0, 1, None]:
        gap, k1, k2 = bandgap(calc, direct=direct, spin=spin)
        print(gap, k1, k2)
        assert gap == gaps.pop(0)
