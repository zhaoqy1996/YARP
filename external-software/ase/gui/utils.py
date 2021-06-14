def get_magmoms(atoms):
    if atoms.calc is not None:
        if not atoms.calc.calculation_required(atoms, ['magmoms']):
            return atoms.get_magnetic_moments()
    return atoms.get_initial_magnetic_moments()
