from __future__ import print_function
import numpy as np
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.emt import EMT
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list

# parameters
N_cell = 2
R_QMs = np.array([3, 7])

# setup bulk and MM region
bulk_at = bulk("Cu", cubic=True)
sigma = (bulk_at*2).get_distance(0, 1)*(2.**(-1./6))
mm = LennardJones(sigma=sigma, epsilon=0.05)
qm = EMT()

# compute MM and QM equations of state
def strain(at, e, calc):
    at = at.copy()
    at.set_cell((1.0 + e)*at.cell, scale_atoms=True)
    at.set_calculator(calc)
    v = at.get_volume()
    e = at.get_potential_energy()
    return v, e

eps = np.linspace(-0.01, 0.01, 13)
v_qm, E_qm = zip(*[strain(bulk_at, e, qm) for e in eps])
v_mm, E_mm = zip(*[strain(bulk_at, e, mm) for e in eps])

eos_qm = EquationOfState(v_qm, E_qm)
v0_qm, E0_qm, B_qm = eos_qm.fit()
a0_qm = v0_qm**(1.0/3.0)

eos_mm = EquationOfState(v_mm, E_mm)
v0_mm, E0_mm, B_mm = eos_mm.fit()
a0_mm = v0_mm**(1.0/3.0)

mm_r = RescaledCalculator(mm, a0_qm, B_qm, a0_mm, B_mm)
v_mm_r, E_mm_r = zip(*[strain(bulk_at, e, mm_r) for e in eps])

eos_mm_r = EquationOfState(v_mm_r, E_mm_r)
v0_mm_r, E0_mm_r, B_mm_r = eos_mm_r.fit()
a0_mm_r = v0_mm_r**(1.0/3)

# check match of a0 and B after rescaling is adequete
assert abs((a0_mm_r - a0_qm)/a0_qm) < 1e-3 # 0.1% error in lattice constant
assert abs((B_mm_r - B_qm)/B_qm) < 0.05 # 5% error in bulk modulus

# plt.plot(v_mm, E_mm - np.min(E_mm), 'o-', label='MM')
# plt.plot(v_qm, E_qm - np.min(E_qm), 'o-', label='QM')
# plt.plot(v_mm_r, E_mm_r - np.min(E_mm_r), 'o-', label='MM rescaled')
# plt.legend()

at0 = bulk_at * N_cell
r = at0.get_distances(0, np.arange(1, len(at0)), mic=True)
print(len(r))
del at0[0] # introduce a vacancy
print("N_cell", N_cell, 'N_MM', len(at0))

ref_at = at0.copy()
ref_at.set_calculator(qm)
opt = FIRE(ref_at)
opt.run(fmax=1e-3)
u_ref = ref_at.positions - at0.positions

us = []
for R_QM in R_QMs:
    at = at0.copy()
    mask = r < R_QM
    print('R_QM', R_QM, 'N_QM', mask.sum(), 'N_total', len(at))
    qmmm = ForceQMMM(at, mask, qm, mm, buffer_width=2*qm.rc)
    at.set_calculator(qmmm)
    opt = FIRE(at)
    opt.run(fmax=1e-3)
    us.append(at.positions - at0.positions)

# compute error in energy norm |\nabla u - \nabla u_ref|
def strain_error(at0, u_ref, u, cutoff, mask):
    I, J = neighbor_list('ij', at0, cutoff)
    I, J = np.array([(i,j) for i, j in zip(I, J) if mask[i]]).T
    v = u_ref - u
    dv = np.linalg.norm(v[I, :] - v[J, :], axis=1)
    return np.linalg.norm(dv)

du_global = [strain_error(at0, u_ref, u, 1.5*sigma, np.ones(len(r))) for u in us]
du_local = [strain_error(at0, u_ref, u, 1.5*sigma, r < 3.0) for u in us]

print('du_local', du_local)
print('du_global', du_global)

# check local errors are monotonically decreasing
assert np.all(np.diff(du_local) < 0)

# check global errors are monotonically converging
assert np.all(np.diff(du_global) < 0)

# biggest QM/MM should match QM result
assert du_local[-1] < 1e-10
assert du_global[-1] < 1e-10
