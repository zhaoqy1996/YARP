"""Water dimer calculation in which each molecule is calculated quantum
mechanically and the interaction between the molecules is electrostatic.
The process is repeated until self consitence. """

from numpy.linalg import norm
from ase.collections import s22
from ase.calculators.turbomole import Turbomole


def polarization_cycle(partition_1, partition_2, charges_2=None):
    """Performs an iteration of a polarization calculation."""
    properties = {}
    calc = Turbomole(atoms=partition_1, **params)
    if charges_2 is not None:
        calc.embed(charges=charges_2, positions=partition_2.positions)
    properties['e1'] = partition_1.get_potential_energy()
    properties['c1'] = partition_1.get_charges()
    calc = Turbomole(atoms=partition_2, **params)
    calc.embed(charges=properties['c1'], positions=partition_1.positions)
    properties['e2'] = partition_2.get_potential_energy()
    properties['c2'] = partition_2.get_charges()
    return properties


params = {'esp fit': 'kollman', 'multiplicity': 1}
dimer = s22['Water_dimer']

# system partitioning
part1 = dimer[0:3]
part2 = dimer[3:6]

new = polarization_cycle(part1, part2)
prop = {'e1': [], 'e2': [], 'c1': [], 'c2': []}
for key in prop:
    prop[key].append(new[key])

# start values and convergence criteria
conv = {'e': [1.0], 'c': [1.0]}
thres = {'e': 1e-4, 'c': 1e-2}
iteration = 0
while any([conv[key][-1] > thres[key] for key in conv]):
    iteration += 1
    new = polarization_cycle(part1, part2, charges_2=prop['c2'][-1])
    for key in prop:
        prop[key].append(new[key])

    (new1, old1) = (prop['e1'][-1], prop['e1'][-2])
    (new2, old2) = (prop['e2'][-1], prop['e2'][-2])
    conv['e'].append((abs(new1 - old1) + abs(new2 - old2)) /
                     (abs(old1) + abs(old2)))
    (new1, old1) = (prop['c1'][-1], prop['c1'][-2])
    (new2, old2) = (prop['c2'][-1], prop['c2'][-2])
    conv['c'].append((norm(new1 - old1) + norm(new2 - old2)) /
                     (norm(old1) + norm(old2)))
    fmt = 'iteration {0:d}: convergence of energy {1:10e}; of charges {2:10e}'
    print(fmt.format(iteration, conv['e'][-1], norm(conv['c'][-1])))

# check the result
ref = {
    'e1': -2077.7082947500003,
    'e2': -2077.3347674372353,
    'c1': [-0.133033, 0.238218, -0.105186],
    'c2': [-0.844336, 0.422151, 0.422184]
}

dev = {}
for key in ref:
    val = prop[key][-1]
    measurement = val if isinstance(val, float) else norm(val)
    reference = ref[key] if isinstance(ref[key], float) else norm(ref[key])
    dev[key] = (measurement - reference) / reference
    print('Deviation of {0} is {1:10f}'.format(key, dev[key]))

# allow deviations of up to 5%
assert all([dev[key] < 5e-2 for key in dev]), 'deviation too large'
