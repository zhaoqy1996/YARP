"""
Provides LoggingCalculator class to wrap a Calculator and record
number of enery and force calls
"""

import json
import logging
import time

import numpy as np

from ase.calculators.calculator import Calculator, all_properties

logger = logging.getLogger(__name__)


class LoggingCalculator(Calculator):
    """Calculator wrapper to record and plot history of energy and function
    evaluations
    """
    implemented_properties = all_properties
    default_parameters = {}
    name = 'LoggingCalculator'

    property_to_method_name = {
        'energy': 'get_potential_energy',
        'energies': 'get_potential_energies',
        'forces': 'get_forces',
        'stress': 'get_stress',
        'stresses': 'get_stresses'}

    def __init__(self, calculator, jsonfile=None, dumpjson=False):
        Calculator.__init__(self)
        self.calculator = calculator
        self.fmax = {}
        self.walltime = {}
        self.energy_evals = {}
        self.energy_count = {}
        self.set_label('(none)')
        if jsonfile is not None:
            self.read_json(jsonfile)
        self.dumpjson = dumpjson

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if isinstance(self.calculator, Calculator):
            results = [self.calculator.get_property(prop, atoms)
                       for prop in properties]
        else:
            results = []
            for prop in properties:
                method_name = self.property_to_method_name[prop]
                method = getattr(self.calculator, method_name)
                results.append(method(atoms))

        if 'energy' in properties or 'energies' in properties:
            self.energy_evals.setdefault(self.label, 0)
            self.energy_evals[self.label] += 1
            try:
                energy = results[properties.index('energy')]
            except IndexError:
                energy = sum(results[properties.index('energies')])
            logger.info('energy call count=%d energy=%.3f',
                        self.energy_evals[self.label], energy)
        self.results = dict(zip(properties, results))

        if 'forces' in self.results:
            fmax = self.fmax.setdefault(self.label, [])
            walltime = self.walltime.setdefault(self.label, [])
            forces = self.results['forces'].copy()
            energy_count = self.energy_count.setdefault(self.label, [])
            energy_evals = self.energy_evals.setdefault(self.label, 0)
            energy_count.append(energy_evals)
            for constraint in atoms.constraints:
                constraint.adjust_forces(atoms, forces)
            fmax.append(abs(forces).max())
            walltime.append(time.time())
            logger.info('force call fmax=%.3f', fmax[-1])

        if self.dumpjson:
            self.write_json('dump.json')

    def write_json(self, filename):
        f = open(filename, 'w')
        json.dump({'fmax': self.fmax,
                   'walltime': self.walltime,
                   'energy_evals': self.energy_evals,
                   'energy_count': self.energy_count}, f)
        f.close()

    def read_json(self, filename, append=False, label=None):
        f = open(filename, 'r')
        dct = json.load(f)
        f.close()
        labels = dct['fmax'].keys()
        if label is not None and len(labels) == 1:
            for key in ('fmax', 'walltime', 'energy_evals', 'energy_count'):
                dct[key][label] = dct[key][labels[0]]
                del dct[key][labels[0]]
        if not append:
            self.fmax = {}
            self.walltime = {}
            self.energy_evals = {}
            self.energy_count = {}
        self.fmax.update(dct['fmax'])
        self.walltime.update(dct['walltime'])
        self.energy_evals.update(dct['energy_evals'])
        self.energy_count.update(dct['energy_count'])

    def tabulate(self):
        fmt1 = '%-10s %10s %10s %8s'
        title = fmt1 % ('Label', '# Force', '# Energy', 'Walltime/s')
        print(title)
        print('-' * len(title))
        fmt2 = '%-10s %10d %10d %8.2f'
        for label in sorted(self.fmax.keys()):
            print(fmt2 % (label, len(self.fmax[label]),
                          len(self.energy_count[label]),
                          self.walltime[label][-1] - self.walltime[label][0]))

    def plot(self, fmaxlim=(1e-2, 1e2), forces=True, energy=True,
             walltime=True,
             markers=None, labels=None, **kwargs):
        import matplotlib.pyplot as plt

        if markers is None:
            markers = [c + s for c in ['r', 'g', 'b', 'c', 'm', 'y', 'k']
                       for s in ['.-', '.--']]
        nsub = sum([forces, energy, walltime])
        nplot = 0

        if labels is not None:
            fmax_values = [v for (k, v) in sorted(zip(self.fmax.keys(),
                                                      self.fmax.values()))]
            self.fmax = dict(zip(labels, fmax_values))

            energy_count_values = [v for (k, v) in
                                   sorted(zip(self.energy_count.keys(),
                                              self.energy_count.values()))]
            self.energy_count = dict(zip(labels, energy_count_values))

            walltime_values = [v for (k, v) in
                               sorted(zip(self.walltime.keys(),
                                          self.walltime.values()))]
            self.walltime = dict(zip(labels, walltime_values))

        if forces:
            nplot += 1
            plt.subplot(nsub, 1, nplot)
            for label, color in zip(sorted(self.fmax.keys()), markers):
                fmax = np.array(self.fmax[label])
                idx = np.arange(len(fmax))
                plt.semilogy(idx, fmax, color, label=label, **kwargs)

            plt.xlabel('Number of force evaluations')
            plt.ylabel('Maximum force / eV/A')
            plt.ylim(*fmaxlim)
            plt.legend()

        if energy:
            nplot += 1
            plt.subplot(nsub, 1, nplot)
            for label, color in zip(sorted(self.energy_count.keys()), markers):
                energy_count = np.array(self.energy_count[label])
                fmax = np.array(self.fmax[label])
                plt.semilogy(energy_count, fmax, color, label=label, **kwargs)

            plt.xlabel('Number of energy evaluations')
            plt.ylabel('Maximum force / eV/A')
            plt.ylim(*fmaxlim)
            plt.legend()

        if walltime:
            nplot += 1
            plt.subplot(nsub, 1, nplot)
            for label, color in zip(sorted(self.walltime.keys()), markers):
                walltime = np.array(self.walltime[label])
                fmax = np.array(self.fmax[label])
                walltime -= walltime[0]
                plt.semilogy(walltime, fmax, color, label=label, **kwargs)

            plt.xlabel('Walltime / s')
            plt.ylabel('Maximum force / eV/A')
            plt.ylim(*fmaxlim)
            plt.legend()

        plt.subplots_adjust(hspace=0.33)
