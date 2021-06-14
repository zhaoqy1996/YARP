from __future__ import print_function

from ase.optimize.optimize import Optimizer
import numpy as np
from scipy.optimize import minimize

from ase.parallel import rank

from ase.optimize.gpmin.gp import GaussianProcess
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.prior import ConstantPrior

import pickle

class GPMin(Optimizer, GaussianProcess):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None, prior=None,
                 master=None, noise=0.005, weight=1., update_prior_strategy='maximum',
                 scale=0.4, force_consistent=None, batch_size=5,
                 update_hyperparams=False):


        """Optimize atomic positions using GPMin algorithm, which uses
        both potential energies and forces information to build a PES
        via Gaussian Process (GP) regression and then minimizes it.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store the training set. If set, file with
            such a name will be searched and the data in the file incorporated
            to the new training set, if the file exists.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        master: boolean
            Defaults to None, which causes only rank 0 to save files. If
            set to True, this rank will save files.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.

        prior: Prior object or None
            Prior for the GP regression of the PES surface
            See ase.optimize.gpmin.prior
            If *Prior* is None, then it is set as the
            ConstantPrior with the constant being updated
            using the update_prior_strategy specified as a parameter

        noise: float
            Regularization parameter for the Gaussian Process Regression.

        weight: float
            Prefactor of the Squared Exponential kernel.
            If *update_hyperparams* is False, changing this parameter
            has no effect on the dynamics of the algorithm.

        update_prior_strategy: string
            Strategy to update the constant from the ConstantPrior
            when more data is collected. It does only work when
            Prior = None

            options:
                'maximum': update the prior to the maximum sampled energy
                'init' : fix the prior to the initial energy
                'average': use the average of sampled energies as prior

        scale: float
            scale of the Squared Exponential Kernel

        update_hyperparams: boolean
            Update the scale of the Squared exponential kernel
            every batch_size-th iteration by maximizing the
            marginal likelhood.

        batch_size: int
            Number of new points in the sample before updating
            the hyperparameters.
            Only relevant if the optimizer is executed in update
            mode: (update = True)
        """

        self.nbatch = batch_size
        self.strategy = update_prior_strategy
        self.update_hp = update_hyperparams
        self.function_calls = 1
        self.force_calls = 0
        self.x_list = []      # Training set features
        self.y_list = []      # Training set targets

        Optimizer.__init__(self, atoms, restart, logfile,
                           trajectory, master, force_consistent)

        if prior is None:
            self.update_prior = True
            prior = ConstantPrior(constant = None) 

        else:
            self.update_prior = False

        Kernel = SquaredExponential()
        GaussianProcess.__init__(self, prior, Kernel)

        self.set_hyperparams(np.array([weight, scale, noise]))

    def acquisition(self, r):
        e = self.predict(r)

        return e[0], e[1:]

    def update(self, r, e, f):
        """Update the PES:
        update the training set, the prior and the hyperparameters.
        Finally, train the model """

        # update the training set
        self.x_list.append(r)
        f = f.reshape(-1)
        y = np.append(np.array(e).reshape(-1), -f)
        self.y_list.append(y)

        # Set/update the constant for the prior
        if self.update_prior:
            if self.strategy == 'average':
                av_e = np.mean(np.array(self.y_list)[:, 0])
                self.prior.set_constant(av_e)
            elif self.strategy == 'maximum':
                max_e = np.max(np.array(self.y_list)[:, 0])
                self.prior.set_constant(max_e)
            elif self.strategy == 'init':
                self.prior.set_constant(e)
                self.update_prior = False

        # update hyperparams
        if self.update_hp and self.function_calls % self.nbatch == 0 and self.function_calls != 0:
            self.fit_to_batch()

        # build the model
        self.train(np.array(self.x_list), np.array(self.y_list))

    def relax_model(self, r0):

        result = minimize(self.acquisition, r0, method='L-BFGS-B', jac=True)

        if result.success:
            return result.x
        else:
            self.dump()
            raise RuntimeError(
                "The minimization of the acquisition function has not converged")

    def fit_to_batch(self):
        '''Fit hyperparameters and collect exception'''
        try:
            self.fit_hyperparameters(np.asarray(
                self.x_list), np.asarray(self.y_list))
        except Exception:
            pass

    def step(self, f):

        atoms = self.atoms
        r0 = atoms.get_positions().reshape(-1)
        e0 = atoms.get_potential_energy(force_consistent=self.force_consistent)
        self.update(r0, e0, f)

        r1 = self.relax_model(r0)
        self.atoms.set_positions(r1.reshape(-1, 3))
        e1 = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent)
        f1 = self.atoms.get_forces()

        self.function_calls += 1
        self.force_calls += 1

        count = 0
        while e1 >= e0:

            self.update(r1, e1, f1)
            r1 = self.relax_model(r0)

            self.atoms.set_positions(r1.reshape(-1, 3))
            e1 = self.atoms.get_potential_energy(
                force_consistent=self.force_consistent)
            f1 = self.atoms.get_forces()

            self.function_calls += 1
            self.force_calls += 1

            if self.converged(f1):
                break

            count += 1
            if count == 30:
                raise RuntimeError('A descent model could not be built')
        self.dump()

    def dump(self):
        '''Save the training set'''
        if rank == 0 and self.restart is not None:
            with open(self.restart, 'wb') as fd:
                pickle.dump((self.x_list, self.y_list), fd, protocol = 2)

    def read(self):
        self.x_list, self.y_list = self.load()

