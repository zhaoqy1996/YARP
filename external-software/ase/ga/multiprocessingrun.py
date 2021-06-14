""" Class for handling several simultaneous jobs.
The class has been tested on Niflheim-opteron4.
"""
from multiprocessing import Pool
import time
from ase.io import write, read


class MultiprocessingRun(object):
    """Class that allows for the simultaneous relaxation of
    several candidates on a cluster. Best used if each individual
    calculation is too small for using a queueing system.

    Parameters:
    
    data_connection: DataConnection object.
    
    tmp_folder: Folder for temporary files.
    
    n_simul: The number of simultaneous relaxations.
    
    relax_function: The relaxation function. This needs to return
    the filename of the relaxed structure.
    """
    def __init__(self, data_connection, relax_function,
                 tmp_folder, n_simul=None):
        self.dc = data_connection
        self.pool = Pool(n_simul)
        self.relax_function = relax_function
        self.tmp_folder = tmp_folder
        self.results = []

    def relax(self, a):
        """Relax the atoms object a by submitting the relaxation
        to the pool of cpus."""
        self.dc.mark_as_queued(a)
        fname = '{0}/cand{1}.traj'.format(self.tmp_folder,
                                          a.info['confid'])
        write(fname, a)
        self.results.append(self.pool.apply_async(self.relax_function,
                                                  [fname]))
        self._cleanup()
        
    def _cleanup(self):
        for r in self.results:
            if r.ready() and r.successful():
                fname = r.get()
                a = read(fname)
                self.dc.add_relaxed_step(a)
                self.results.remove(r)
                
    def finish_all(self):
        """Checks that all calculations are finished, if not
        wait and check again. Return when all are finished."""
        while len(self.results) > 0:
            self._cleanup()
            time.sleep(2.)
