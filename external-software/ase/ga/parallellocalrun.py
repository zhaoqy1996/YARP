""" Class for handling several simultaneous jobs.
    The class has been tested on linux and Mac OS X.
"""
from subprocess import Popen, PIPE
import os
import time
from ase.io import write, read


class ParallelLocalRun(object):

    """ Class that allows for the simultaneous relaxation of
         several candidates on the same computer.
        The method is based on starting each relaxation with an
         external python script and then monitoring when the
         relaxations are done adding in the resulting structures
         to the database.

        Parameters:
         data_connection: DataConnection object.
         tmp_folder: Folder for temporary files
         n_simul: The number of simultaneous relaxations.
         calc_script: Reference to the relaxation script.
    """

    def __init__(self, data_connection, tmp_folder,
                 n_simul, calc_script):
        self.dc = data_connection
        self.n_simul = n_simul
        self.calc_script = calc_script
        self.tmp_folder = tmp_folder
        self.running_pids = []

    def get_number_of_jobs_running(self):
        """ Returns the number of jobs running.
             It is a good idea to check that this is 0 before
             terminating the main program. """
        self.__cleanup__()
        return len(self.running_pids)

    def relax(self, a):
        """ Relax the input atoms object a. If n_simul relaxations
             are already running the function sleeps until a processor
             becomes available.
        """
        self.__cleanup__()

        # Wait until a thread is available.
        while len(self.running_pids) >= self.n_simul:
            time.sleep(2.)
            self.__cleanup__()

        # Mark the structure as queued and run the external py script.
        self.dc.mark_as_queued(a)
        if not os.path.isdir(self.tmp_folder):
            os.mkdir(self.tmp_folder)
        fname = '{0}/cand{1}.traj'.format(self.tmp_folder,
                                          a.info['confid'])
        write(fname, a)
        p = Popen(['python', self.calc_script, fname])
        self.running_pids.append([a.info['confid'], p.pid])

    def __cleanup__(self):
        """ Checks if any relaxations are done and load in the structure
            from the traj file. """
        p = Popen(['ps -x -U `whoami`'], shell=True,
                  stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True,
                  universal_newlines=True)
        (_, fout) = (p.stdin, p.stdout)
        lines = fout.readlines()
        lines = [l for l in lines if l.find('defunct') == -1]

        stopped_runs = []
        for i in range(len(self.running_pids) - 1, -1, -1):
            found = False
            for l in lines:
                if l.find(str(self.running_pids[i][1])) != -1:
                    found = True
                    break
            if not found:
                stopped_runs.append(self.running_pids.pop(i))

        # All processes not running any more must be complete and should
        # be loaded in.
        for (confid, _) in stopped_runs:
            try:
                tf = self.tmp_folder
                a = read('{0}/cand{1}_done.traj'.format(tf,
                                                        confid))
                self.dc.add_relaxed_step(a)
            except IOError as e:
                print(e)
