""" Class for handling interaction with the PBS queuing system."""
from ase.io import write
import os
from ase.io.trajectory import Trajectory
from subprocess import Popen, PIPE
import time


class PBSQueueRun(object):

    """ Class for communicating with the commonly used PBS queing system
         at a computer cluster.

        The user needs to supply a job file generator which takes
        as input a job name and the relative path to the traj
        file which is to be locally optimized. The function returns
        the job script as text.
        If the traj file is called f the job must write a file
        f[:-5] + '_done.traj' which is then read by this object.

       Parameters:

       data_connection: The DataConnection object.
       tmp_folder: Temporary folder for all calculations
       job_prefix: Prefix of the job submitted. This identifier is used
       to determine how many jobs are currently running.
       n_simul: The number of simultaneous jobs to keep in the queuing system.
       job_template_generator: The function generating the job file.
       This function should return the content of the job file as a
       string.
       qsub_command: The name of the qsub command (default qsub).
       qstat_command: The name of the qstat command (default qstat).
    """
    def __init__(self, data_connection, tmp_folder, job_prefix,
                 n_simul, job_template_generator,
                 qsub_command='qsub', qstat_command='qstat',
                 find_neighbors=None, perform_parametrization=None):
        self.dc = data_connection
        self.job_prefix = job_prefix
        self.n_simul = n_simul
        self.job_template_generator = job_template_generator
        self.qsub_command = qsub_command
        self.qstat_command = qstat_command
        self.tmp_folder = tmp_folder
        self.find_neighbors = find_neighbors
        self.perform_parametrization = perform_parametrization
        self.__cleanup__()

    def relax(self, a):
        """ Add a structure to the queue. This method does not fail
            if sufficient jobs are already running, but simply
            submits the job. """
        self.__cleanup__()
        self.dc.mark_as_queued(a)
        if not os.path.isdir(self.tmp_folder):
            os.mkdir(self.tmp_folder)
        fname = '{0}/cand{1}.traj'.format(self.tmp_folder,
                                          a.info['confid'])
        write(fname, a)
        job_name = '{0}_{1}'.format(self.job_prefix, a.info['confid'])
        f = open('tmp_job_file.job', 'w')
        f.write(self.job_template_generator(job_name, fname))
        f.close()
        os.system('{0} tmp_job_file.job'.format(self.qsub_command))

    def enough_jobs_running(self):
        """ Determines if sufficient jobs are running. """
        return self.number_of_jobs_running() >= self.n_simul

    def number_of_jobs_running(self):
        """ Determines how many jobs are running. The user
            should use this or the enough_jobs_running method
            to verify that a job needs to be started before
            calling the relax method."""
        self.__cleanup__()
        p = Popen(['`which {0}` -u `whoami`'.format(self.qstat_command)],
                  shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE,
                  close_fds=True, universal_newlines=True)
        fout = p.stdout
        lines = fout.readlines()
        n_running = 0
        for l in lines:
            if l.find(self.job_prefix) != -1:
                n_running += 1
        return n_running

    def __cleanup__(self):
        """ Tries to load in structures previously
            submitted to the queing system. """
        confs = self.dc.get_all_candidates_in_queue()
        for c in confs:
            fdone = '{0}/cand{1}_done.traj'.format(self.tmp_folder,
                                                   c)
            if os.path.isfile(fdone) and os.path.getsize(fdone) > 0:
                try:
                    a = []
                    niter = 0
                    while len(a) == 0 and niter < 5:
                        t = Trajectory(fdone, 'r')
                        a = [ats for ats in t]
                        if len(a) == 0:
                            time.sleep(1.)
                        niter += 1
                    if len(a) == 0:
                        txt = 'Could not read candidate ' + \
                            '{0} from the filesystem'.format(c)
                        raise IOError(txt)
                    a = a[-1]
                    a.info['confid'] = c
                    self.dc.add_relaxed_step(
                        a,
                        find_neighbors=self.find_neighbors,
                        perform_parametrization=self.perform_parametrization)
                except IOError as e:
                    print(e)
