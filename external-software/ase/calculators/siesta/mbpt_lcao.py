from __future__ import division
import numpy as np
import os
from ase.utils import basestring


class MBPT_LCAO:
    """
      Calculator for mbpt_lcao program, see
      http://mbpt-domiprod.wikidot.com/
      contains:
        __init__
        write_tddft_inp
        run_mbpt_lcao

      PARAMETERS
      -----------
      mbpt_inp (dict): dictionary containing the input of the mbpt_lcao program

      Take into the kwargs argument the input for the program,
      see ase/ase/test/siesta/mbpt_lcao/script_mbpt_lcao.py for a complete
      example.
    """

    def __init__(self, mbpt_inp):

        self.param = mbpt_inp
        self.command = os.environ.get('MBPT_COMMAND')
        if self.command is None:
            mess = "The 'MBPT_COMMAND' environment is not defined."
            raise ValueError(mess)

    def write_tddft_inp(self):
        """
        Write the input file tddft_lr.inp for the mbpt_lcao program
        """

        if len(self.param.keys()) == 0:
            raise ValueError('Can not write mbpt_lcao input, dict empty')

        f = open('tddft_lr.inp', 'w')

        for k, v in self.param.items():
            if isinstance(v, np.ndarray):
                f.write(k + '  {0}  {1}  {2}\n'.format(v[0], v[1], v[2]))
            elif isinstance(v, basestring):
                f.write(k + '      ' + v + '\n')
            elif k == 'group_species' or k == 'species_iter':
                gp = '{'
                for k1, v1 in v.items():
                    gp = gp + '{0}: ['.format(k1)
                    for w in range(len(v1) - 1):
                        gp = gp + str(v1[w]) + ', '
                    if k1 < max(v.keys()):
                        gp = gp + str(v1[len(v1) - 1]) + '], '
                    else:
                        gp = gp + str(v1[len(v1) - 1]) + ']'

                gp = gp + '}\n'
                f.write(k + '     ' + gp)
            else:
                f.write(k + '     {0}\n'.format(v))

        f.close()

    def run_mbpt_lcao(self, output_name='mbpt_lcao.out', write_inp=False):
        """
        run mbpt_lcao
        Parameters
        ----------
        output_name : str, optional
            name of the output file, defualt: mbpt_lcao.out

        write_inp : bool, optional
            write the tddft_lr,inp file before to run the program, by default
            False
        """

        import subprocess

        if write_inp:
            self.write_tddft_inp()

        try:
            self.command = self.command % output_name
        except TypeError:
            raise ValueError("The 'MBPT_COMMAND' environment must " +
                             "be a format string" +
                             " with one string arguments.\n" +
                             "Example : 'mbpt > ./%s'.\n" +
                             "Got '%s'" % self.command)

        subprocess.call(self.command, shell=True)
