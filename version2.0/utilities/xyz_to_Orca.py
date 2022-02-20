#!/bin/env python                                                                                                                                                             
# Author: Brett Savoie (brettsavoie@gmail.com)

from numpy import *
import sys,argparse,subprocess,random,os
from subprocess import PIPE
from scipy.spatial.distance import *
from copy import deepcopy

# Add TAFFY Lib to path
from taffi_functions import xyz_parse

def main(argv):

    parser = argparse.ArgumentParser(description='Converts an xyz into an orca geometry optimization input file. Allows for the randomization of atom positions')

    #required (positional) arguments                                                                                                  
    parser.add_argument('coord_file', help = 'The input file. (currently must be an xyz with the atom types in the fourth column and the ion listed last.')

    #optional arguments
    parser.add_argument('-p', dest='proc_num', default=1, help = 'Sets the number of processors to run the orca job on (default: 1)')

    parser.add_argument('-r', dest='random_factor', default=0.0,
                        help = 'Scaling of the position random number. Everytime an atom is placed, a random number, R, between 1 and 0 is drawn, and the positions moved by '+\
                               'R*scaling factor. By default scaling_fact=0.0 and the positions are copied as in the .xyz file. (default: 0.0)')

    parser.add_argument('-o', dest='output_name', default=[],
                        help = 'Determines the name of the output file. If none is supplied then the default behavior is to use the base of the input filename (default: base_filename)')

    parser.add_argument('-q', dest='charge', default=0, help = 'Sets the total charge for the calculation. (default: 0)')

    parser.add_argument('-f', dest='functional', default='wB97X-D3', help = 'Sets the functional for the calculation. (default: wB97X-D3; other typical options are B3LYP, and M062X)')

    parser.add_argument('-b', dest='basis', default=None, help = 'Sets the basis set for the calculation. (default: None)')

    parser.add_argument('-d', dest='dispersion', default=None, help = 'Sets the dispersion for the calculation. (default: None, available: D2, D3ZERO, D3BJ, D4')

    parser.add_argument('-m', dest='multiplicity', default=1, help = 'Sets the multiplicity for the calculation. (default: 1)')

    parser.add_argument('-ty', dest='job_type', default='Opt', help = 'Job types in Gaussian, e.g SP, Opt, OptTS look up Orca user manual for help')

    parser.add_argument('-mem', dest='memory', default=1000, help = 'Specifies the memory use of each cpu (recommend no more than NMEM/NCPU for each cluster)')

    parser.add_argument('-s', dest='IRC_step', default=60, help = 'Maximum IRC steps')

    parser.add_argument('-hs', dest='hessian_step', default=5, help = 'Frequency of hessian recalc')

    parser.add_argument('--RIJCOSX', dest='RIJCOSX_option', default="", action='store_const', const="RIJCOSX",
                        help = 'When this flag is present, the RIJCOX approximation is enabled. This is usually an excellent approximation and greatly speeds up hybrid calculations. (Default: off)')

    # Make relevant inputs lowercase
    args=parser.parse_args(argv)
    
    # Create fake option for function input
    option = dict(args.__dict__.items())

    # Check that the input is an .xyz file. 
    if args.coord_file.split('.')[-1] != 'xyz':
        print( "ERROR: Check to ensure that the input file is in .xyz format.")
        return
    
    # Create geo_dict for function input
    geo_dict = {}
    Elements,Geometry = xyz_parse(args.coord_file)
    geo_dict["elements"] = Elements
    geo_dict["geo"] = Geometry    
    fun(option,geo_dict)
    
    
def fun(config,geo_dict):

    option = miss2default(config)
    proc_num = option['proc_num']
    random_factor = option['random_factor']
    charge = option['charge']
    functional = option['functional']
    basis = option['basis']
    dispersion = option['dispersion']
    multiplicity = option['multiplicity']
    job_type = option["job_type"]
    IRC_step = option["IRC_step"]
    RIJCOSX_option = option['RIJCOSX_option']
    output_name   = option['output_name']
    memory        = int(option['memory'])
    hessian_step  = int(option['hessian_step'])

    # Parse inputs from config
    proc_num = int(proc_num)
    random_factor = float(random_factor)
    charge = int(charge)
    multiplicity = int(multiplicity)

    # determine the level of theory
    level = functional
    if dispersion is not None: level += ' {}'.format(dispersion)
    if basis is not None: level +=' {}'.format(basis)

    # Extract Element list and Coord list from the file
    Elements = geo_dict["elements"]
    Geometry = geo_dict["geo"]

    # Write input file
    with open("{}.in".format(output_name),'w') as f:
        if 'xtb' in functional.lower() and 'Freq' in job_type: job_type=job_type.replace('Freq','NumFreq')
        f.write("! {} TIGHTSCF {} {}\n".format(level,job_type,RIJCOSX_option))
        f.write("\n%scf\nMaxIter 300\nend\n")
        f.write("\n%pal\nnproc {}\nend\n".format(proc_num))
        
        # sepcial for TS-opt
        if 'optts' in job_type.lower():
            if 'xtb' in functional.lower(): f.write('\n%geom\nCalc_Hess true\nNumHess true\nRecalc_Hess 8\nend\n')
            else: f.write('\n%geom\nCalc_Hess true\nRecalc_Hess {}\nend\n'.format(hessian_step))

        # sepcial for IRC
        if 'xtb' in functional.lower(): IRCcommand = '\nInitHess calc_numfreq'
        else: IRCcommand = '\nInitHess calc_anfreq'

        if 'irc' in job_type.lower():
            f.write('\n%irc\nMaxIter {}\nPrintLevel 1\nDirection both\nFollow_CoordType cartesian{}\nScale_Displ_SD 0.15\nAdapt_Scale_Displ true\nTolRMSG 5.e-4\nTolMaxG 2.e-3\nend\n'.format(IRC_step,IRCcommand))
            
        if 'calc_numfreq' in IRCcommand:
            f.write('\n%freq\nCentralDiff true\nend\n')
            
        # Write job name and geometry
        f.write("\n%maxcore {}\n".format(memory*int(proc_num)))
        f.write("\n%base \"{}\"\n\n* xyz {} {}\n".format(output_name.split('/')[-1],charge,multiplicity))
        for count_i,i in enumerate(Geometry):
            f.write("  {:3s}".format(Elements[count_i]))
            for j in i:
                f.write(" {:< 16.8f}".format((random.random()*2.0-1.0)*random_factor+j))
            f.write("\n")
        f.write("*\n")

def miss2default(config):
 
    # Create default dict
    options = ['proc_num','random_factor','output_name','charge','functional','basis','dispersion','multiplicity','job_type','IRC_step','RIJCOSX_option','memory']

    defaults = [1, 0.0, 'orca', 0, 'B3LYP', None, None, 1, 'Opt', 40, '', 1000]

    N_position = int (len(options) - len(defaults))

    default = {}
    for count_i,i in enumerate(defaults):
        default[options[N_position + count_i]] = i

    # Combine config
    option = {}
    for key in config:
        option[key] = config[key]

    missing = [ i in option for i in options]
        
    # set missing option to default
    for count_i,i in enumerate(missing):
        if i is False:
            option[options[count_i]] = default[options[count_i]]

    return option

if __name__ == "__main__":
   main(sys.argv[1:])
