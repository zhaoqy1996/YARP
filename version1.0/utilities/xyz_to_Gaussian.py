#!/bin/env python
#
# This script is used to take in xyz file and generate input file for Gaussian (.gjf file)
# Author: Qiyuan 
import sys,argparse,random,os
from copy import deepcopy

# Add TAFFY Lib to path
from taffi_functions import xyz_parse

def main(argv):

    parser = argparse.ArgumentParser(description='Converts an xyz into a Gaussian input file. Allows for the randomization of atom positions')

    #required (positional) arguments                                                                                                  
    parser.add_argument('coord_file', help = 'The input file')

    #optional arguments
    parser.add_argument('-p', dest='proc_num', default=25,
                        help = 'Sets the number of processors to run the Gaussian job on (default: 25(None))')

    parser.add_argument('-r', dest='random_factor', default=0.0,
                        help = 'Scaling of the position random number. Everytime an atom is placed, a random number, R, between 1 and 0 is drawn, and the positions moved by '+\
                               'R*scaling factor. By default scaling_fact=0.0 and the positions are copied as in the .xyz file. (default: 0.0)')

    parser.add_argument('-o', dest='output_name', default=[],
                        help = 'Determines the name of the output file. If none is supplied then the default behavior is to use the base of the input filename (default: base_filename)')

    parser.add_argument('-b', dest='basis', default=None,
                        help = 'Sets the basis set for the calculation. 6-311G(d) is default of Gn method, check for http://gaussian.com/basissets/?tabid=0')

    parser.add_argument('-t', dest='job_title', default='Energy',
                        help = 'Job types in Gaussian, e.g SP, Opt... look up http://gaussian.com/basissets/?tabid=0')

    parser.add_argument('-ty', dest='job_type', default='G4',
                        help = 'Job types in Gaussian, e.g SP, Opt... look up http://gaussian.com/basissets/?tabid=0')

    parser.add_argument('-q', dest='charge', default=0,
                        help = 'Sets the total charge for the calculation. (default: 0)')

    parser.add_argument('-m', dest='multiplicity', default=1,
                        help = 'Sets the multiplicity for the calculation. (default: 1)')

    parser.add_argument('-c', dest='check_file', default=True,
                        help = 'use check file or not. (default: True)')


    # Make relevant inputs lowercase
    args=parser.parse_args()

    # Parse inputs
    args.proc_num = int(args.proc_num)
    if args.proc_num > 25: print("ERROR: Too many procs requested, Exiting...");quit()
    
    args.random_factor = float(args.random_factor)
    if args.output_name != []:
        args.output_name = str(args.output_name)
    else:
        args.output_name = '.'.join([ i for i in args.coord_file.split('.')[:-1]])+'.gjf'

    if str(args.check_file).lower() == 'false':
        args.check_file = False
    else: 
        args.check_file = True

    args.charge = int(args.charge)
    args.multiplicity = int(args.multiplicity)
    args.ty = str(args.job_type)
    args.t = str(args.job_title)
    
    # Check that the input is an .xyz file. 
    if args.coord_file.split('.')[-1] != 'xyz':
        print("ERROR: Check to ensure that the input file is in .xyz format.")
        return
    
    # Extract Element list and Coord list from the file
    Elements,Geometry = xyz_parse(args.coord_file)

    # Write input file
    with open(args.output_name,'w') as f:
        if args.check_file:
            f.write("%RWF=myrwf\n")
            f.write("%NoSave\n")
            f.write("%Chk=mychk\n")

        if args.proc_num < 25:
            f.write("%NProcShared={}\n".format(args.proc_num))
        
        if args.basis == None:
          f.write("# {}\n".format(args.ty))
        else: 
          f.write("# {}/{}\n".format(args.ty,args.basis))

        f.write("\n{}\n".format(args.t))
        f.write("\n{} {}\n".format(args.charge,args.multiplicity))

        for count_i,i in enumerate(Geometry):
            f.write("{:3s}".format(Elements[count_i]))
            for j in i:
                f.write(" {:< 16.8f}".format((random.random()*2.0-1.0)*args.random_factor+j))
            f.write("\n")
        f.write("\n")

if __name__ == "__main__":
   main(sys.argv[1:])
