# All of the function have to call xtb 
import numpy as np
from subprocess import Popen, PIPE
import shutil
import sys,os
import time

# Add TAFFY Lib to path
from taffi_functions import *

######## xtb -h info ##############
"""Construct xTB calculator.
    The keyword arguments (kwargs) can be one of the ASE standard keywords: 'acc', 'iterations' 
    and 'opt [level]', detail is shown as following :

    -c,--chrg INT     specify molecular charge as INT
    -u,--uhf INT      specify Nalpha-Nbeta as INT (spin)
    -a,--acc REAL     accuracy for SCC calculation, lower is better (default = 1.0)
    --iteration INT number of iterations in SCC (default = 250)
    --cycles       number of cycles in ANCopt (default = automatic)
    --gfn INT      specify parametrisation of GFN-xTB (default = 2)
    --qmdff        use QMDFF for single point (needs solvent-file)
    --etemp REAL   electronic temperature (default = 300K)
    -g,--gbsa [SOLVENT [STATE]] generalized born (GB) model with solvent accessable surface area (SASA) model
    --pop          requests printout of Mulliken population analysis
    --dipole      requests dipole printout
    --wbo          requests Wiberg bond order printout
    --lmo          requests localization of orbitals
    --scc, --sp    performs a single point calculation
    --esp          calculate electrostatic potential on VdW-grid
    --grad         performs a gradient calculation
    -o,--opt [LEVEL]  call ancopt(3) to perform a geometry optimization,
                   levels from crude, sloppy, loose, normal (default),tight, verytight to extreme can be chosen
    --hess         perform a numerical hessian calculation on input geometry
    --ohess [LEVEL] perform a numerical hessian calculation on an ancopt(3) optimized geometry
    --siman        conformational search by simulated annealing based on  molecular dynamics. Conformers are optimized with ancopt.
    -I,--input FILE   use FILE as input source for xcontrol(7) instructions
    --namespace STRING give this xtb(1) run a namespace on all files, even temporary ones, will be named accordingly
    --[no]restart  restarts calculation from xtbrestart (default = true)
    -P,--parallel INT number of parallel processes
"""

################################################################################################################################################
#
################################################### Function to obtain single point energy  ####################################################
# 
################################################################################################################################################

def xtb_energy(xyz_file,charge=0,unpair=0,niter=100,accuracy=1.0,namespace='xTB',workdir='.',method='sp',etemp=None):
    current_dir = os.getcwd()
    # change to xTB working directory
    os.chdir(workdir)
    if etemp is None:
        substring= "xtb -c {} -u {} -a {} --iterations {} --{} --namespace '{}' {}"
        code_exe = substring.format(charge,unpair,accuracy,niter,method,namespace,xyz_file)
    else:
        substring= "xtb -c {} -u {} -a {} --iterations {} --{} --etemp {} --namespace '{}' {}"
        code_exe = substring.format(charge,unpair,accuracy,niter,method,etemp,namespace,xyz_file)

    output = Popen(code_exe, shell=True, stdout=PIPE, stderr=PIPE).communicate()[0]
    output = output.decode('utf-8')
    
    with open("{}_xTB-Energy.txt".format(namespace),"w") as g:
        g.write(output)
        
    with open("{}_xTB-Energy.txt".format(namespace),"r") as g:
        for lc,lines in enumerate(g):
            fields=lines.split()
            if len(fields)==6 and fields[1] =="TOTAL" and fields[2] == "ENERGY":
                Energy = float(fields[3])

    Popen("rm {}.*".format(namespace),shell=True)            
    os.chdir(current_dir)
    return Energy

################################################################################################################################################
#
##################################################### Function to obtain optimized geometry ####################################################
# 
################################################################################################################################################
def xtb_geo_opt(xyz_file,charge=0,unpair=0,niter=100,accuracy=1.0,namespace='xTB',workdir='.',level='normal',fixed_atoms=[],output_xyz=None,cleanup=False):

    current_dir = os.getcwd()

    # change to xTB working directory
    os.chdir(workdir)

    # write input file is needed
    if len(fixed_atoms) > 0:
        with open('{}-xtb.inp'.format(namespace),'w') as f:
            f.write('$fix\n')
            for ind in fixed_atoms:
                f.write('atoms: {}\n'.format(ind))
            f.write('$end\n')

        # generate command line
        substring= "xtb -c {} -u {} -a {} --input {}-xtb.inp --iterations {} --opt {} --namespace '{}' {}"
        code_exe = substring.format(charge,unpair,accuracy,namespace,niter,level,namespace,xyz_file)
        
    else:
        # generate command line
        substring= "xtb -c {} -u {} -a {} --iterations {} --opt {} --namespace '{}' {}"
        code_exe = substring.format(charge,unpair,accuracy,niter,level,namespace,xyz_file)

    # run xtb
    output = Popen(code_exe, shell=True, stdout=PIPE, stderr=PIPE).communicate()[0]
    output = output.decode('utf-8')
    Energy = 0.0
    with open("{}_xTB-opt.txt".format(namespace),"w") as g: g.write(output)
        
    with open("{}_xTB-opt.txt".format(namespace),"r") as g:
        for lc,lines in enumerate(g):
            fields=lines.split()
            if len(fields)==6 and fields[1] =="TOTAL" and fields[2] == "ENERGY":
                Energy = float(fields[3])

    opt_xyz_file = "{}/{}.xtbopt.xyz".format(workdir,namespace)
    if Energy != 0.0:
    
        # clean up the xtb files is the flag is on
        if cleanup:
            if os.path.isfile("{}/{}.xtbopt.xyz".format(current_dir,namespace)) is False:
                shutil.move(opt_xyz_file,current_dir)
            Popen("rm {}.*".format(namespace),shell=True)            
            time.sleep(0.1)
            if os.path.isfile("{}/{}.xtbopt.xyz".format(workdir,namespace)) is False:
                shutil.move("{}/{}.xtbopt.xyz".format(current_dir,namespace),workdir)

        # change back to original folder
        os.chdir(current_dir)

        # copy the optput xyz file to given path if is needed
        if output_xyz is not None:
            shutil.copy2(opt_xyz_file,output_xyz)
            print("Geometry optimization is done at xtb level with single point energy:{} and resulting xyz file {}".format(Energy,output_xyz))
            return Energy,output_xyz,True

        else:
            print("Geometry optimization is done at xtb level with single point energy:{} and resulting xyz file {}".format(Energy,opt_xyz_file))
            return Energy,opt_xyz_file,True
    else:
        print("xTB Geo-opt fails")
        # change back to original folder
        os.chdir(current_dir)
        return Energy,opt_xyz_file,False

################################################################################################################################################
# Function to calculate enthalpy of formation...
################################################################################################################################################
'''
# test for energy calculation
Energy = xtb_energy('/home/zhao922/bin/DES/xyz-input/0_opt.xyz',namespace='reactant',workdir='/home/zhao922/bin/DES/xtb_folder',method='sp')
print(Energy)

# test for geo-opt
Energy,opt_geo = xtb_geo_opt('/home/zhao922/bin/DES/reaction-test/xyz-input/9-21-med.xyz',namespace='test',workdir='/home/zhao922/bin/DES/xtb_folder',cleanup=True)
print(Energy)
E,G = xyz_parse(opt_geo)
print(G)

'''
