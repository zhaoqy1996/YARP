# This file contains fuction to submit jobs to clusters
#!/bin/env python                                             
import os,sys,subprocess,argparse,fnmatch
import numpy as np

def main(argv):
    parser = argparse.ArgumentParser(description='submit all files matching the -n string.')
    parser.add_argument('-n',dest='job_name',default=None,help = 'The program will submit typical job')
    parser.add_argument('-f', dest='File_type', default='*.submit',help = 'The program submit all files within a typical type')
    parser.add_argument('-sched',dest='sched',default='torque-halstead',help= 'which scheduler is using')
    parser.add_argument('-d', dest='path', default=None,
                        help = 'The program operates on all files discovered during a directory walk that match the -f argument. Optionally, a directory name or any string can \
                                also be supplied via this argument and only files matching -f whose directory string includes -d will be operated on. (default: "")')
    parser.add_argument('--single',dest='single',default=False,action='store_const',const=True,
                        help = 'When this flag on, only submit typical jobs')

    args=parser.parse_args()
    Filename = str(args.File_type)
    if args.single is False:
        Files = {}    
        if args.path is None: 
            Files[Filename] = [ os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames if fnmatch.fnmatch(f,Filename) ]
        else:
            args.path = args.path.split()
            Files[Filename] = [ os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames if (fnmatch.fnmatch(f,Filename) and True in [ i in dp for i in args.path ]) ]
        for i in Files.keys():
            Current_files = sorted(Files[i])
            for j in Current_files:
                subprocess.call(['chmod', '777', j])
                if args.sched == 'torque-halstead':
                    subprocess.call(['qsub', j])
                if args.sched == 'slurm':
                    subprocess.call(['sbatch', j])

    else:
        args.name=str(args.job_name)
        subprocess.call(['chmod', '777', args.name])
        if args.sched == 'torque-halstead':
            subprocess.call(['qsub',args.name])
        if args.sched == 'slurm':
            subprocess.call(['sbatch',args.name])


# Description: generate submit file from a given list of reactant/product pairs
#
# input:   pair_list: a list of xyzfiles that contain geonetry of reactant/product pairs   
#          Njobs    : number of jobs per node
#          Wt       : wall time for each job 
#          queue    : select a queue to submit jobs (standby, bsavoie, etc.)
#          sched    : scheduler argument for the script (for halstead/brown, slurm is the only option)
#
# GSM parameters:  charge      : charge of the system
#                  unpair      : number of unpaired electrons in the system
#                  multiplicity: multiplicity of the system
#                  Nimage      : Number of images for each reaction pathway (9 is default for pyGSM DE-GSM)
#                  conv_tor    : Convergence tolerance for optimizing nodes (default: 0.005)
#                  add_tor     : Convergence tolerance for adding new node (default: 0.01)
#
# output:  output_list : contains all output folders
#
def submit_GSM(pair_list,output_path,Njobs,pygsm_path,level='xtb',procs=1,Wt='4',sched='slurm',queue='standby',charge=0,unpair=0,Nimage=9,conv_tor=0.005,add_tor=0.01,temperature=None,relax_end=True,package='Gaussian'):

    multiplicity = unpair + 1
    Total_jobs= len(pair_list)
    N_bundles = int(np.ceil(Total_jobs/float(Njobs)))
    
    # parse Wall time parameter, determine whether this is in minute of hour
    if type(Wt) == str and "min" in Wt: Wt = int(Wt.split('min')[0]); min_flag = 1
    else: Wt = int(Wt); min_flag = 0

    # get working folder direactory 
    working_dir = os.getcwd()

    # initialize output list 
    output_list = []

    # Write GSM.N.submit files
    for n in range(N_bundles):
        
        # Create input files and submit each bundle
        input_products = pair_list[n * Njobs : (n+1) * Njobs]

        # Submisson file
        if sched == "slurm":    
            with open('GSM.{}.submit'.format(n),'w') as f:                                                                 
                f.write("#!/bin/bash\n")
                f.write("#\n")
                f.write("#SBATCH --job-name=GSM.{}\n".format(n))
                f.write("#SBATCH --output=GSM.{}.out\n".format(n))
                f.write("#SBATCH --error=GSM.{}.err\n".format(n))
                f.write("#SBATCH -A {}\n".format(queue))
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --mem-per-cpu=1G\n")

                # assign procs
                if int(procs) == 1:
                    f.write("#SBATCH --ntasks-per-node={}\n".format(Njobs))
                    f.write("#SBATCH --cpus-per-task=1\n")
                else:
                    f.write("#SBATCH --ntasks-per-node={}\n".format(procs))
                
                # assign wall-time
                if min_flag == 0:
                    f.write("#SBATCH --time {}:00:00\n".format(Wt))
                elif min_flag == 1:
                    f.write("#SBATCH --time 00:{}:00\n".format(Wt))

                # print out information
                f.write("\n# cd into the submission directory\n")
                f.write("cd {}\n".format(working_dir))
                f.write("echo Working directory is ${}\n".format(working_dir))
                f.write("echo Running on host `hostname`\n")
                f.write("echo Start Time is `date`\n")

                if level.lower() not in ['xtb','ani'] and package.lower() == 'orca':
                    # load orca
                    f.write("\n #load orca and openmpi")
                    f.write("\nmodule unload openmpi\n")
                    f.write("export PATH=/depot/bsavoie/apps/orca_5_0_1_openmpi411:$PATH\n")
                    f.write("export LD_LIBRARY_PATH=/depot/bsavoie/apps/orca_5_0_1_openmpi411:$LD_LIBRARY_PATH\n")
                    f.write("export PATH=/depot/bsavoie/apps/openmpi_4_1_1/bin:$PATH\n")
                    f.write("export LD_LIBRARY_PATH=/depot/bsavoie/apps/openmpi_4_1_1/lib:$LD_LIBRARY_PATH\n")

                if level.lower() not in ['xtb','ani'] and package.lower() =='gaussian': 
                    # load gaussian
                    f.write("\nmodule load gaussian16/B.01\n")

                # Create sub-folders
                for product in input_products:
                    name = product.split('/')[-1].split('.xyz')[0]

                    output_folder = output_path+'/'+name
                    if os.path.isdir(output_folder) is False:
                        os.mkdir(output_folder)
                    output_list += [output_folder]

                    f.write("\ncd {}\n".format(output_folder))
                    f.write("# Insert nodes and cores in the header of the input file\n")

                    if level.lower() == 'xtb':
                        with open("{}/xtb_lot.txt".format(output_folder),"w") as g:
                            g.write("# set up parameters\n")
                            g.write("charge \t\t{}\n".format(charge))
                            g.write("spin \t\t{}\n".format(unpair))
                            g.write("namespace \t{}\n".format(name))
                            g.write("calc_type \t{}\n".format("grad"))
                            
                        core_command = ' -package xTB -lot_inp_file xtb_lot.txt '

                    elif level.lower() == 'ani':
                        with open("{}/ani_lot.txt".format(output_folder),"w") as g:
                            g.write("# set up parameters\n")
                            g.write("namespace \t{}\n".format(name))

                        core_command = ' -package ANI -lot_inp_file ani_lot.txt '

                    else:
                        if package.lower() == 'orca':
                            with open("{}/orca_lot.txt".format(output_folder),"w") as g:
                                g.write("# namespace {}\n".format(name))
                                if len(level.split('/')) == 1:
                                    g.write("! {} engrad\n".format(level))
                                else:
                                    g.write("! {} {} engrad\n".format(level.split('/')[0],level.split('/')[1]))

                                if temperature is not None:
                                    g.write("\n%scf\nsmeartemp {}\nend\n".format(temperature))
                                g.write("\n%maxcore 1000\n\n%pal\nnproc {}\nend".format(procs))
                            core_command = ' -package Orca -lot_inp_file orca_lot.txt '

                        elif package.lower() == 'gaussian':
                            with open("{}/gaussian_lot.txt".format(output_folder),"w") as g:
                                g.write("# namespace {}\n".format(name))
                                g.write("# functional {}\n# basis_set {}\n".format(level.split('/')[0],level.split('/')[1]))
                                g.write("# nprocs {}".format(procs))
                            core_command = ' -package Gaussian -lot_inp_file gaussian_lot.txt '

                    if int(procs) == 1:
                        if relax_end:
                            command_line = 'srun -N1 -n1 --exclusive python {} -xyzfile {} -mode DE_GSM {} -CONV_TOL {} -ADD_NODE_TOL {} -num_nodes {} -charge {} -multiplicity {} > log &\n'
                        else:
                            command_line = 'srun -N1 -n1 --exclusive python {} -xyzfile {} -mode DE_GSM {} ' +\
                                           '-reactant_geom_fixed -product_geom_fixed -CONV_TOL {} -ADD_NODE_TOL {} -num_nodes {} -charge {} -multiplicity {} > log &\n'
                        command_line = command_line.format(pygsm_path,product,core_command,conv_tor,add_tor,Nimage,charge,multiplicity)

                    else:
                        if relax_end:
                            command_line = 'python {} -xyzfile {} -mode DE_GSM {} -CONV_TOL {} -ADD_NODE_TOL {} -num_nodes {} -charge {} -multiplicity {} > log &\n\nwait\n'
                        else:
                            command_line = 'python {} -xyzfile {} -mode DE_GSM {} ' +\
                                           '-reactant_geom_fixed -product_geom_fixed -CONV_TOL {} -ADD_NODE_TOL {} -num_nodes {} -charge {} -multiplicity {} > log &\n\nwait\n'
                        command_line = command_line.format(pygsm_path,product,core_command,conv_tor,add_tor,Nimage,charge,multiplicity)
                        
                    f.write(command_line)
                    f.write("cd {}\n".format(working_dir))

                f.write("\nwait\n")
                f.write("echo Total End Time is `date`\n")

    return output_list

# Description: generate submit file from a given list of reactants
#
# input:   R_list : a list of xyzfiles of reactants
#          Njobs  : number of jobs per node
#          Wt     : wall time for each job 
#          nprocs : number of processors used for each job (default: 8)
#          queue  : select a queue to submit jobs (standby, bsavoie, etc.)
#          sched  : scheduler argument for the script (for halstead/brown, slurm is the only option)
#
# CREST parameters:  charge : charge of the system
#                    unpair : number of unpaired electrons in the system
#
# output:  output_list : contains all output folders 
#
def submit_crest(R_list,output_path,Njobs,conf_path,crest_path=None,xtb_path=None,Wt='4',sched='slurm',queue='standby',nprocs=8,charge=0,unpair=0):
    
    Total_jobs= len(R_list)
    N_bundles = int(np.ceil(Total_jobs/float(Njobs)))
    
    # parse Wall time parameter, determine whether this is in minute of hour
    if type(Wt) == str and "min" in Wt: Wt = int(Wt.split('min')[0]); min_flag = 1
    else: Wt = int(Wt); min_flag = 0

    # get working folder direactory 
    working_dir = os.getcwd()

    # initialize output list 
    output_list = []

    # Write CREST.N.submit files
    for n in range(N_bundles):
        
        # Create input files and submit each bundle
        input_reactants = R_list[n * Njobs : (n+1) * Njobs]
        # Submisson file
        if sched == "slurm":    
            with open('CREST.{}.submit'.format(n),'w') as f:                                                                 
                f.write("#!/bin/bash\n")
                f.write("#\n")
                f.write("#SBATCH --job-name=CREST.{}\n".format(n))
                f.write("#SBATCH --output=CREST.{}.out\n".format(n))
                f.write("#SBATCH --error=CREST.{}.err\n".format(n))
                f.write("#SBATCH -A {}\n".format(queue))
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --ntasks-per-node={}\n".format(nprocs))
                if min_flag == 0:
                    f.write("#SBATCH --time {}:00:00\n".format(Wt))
                elif min_flag == 1:
                    f.write("#SBATCH --time 00:{}:00\n".format(Wt))

                # print out information
                f.write("\n# cd into the submission directory\n")
                f.write("cd {}\n".format(working_dir))
                f.write("echo Working directory is ${}\n".format(working_dir))
                f.write("echo Running on host `hostname`\n")
                f.write("echo Time is `date`\n\n")
                f.write("export OMP_STACKSIZE=4G\n")
                f.write("export OMP_NUM_THREADS={}\n".format(nprocs))

                # Create sub-folders
                for reactant in input_reactants:

                    name = reactant.split('/')[-1].split('.xyz')[0]
                    output_folder = conf_path+'/'+name
                    if os.path.isdir(output_folder) is False: os.mkdir(output_folder)
                    output_list += [output_folder]

                    f.write("\ncd {}\n".format(output_folder))
                    f.write("# Running crest jobs for the input file\n")
                    if crest_path is None: crest_path = 'crest'
                    if xtb_path is None: xtb_path = 'xtb'

                    command_line = '{} {} -gfn2 -xnam {} -nozs -chrg {} -uhf {} -quick\n'
                    #command_line = '{} {} -bthr 0.001 -athr 0.001 -gfn2 -xnam {} -nozs -chrg {} -uhf {}\n'
                    command_line = command_line.format(crest_path,reactant,xtb_path,charge,unpair)
                    f.write(command_line)
                    f.write("wait\n")
                    f.write("echo Time is `date`\n\n")
                    f.write("cd {}\n".format(working_dir))
                    f.write('python analyze_crest_output.py {}'.format(output_folder))
                    f.write("\nwait\n")

    return output_list

# Description: generate submit file for generating conformers
#
# input:   conf_pair : a list stores the information of CREST output folder, product xyz file and name for each reaction pathway 
#          Njobs  : number of jobs per node
#          ff     : force field applied to optimize product geometry
#          Nmax   : maximum number of conformers
#          queue  : select a queue to submit jobs (standby, bsavoie, etc.)
#          sched  : scheduler argument for the script (for halstead/brown, slurm is the only option)
#
def submit_select(conf_pair,output_path,Njobs,Nmax=10,ff='uff',sched='slurm',queue='standby',product_opt=False,rank_by_energy=True,remove_constraints=True):
    
    Njobs = int(Njobs)
    Total_jobs= len(conf_pair)
    N_bundles = int(np.ceil(Total_jobs/float(Njobs)))
    
    # get working folder direactory 
    working_dir = os.getcwd()

    # add flags
    flag = ''
    if product_opt: flag += ' --product_opt'
    if rank_by_energy: flag += ' --rank_by_energy'
    if remove_constraints: flag += ' --remove_constraints'

    # Write GSM.N.submit files
    for n in range(N_bundles):

        # Create input files and submit each bundle
        input_products = conf_pair[n*Njobs:(n+1)*Njobs]

        # Submisson file
        if sched == "slurm":    
            with open('CONF_GEN.{}.submit'.format(n),'w') as f:                                                                 
                f.write("#!/bin/bash\n")
                f.write("#\n")
                f.write("#SBATCH --job-name=CONF_GEN.{}\n".format(n))
                f.write("#SBATCH --output=CONF_GEN.{}.out\n".format(n))
                f.write("#SBATCH --error=CONF_GEN.{}.err\n".format(n))
                f.write("#SBATCH -A {}\n".format(queue))
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --ntasks-per-node=1\n")
                f.write("#SBATCH --time 4:00:00\n")

                # print out information
                f.write("\n# cd into the submission directory\n")
                f.write("cd {}\n".format('/'.join(os.path.abspath(__file__).split('/')[:-1])))
                f.write("echo Time is `date`\n")

                # Create sub-folders
                for item in input_products:

                    command_line = '\npython generate_conf.py -i {} -p {} -oF {} -o {} -N {} -ff {} {}'.format(item[0],item[1],output_path,item[2],Nmax,ff,flag)
                    f.write(command_line)
                    f.write("\nwait\n")

    return True

# Description: generate submit file for generating conformers (new version)
#
# input:   conf_pair : a list stores the information of CREST output folder, product xyz file and name for each reaction pathway 
#          Njobs  : number of jobs per node
#          ff     : force field applied to optimize product geometry
#          Nmax   : maximum number of conformers
#          queue  : select a queue to submit jobs (standby, bsavoie, etc.)
#          sched  : scheduler argument for the script (for halstead/brown, slurm is the only option)
#          rank_by_energy: select conformer based on one side energy (won't work for sampling from both reactant and product side)
#
def submit_select_new(conf_pair,output_path,Njobs,Nmax=10,ff='uff',sched='slurm',queue='standby',product_opt=False,rank_by_energy=False,remove_constraints=True):
    
    Njobs = int(Njobs)
    Total_jobs= len(conf_pair)
    N_bundles = int(np.ceil(Total_jobs/float(Njobs)))
    
    # get working folder direactory 
    working_dir = os.getcwd()

    # add flags
    flag = ''
    if product_opt: flag += ' --product_opt'
    if rank_by_energy: flag += ' --rank_by_energy'
    if remove_constraints: flag += ' --remove_constraints'

    # Write GSM.N.submit files
    for n in range(N_bundles):

        # Create input files and submit each bundle
        input_products = conf_pair[n*Njobs:(n+1)*Njobs]

        # Submisson file
        if sched == "slurm":    
            with open('CONF_GEN.{}.submit'.format(n),'w') as f:                                                                 
                f.write("#!/bin/bash\n")
                f.write("#\n")
                f.write("#SBATCH --job-name=CONF_GEN.{}\n".format(n))
                f.write("#SBATCH --output=CONF_GEN.{}.out\n".format(n))
                f.write("#SBATCH --error=CONF_GEN.{}.err\n".format(n))
                f.write("#SBATCH -A {}\n".format(queue))
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --ntasks-per-node=1\n")
                f.write("#SBATCH --time 4:00:00\n")

                # print out information
                f.write("\n# cd into the submission directory\n")
                f.write("cd {}\n".format('/'.join(os.path.abspath(__file__).split('/')[:-1])))
                f.write("echo Time is `date`\n")

                # Create sub-folders
                for item in input_products:

                    command_line = '\npython generate_conf_new.py -i {} -oF {} -o {} -N {} -ff {} {}'.format(item[0],output_path,item[1],Nmax,ff,flag)
                    f.write(command_line)
                    f.write("\nwait\n")

    return True

if __name__ == "__main__":
   main(sys.argv[1:])
