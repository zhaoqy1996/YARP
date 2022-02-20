#!/bin/env python                                             

# This file is made to generate Orca submit files in cluster 
# Author: Qiyuan Zhao

import sys,argparse,os,ast,re,fnmatch,matplotlib,subprocess,shutil
import numpy as np

def main(argv):

    parser = argparse.ArgumentParser(description='Average over all files matching the -f string.')

    #optional arguments                                                                                                                
    
    parser.add_argument('-f', dest='Filename', default='*.in',
                        help = 'The program operates on all input files discovered during a directory walk from the working directory whose name matches this variable.' + \
                               'For example, if the user supplies *.in, then all files ending in .in within all subfolders relative the working directory will be submitted.') 

    parser.add_argument('-ff', dest='file_in_folder', default=[],
                        help = 'process on given list of gjf files')

    parser.add_argument('-d', dest='path', default=None,
                        help = 'check for output file')

    parser.add_argument('-mem', dest='memory', default=1000,
                        help = 'Specifies the memory use of each cpu (recommend no more than NMEM/NCPU for each cluster)')

    parser.add_argument('-para', dest='parallel', default='horizontal',
                        help = 'How to submit jobs in one node. Horizontal means each job take part of total processors and run simultaneously;' +\
                               'perpendicular means all of them occupy full processors, run one by one')

    parser.add_argument('-p', dest='procs', default=1,
                        help = 'Specifies the number of processors for each job (default: 1; works for horizontal parallel)')

    parser.add_argument('-n', dest='Njobs', default=1,
                        help = 'Specifies bumber of job for each node (default: 1; works for perpendicular)')
                        
    parser.add_argument('-o', dest='outputname', default="Orca_job",
                        help = 'Specifies the job name (default: orca_job.out)')

    parser.add_argument('-t', dest='walltime', default=4,
                        help = 'Specifies the walltime for each job (default: 48, hours by default, if Xmin is used then the argument will be interpretted in minutes)')

    parser.add_argument('-q', dest='queue', default='standby',
                        help = 'Specifies the queue for the job (default: ccm_queue; see NERSC webpage for valid options)')

    parser.add_argument('-ppn', dest='ppn', default=24,
                        help = 'Specifies the number of processors per node on the cluster architecture. the -ppn %% -p should equal zero. (default: 24)') 

    parser.add_argument('-sched', dest='sched', default='slurm',
                        help = 'Specifies the scheduler protocol to use (torque-halstead and torque-titan are implemented)')

    parser.add_argument('-path_to_exe',dest='path_to_exe', default="/depot/bsavoie/apps/orca_5_0_1_openmpi411",help = 'specifies the shell submission script to call.')

    parser.add_argument('-path_to_mpi',dest='path_to_mpi', default="/depot/bsavoie/apps/openmpi_4_1_1",help = 'specifies the openmpi path containing the bin and lib folders.')

    parser.add_argument('--silent',dest='silent', default=False, const=True, action='store_const',
                        help = 'Shortcircuits all script related print statements (scheduler will probably still echo the job id)')

    args=parser.parse_args()
    if type(args.walltime) == str and "min" in args.walltime: args.walltime = int(args.walltime.split('min')[0]); min_flag = 1
    else: args.walltime = int(args.walltime); min_flag = 0
    args.procs = int(args.procs)
    args.Njobs = int(args.Njobs)
    args.memory= int(args.memory)
    args.ppn = int(args.ppn)
    args.size = int(np.floor(float(args.ppn/args.procs)))
    Filename = args.Filename
    working_dir = os.getcwd()

    # Check that the number of processors per job divides into the number of processors per node.
    if args.ppn % args.procs != 0 and args.parallel == 'horizontal':
        print("ERROR: the -ppn % -p must be zero to ensure that jobs aren't split across nodes. Exiting...")
        quit()
    
    # Create a dictionary from the filenames, where each dictionary key corresponds to a filename and each entry is a list
    # of subfiles to be processed as a batch. e.g., molecule.in might show up in twenty subfolders. molecule.in would end
    # up as a key, and the list of individual instances of molecule.in would constitute the entry.
    Files={}
    if len(args.file_in_folder) == 0:
        Files[Filename] = [ os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames if fnmatch.fnmatch(f,Filename) ]
    else:
        FF_str = ((args.file_in_folder).split('[')[1]).split(']')[0]
        Files[Filename] = [(i.replace(' ','')).replace('\'','') for i in FF_str.split(',')]

    #################################################################
    # Iterate over all discovered input files, cd to the containing #
    # directory and check if the job already has an output file     #
    #################################################################    
    input_files = []
    input_paths = []

    for i in Files.keys():

        # Sort the filenames as string
        Current_files = sorted(Files[i])
        if args.silent is False:        
            print("{}".format("#"*80))
            print("# {:^76s} #".format("PROCESSING THE FOLLOWING FILES"))
            print("{}".format("#"*80))

        # Iterate over the discovered input files
        for j in Current_files:
            # Change to the directory holding the current file
            path_to_file = '/'.join(j.split('/')[0:-1])
            if path_to_file != '':
                os.chdir(path_to_file)

            # Save the submission input files and paths to lists
            current_name = j.split('/')[-1]

            if "{}/{}/{}.out".format(args.path,current_name.split('.')[0],current_name.split('.')[0]) not in os.listdir('.'):
                input_files += [current_name]
                input_paths += [path_to_file]
            
            # Skip any that already have output files present
            else:
                if args.silent is False: print("Skipped file {} because output was already found".format(current_name))
            os.chdir(working_dir)

    # If no viable jobs were discovered then quit
    if len(input_files) == 0:
        if args.silent is False:
            print("No jobs in need of running, exiting...")
        quit()

    # Insert escape characters for ( and )
    for i in [ "(", ")" ]:
        input_files = [ _.replace(i,"\{}".format(i)) for _ in input_files ]
        input_paths = [ _.replace(i,"\{}".format(i)) for _ in input_paths ]
        
    if args.parallel not in ['horizontal','perpendicular']:
        print("Error, args.parallel can just be 'horizontal' or 'perpendicular'")

    if args.parallel == 'perpendicular':
    
        # Calculate the number of separate jobs to be submitted
        N_bundles = int(np.ceil(float(len(input_files))/float(args.Njobs)))

        # calculate share memory
        args.procs = args.ppn
        mem = args.memory*args.ppn

        # Bundle the jobs
        bundled_files = [[] for i in range(N_bundles) ]
        bundled_paths = [[] for i in range(N_bundles) ]
        for i in range(N_bundles):
            bundled_files[i] = input_files[i*args.Njobs:(i+1)*args.Njobs]
            bundled_paths[i] = input_paths[i*args.Njobs:(i+1)*args.Njobs]

        # Create input files and submit each bundle
        for n in range(len(bundled_files)):
            
            # Set input_files and input_paths to point towards the bundled_files and bundled_paths sublists
            # NOTE: the reuse of variable names (input_files, input_paths) is just a convenience since the following loops weren't written for the bundled feature.
            input_files = bundled_files[n]
            input_paths = bundled_paths[n]

            # Initialize working variable for the number of sub jobs being submitted, total cores, total nodes, and jobs per node
            N_jobs = len(input_files)
            N_cores = N_jobs*args.procs

            # Submisson file
            if args.sched == "torque-halstead":    
                with open('{}.{}.submit'.format(args.outputname,n),'w') as f:
                    f.write("#PBS -N {}.{}\n".format(args.outputname,n))
                    f.write("#PBS -l nodes={}:ppn={}\n".format(1,int(args.ppn)))
                    if min_flag == 0:
                        f.write("#PBS -l walltime={}:00:00\n".format(args.walltime))
                    elif min_flag == 1:
                        f.write("#PBS -l walltime=00:{}:00\n".format(args.walltime))
                    f.write("#PBS -q {}\n".format(args.queue))
                    f.write("#PBS -S /bin/sh\n")
                    f.write("#PBS -o {}.{}.out\n".format(args.outputname,n))
                    f.write("#PBS -e {}.{}.err\n\n".format(args.outputname,n))

                    # print out information
                    f.write("\n# cd into the submission directory\n")
                    f.write("cd {}\n".format(working_dir))
                    f.write("echo Working directory is ${}\n".format(working_dir))
                    f.write("echo Running on host `hostname`\n")
                    f.write("echo Start Time is `date`\n\n")

                    # Load Orca
                    f.write("# Load environment for Orca\n")
                    f.write("module unload openmpi \n")
                    f.write("export PATH={}:$PATH\n".format(args.path_to_exe))
                    f.write("export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH\n".format(args.path_to_exe))
                    f.write("export PATH={}/bin:$PATH\n".format(args.path_to_mpi))
                    f.write("export LD_LIBRARY_PATH={}/lib:$LD_LIBRARY_PATH\n".format(args.path_to_mpi))
                    f.write("\n# Define number of cores and number of nodes\n")
                    f.write("export NPROCS=`wc -l < $PBS_NODEFILE`\n")
                    f.write("export NODES=`uniq $PBS_NODEFILE | awk '{printf(\"%s,\", $0)}' | sed 's/.$//'`\n")

                    for lj,j in enumerate(input_paths):
                        f.write("\ncd {}\n".format(j))
                        name = '.'.join(input_files[lj].split('.')[:-1])
                        f.write("{}/orca {} > {}.out &\n".format(args.path_to_exe,input_files[lj],name))
                        f.write("cd {}\n".format(working_dir))
                        f.write("\nwait\n")
                    f.write("echo End Time is `date`\n\n")

            if args.sched == "slurm":    
                with open('{}.{}.submit'.format(args.outputname,n),'w') as f:                                                                 
                    f.write("#!/bin/bash\n")
                    f.write("#\n")
                    f.write("#SBATCH --job-name={}.{}\n".format(args.outputname,n))
                    f.write("#SBATCH --output={}.{}.out\n".format(args.outputname,n))
                    f.write("#SBATCH --error={}.{}.err\n".format(args.outputname,n))
                    f.write("#SBATCH -A {}\n".format(args.queue))
                    f.write("#SBATCH --nodes={}\n".format(1))
                    f.write("#SBATCH --ntasks-per-node={}\n".format(args.ppn))
                    f.write("#SBATCH --mem-per-cpu={}MB\n".format(args.memory))
                    if min_flag == 0:
                        f.write("#SBATCH --time {}:00:00\n".format(args.walltime))
                    elif min_flag == 1:
                        f.write("#SBATCH --time 00:{}:00\n".format(args.walltime))

                    f.write("\n# cd into the submission directory\n")
                    f.write("cd {}\n".format(working_dir))
                    f.write("echo Working directory is ${}\n".format(working_dir))
                    f.write("echo Running on host `hostname`\n")
                    f.write("echo Start Time is `date`\n\n")

                    # Load Orca
                    f.write("# Load environment for Orca\n")
                    f.write("module unload openmpi \n")
                    f.write("export PATH={}:$PATH\n".format(args.path_to_exe))
                    f.write("export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH\n".format(args.path_to_exe))
                    f.write("export PATH={}/bin:$PATH\n".format(args.path_to_mpi))
                    f.write("export LD_LIBRARY_PATH={}/lib:$LD_LIBRARY_PATH\n".format(args.path_to_mpi))

                    # print out information
                    f.write("\n# cd into the submission directory\n")
                    for lj,j in enumerate(input_paths):
                        f.write("\ncd {}\n".format(j))
                        name = '.'.join(input_files[lj].split('.')[:-1])
                        f.write("{}/orca {} > {}.out &\n".format(args.path_to_exe,input_files[lj],name))
                        f.write("cd {}\n".format(working_dir))
                        f.write("\nwait\n")

                    f.write("echo End Time is `date`\n\n")

    else:
        # Calculate the number of separate jobs to be submitted
        N_bundles = int(np.ceil(float(len(input_files))/float(args.Njobs)))

        # calculate share memory
        mem = int(args.memory * args.procs)

        # Bundle the jobs
        bundled_files = [[] for i in range(N_bundles) ]
        bundled_paths = [[] for i in range(N_bundles) ]

        for i in range(N_bundles):
            bundled_files[i] = input_files[i*args.Njobs:(i+1)*args.Njobs]
            bundled_paths[i] = input_paths[i*args.Njobs:(i+1)*args.Njobs]

        # Create input files and submit each bundle
        for n in range(len(bundled_files)):
            
            # Set input_files and input_paths to point towards the bundled_files and bundled_paths sublists
            # NOTE: the reuse of variable names (input_files, input_paths) is just a convenience since the following loops weren't written for the bundled feature.
            input_files = bundled_files[n]
            input_paths = bundled_paths[n]

            # Initialize working variable for the number of sub jobs being submitted, total cores, total nodes, and jobs per node
            N_jobs = len(input_files)
            N_cores = N_jobs*args.procs

            # Submisson file
            if args.sched == "torque-halstead":    
                with open('{}.{}.submit'.format(args.outputname,n),'w') as f:
                    f.write("#PBS -N {}.{}\n".format(args.outputname,n))
                    f.write("#PBS -l nodes={}:ppn={}\n".format(1,int(args.ppn)))
                    if min_flag == 0:
                        f.write("#PBS -l walltime={}:00:00\n".format(args.walltime))
                    elif min_flag == 1:
                        f.write("#PBS -l walltime=00:{}:00\n".format(args.walltime))
                    f.write("#PBS -q {}\n".format(args.queue))
                    f.write("#PBS -S /bin/sh\n")
                    f.write("#PBS -o {}.{}.out\n".format(args.outputname,n))
                    f.write("#PBS -e {}.{}.err\n\n".format(args.outputname,n))

                    # print out information
                    f.write("\n# cd into the submission directory\n")
                    f.write("cd {}\n".format(working_dir))
                    f.write("echo Working directory is ${}\n".format(working_dir))
                    f.write("echo Running on host `hostname`\n")
                    f.write("echo Start Time is `date`\n\n")

                    # Load Orca
                    f.write("# Load environment for Orca\n")
                    f.write("module unload openmpi \n")
                    f.write("export PATH={}:$PATH\n".format(args.path_to_exe))
                    f.write("export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH\n".format(args.path_to_exe))
                    f.write("export PATH={}/bin:$PATH\n".format(args.path_to_mpi))
                    f.write("export LD_LIBRARY_PATH={}/lib:$LD_LIBRARY_PATH\n".format(args.path_to_mpi))
                    f.write("\n# Define number of cores and number of nodes\n")
                    f.write("export NPROCS=`wc -l < $PBS_NODEFILE`\n")
                    f.write("export NODES=`uniq $PBS_NODEFILE | awk '{printf(\"%s,\", $0)}' | sed 's/.$//'`\n")

                    # print out information
                    f.write("\n# cd into the submission directory\n")
                    for lj,j in enumerate(input_paths):
                        f.write("\ncd {}\n".format(j))
                        name = '.'.join(input_files[lj].split('.')[:-1])
                        f.write("{}/orca {} > {}.out &\n".format(args.path_to_exe,input_files[lj],name))
                        f.write("cd {}\n".format(working_dir))

                    f.write("\nwait\n")
                    f.write("echo End Time is `date`\n\n")

            if args.sched == "slurm":    
                with open('{}.{}.submit'.format(args.outputname,n),'w') as f:                                                                 
                    f.write("#!/bin/bash\n")
                    f.write("#\n")
                    f.write("#SBATCH --job-name={}.{}\n".format(args.outputname,n))
                    f.write("#SBATCH --output={}.{}.out\n".format(args.outputname,n))
                    f.write("#SBATCH --error={}.{}.err\n".format(args.outputname,n))
                    f.write("#SBATCH -A {}\n".format(args.queue))
                    f.write("#SBATCH --nodes={}\n".format(1))
                    f.write("#SBATCH --ntasks-per-node={}\n".format(args.ppn))
                    f.write("#SBATCH --mem-per-cpu={}MB\n".format(args.memory))
                    if min_flag == 0:
                        f.write("#SBATCH --time {}:00:00\n".format(args.walltime))
                    elif min_flag == 1:
                        f.write("#SBATCH --time 00:{}:00\n".format(args.walltime))

                    # print out information
                    f.write("\n# cd into the submission directory\n")
                    f.write("cd {}\n".format(working_dir))
                    f.write("echo Working directory is ${}\n".format(working_dir))
                    f.write("echo Running on host `hostname`\n")
                    f.write("echo Start Time is `date`\n\n")
                    
                    # Load Orca
                    f.write("# Load environment for Orca\n")
                    f.write("module unload openmpi \n")
                    f.write("export PATH={}:$PATH\n".format(args.path_to_exe))
                    f.write("export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH\n".format(args.path_to_exe))
                    f.write("export PATH={}/bin:$PATH\n".format(args.path_to_mpi))
                    f.write("export LD_LIBRARY_PATH={}/lib:$LD_LIBRARY_PATH\n".format(args.path_to_mpi))

                    # print out information
                    f.write("\n# cd into the submission directory\n")
                    for lj,j in enumerate(input_paths):
                        f.write("\ncd {}\n".format(j))
                        name = '.'.join(input_files[lj].split('.')[:-1])
                        f.write("{}/orca {} > {}.out &\n".format(args.path_to_exe,input_files[lj],name))
                        f.write("cd {}\n".format(working_dir))

                    f.write("\nwait\n")
                    f.write("echo End Time is `date`\n\n")
                        
if __name__ == "__main__":
   main(sys.argv[1:])
