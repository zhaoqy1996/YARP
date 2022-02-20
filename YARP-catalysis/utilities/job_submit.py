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
# output:  output_list : contains 
#
def submit_GSM(pair_list,output_path,Njobs,Wt='4',sched='slurm',queue='standby',charge=0,unpair=0,multiplicity=1,Nimage=9,conv_tor=0.005,add_tor=0.01):
    
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
        input_products = pair_list[n * Njobs : (n+1)*Njobs]

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
                f.write("#SBATCH --ntasks-per-node=1\n")
                if min_flag == 0:
                    f.write("#SBATCH --time {}:00:00\n".format(Wt))
                elif min_flag == 1:
                    f.write("#SBATCH --time 00:{}:00\n".format(Wt))

                # print out information
                f.write("\n# cd into the submission directory\n")
                f.write("cd {}\n".format(working_dir))
                f.write("echo Working directory is ${}\n".format(working_dir))
                f.write("echo Running on host `hostname`\n")
                f.write("echo Time is `date`\n")

                # Create sub-folders
                for product in input_products:
                    name = product.split('/')[-1].split('.xyz')[0]

                    output_folder = output_path+'/'+name
                    if os.path.isdir(output_folder) is False:
                        os.mkdir(output_folder)
                    output_list += [output_folder]

                    with open("{}/xtb_lot.txt".format(output_folder),"w") as g:
                        g.write("# set up parameters\n")
                        g.write("charge \t\t{}\n".format(charge))
                        g.write("spin \t\t{}\n".format(unpair))
                        g.write("namespace \t{}\n".format(name))
                        g.write("calc_type \t{}\n".format("grad"))

                        f.write("\ncd {}\n".format(output_folder))
                        f.write("# Insert nodes and cores in the header of the input file\n")
                        command_line = 'python /home/zhao922/bin/pyGSM/pygsm/wrappers/main.py -xyzfile {} -mode DE_GSM -package xTB -lot_inp_file {} ' +\
                                       '-reactant_geom_fixed -product_geom_fixed -CONV_TOL {} -ADD_NODE_TOL {} -num_nodes {} -charge {} -multiplicity {} > log &\n'
                        #command_line = 'python /home/zhao922/bin/pyGSM/pygsm/wrappers/main.py -xyzfile {} -mode DE_GSM -package xTB -lot_inp_file {} ' +\
                        #               '-CONV_TOL {} -ADD_NODE_TOL {} -num_nodes {} -charge {} -multiplicity {} > log &\n'

                        command_line = command_line.format(product,output_folder+'/xtb_lot.txt',conv_tor,add_tor,Nimage,charge,multiplicity)
                        f.write(command_line)
                        f.write("cd {}\n".format(working_dir))
                        f.write("\nwait\n")

    return output_list
    
if __name__ == "__main__":
   main(sys.argv[1:])
