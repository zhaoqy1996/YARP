import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
import pickle,json
import numpy as np

# Load modules in same folder        
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/utilities')
from xtb_functions import xtb_energy,xtb_geo_opt
from taffi_functions import *
from job_submit import *
from utility import *

def main(argv):

    parser = argparse.ArgumentParser(description='Driver script for submitting GSM-xTB jobs. This program can take one reactant xyz file and \
                                     a product xyz file folder and apply GSM to search for reaction pathways.')

    #optional arguments                                             
    parser.add_argument('-c', dest='config', default='refine_config.txt',
                        help = 'The program expects a configuration file from which to assign various run conditions. (default: config.txt in the current working directory)')

    parser.add_argument('-o', dest='outputname', default='result',
                        help = 'Controls the output folder name for the result')

    # parse configuration dictionary (c)
    print("parsing calculation configuration...")
    args=parser.parse_args()
    c = parse_configuration(parser.parse_args())
    sys.stdout = Logger(args.outputname)
    run_GSM(c)

    return

def run_GSM(c):

    # create folders
    if os.path.isdir(c["output_path"]) is False:
        print("Expected a working folder with lower level calculation results, quit...")
        quit()

    # parse energy dictionary
    E_dict=parse_Energy(c["e_dict"])

    # load in reaction dictionary
    try:
        with open(c["reaction_dict"],'rb') as f:
            reaction_dict = pickle.load(f)
    except:
        reaction_dict = {}

    # before going into YARP calculation, first check whether all DFT energies exist for all reactants
    # find smiles strings for each seperated reactants
    reactant_smiles = []

    # parse input xyz 
    input_xyzs = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(c['input_xyz']) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])
    reactant_list= []

    # iterate over each xyz file to identify 
    for xyz in sorted(input_xyzs):

        # obtain input file name (please avoid random naming style)
        react = '_'.join(xyz.split('/')[-1].split('.xyz')[0].split('_')[:-1])

        # add smiles into reactant_smiles
        if react not in reactant_list: 
            reactant_list += [react]
            RE,RG,_ = parse_input(xyz)
            smi = return_smi(RE,RG)
            reactant_smiles += smi.split('.')

    # check whether DFT energy is given
    reactant_smiles = list(set(reactant_smiles))
    DFT_finish,tmp = check_DFT_energy(reactant_smiles,E_dict,conf_path=c['c_path'],functional=c["functional"],basis=c["basis"],dispersion=c["dispersion"],config=c)

    if DFT_finish:
        E_dict = tmp
    else:
        print("Error appears when checking DFT level energies, check CREST folder and energy dict...")
        quit()

    # find all xyz files in final input folder ( if input_types==2, directly jumps to this step)
    N_dict,F_dict,adj_dict,hash_dict = parse_pyGSM_inputs(input_xyzs,E_dict)
    qt,unpair = int(c["charge"]),int(c["unpair"])

    ############################################################
    # Analyze the output files and get the initial guess of TS #
    ############################################################
    # make a initial TS guess folder
    TS_folder = c["output_path"] + '/Higher_level_TS'
    if os.path.isdir(TS_folder) is False:
        os.mkdir(TS_folder)

    # generate gjf (Gaussian input file) list
    TSgjf_list = []

    # loop over all of result folders
    for xyz in input_xyzs:

        pname = xyz.split('/')[-1].split('.xyz')[0]

        # use xyz_to_Gaussian to tranfer a xyz file to gjf
        if c["dispersion"] is None:
            substring = "python {}/utilities/xyz_to_Gaussian.py {}/TS-folder/{}-TS.xyz -o {}/{}-TS.gjf -q {} -m {} -c False" + \
                        " -ty \"{}/{} OPT=(TS, CALCFC, NOEIGEN, maxcycles=100) Freq \" -t \"{} TS\" "
            substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),c["output_path"],pname,TS_folder,pname,qt,unpair+1,c["functional"],c["basis"],pname)
            os.system(substring)
        else:
            substring = "python {}/utilities/xyz_to_Gaussian.py {}/TS-folder/{}-TS.xyz -o {}/{}-TS.gjf -q {} -m {} -c False" + \
                        " -ty \"{}/{} EmpiricalDispersion=G{} OPT=(TS, CALCFC, NOEIGEN, maxcycles=100) Freq \" -t \"{} TS\" "
            substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),c["output_path"],pname,TS_folder,pname,qt,unpair+1,c["functional"],c["basis"],c["dispersion"],pname)
            os.system(substring)
                
        # add gjf file to the list
        TSgjf_list += [TS_folder+'/{}-TS.gjf'.format(pname)]

    # generate submit files
    substring="python {}/utilities/Gaussian_submit.py -f '*.gjf' -ff \"{}\" -d {} -para {} -p {} -n {} -ppn {} -q {} -mem {} -sched {} -t {} -o TS --silent"
    substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),TSgjf_list,TS_folder,c["parallel"],c["procs"],c["njobs"],c["ppn"],c["queue"],c["memory"],c["sched"],c["wt"]) 
    os.system(substring)
    #'''
    # submit all the jobs
    substring="python {}/utilities/job_submit.py -f 'TS.*.submit' -sched {}".format('/'.join(os.getcwd().split('/')[:-1]),c["sched"])
    output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        
    if c["batch"] == 'pbs':
        print("\t running {} Gaussian TS_opt jobs...".format(int(len(output.split()))))
    elif c["batch"] == 'slurm':
        print("\t running {} Gaussian TS_opt jobs...".format(int(len(output.split())/4)))
        
    monitor_jobs(output.split())
    os.system('rm *.submit')
    #'''
    ########################################
    # Analyze the TS and classify channels #
    ########################################
    # find here generate IRC_dict (all intended)
    IRC_dict={pname:1 for pname in N_dict.keys()}
    N_grad,reaction_dict,TSgjf_list = channel_classification(output_path=c["output_path"],initial_xyz=c['input_xyz'],gjf_list=TSgjf_list,xTB_IRC_dict=IRC_dict,N_dict=N_dict,adj_dict=adj_dict,F_dict=F_dict,reaction_dict=reaction_dict,\
                                                             functional=c["functional"],basis=c["basis"],dispersion=c["dispersion"],TS_folder=TS_folder,append=True,\
                                                             model='{}//utilities/IRC_model.json'.format('/'.join(os.getcwd().split('/')[:-1])))

    print("Total number of gradient calls is {}".format(N_grad))

    ###############################################
    # Analyze the TS and perform IRC at DFT level #
    ###############################################
    if c["dft-irc"]:
        # make a IRC folder
        IRC_folder = c["output_path"] + '/Higher_level_IRC'
        if os.path.isdir(IRC_folder) is False:
            os.mkdir(IRC_folder)

        # Initialize gjf files for IRC 
        IRCgjf_list = []

        # TS gibbs free energy dictionary
        TSEne_dict={}

        # check job status, if TS successfully found, put into IRC list
        for TSgjf in TSgjf_list:

            # change .gjf to .out will return TS geo-opt output file
            TSout = TSgjf.replace('.gjf','.out')

            # return file name
            pname = TSgjf.split('/')[-1].split('-TS')[0]

            # imag_flag refers whether there is an imaginary frequency in TS output file; finish_flag refers to whether TS geo-opt normally finished
            finish_flag,imag_flag,SPE,zero_E,H_298,F_298 = read_Gaussian_output(TSout)

            # for the success tasks, generate optimized TS geometry
            if imag_flag and finish_flag:

                TSEne_dict[pname]=F_298
                print("TS for reaction payway to {} is found with Energy barrier {}".format(pname,(F_298-F_dict[pname])*627.5))

                # generate IRC Gaussian input files
                substring = "python {}/utilities/xyz_to_Gaussian.py {}/{}-TS.xyz -o {}/{}-IRC.gjf -q {} -m {} -c False" + \
                            " -ty \"{}/{} IRC=(LQA, recorrect=never, CalcFC, maxpoints={}, StepSize={}, maxcyc=100, Report=Cartesians)\" -t \"{} IRC\" "
                substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),TS_folder,pname,IRC_folder,pname,qt,unpair+1,c["functional"],c["basis"],c["irc-image"],c["stepsize"],pname)
                os.system(substring)

                # add gjf file into the list
                IRCgjf_list += [IRC_folder+'/{}-IRC.gjf'.format(pname)]

            else:
                print("TS for reaction payway to {} fails (either no imag freq or geo-opt fails)".format(pname))
                continue

        # Generate IRC calculation job and wait for the result
        substring="python {}/utilities/Gaussian_submit.py -f '*.gjf' -ff \"{}\" -d {} -para {} -p {} -n {} -ppn {} -q {} -mem {} -sched {} -t {} -o IRC --silent"
        substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),IRCgjf_list,IRC_folder,c["parallel"],c["procs"],c["njobs"],c["ppn"],c["queue"],c["memory"],c["sched"],c["wt"]) 
        os.system(substring)
            
        # submit all the jobs
        substring="python {}/utilities/job_submit.py -f 'IRC.*.submit' -sched {}".format('/'.join(os.getcwd().split('/')[:-1]),c["sched"])
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        
        if c["batch"] == 'pbs':
            print("\t running {} Gaussian IRC jobs...".format(int(len(output.split()))))
        elif c["batch"] == 'slurm':
            print("\t running {} Gaussian IRC jobs...".format(int(len(output.split())/4)))

        monitor_jobs(output.split())
        os.system('rm *.submit')
        reaction_dict = read_IRC_outputs(c["output_path"],IRCgjf_list,adj_dict,hash_dict,N_dict,F_dict,TSEne_dict,reaction_dict,functional=c["functional"],basis=c["basis"],dispersion=c["dispersion"],append=True)

    ##########################################################
    # Analyze the IRC result and return product and reactant #
    ##########################################################
    # write updated reactant and rection dictionries into db file
    with open(c["reaction_dict"],'wb') as fp:
        pickle.dump(reaction_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return

# Function for keeping tabs on the validity of the user supplied inputs
def parse_configuration(args):
    
    # Convert inputs to the proper data type
    if os.path.isfile(args.config) is False:
        print("ERROR in python_driver: the configuration file {} does not exist.".format(args.config))
        quit()
    
    # Process configuration file for keywords
    keywords = ["reaction_dict","input_xyz","output_path","e_dict","c_path","dft-irc","batch","sched","wt","njobs","ppn","procs","queue","functional","basis",\
                "dispersion","parallel","memory","irc-image","stepsize","charge","unpair"]

    keywords = [ _.lower() for _ in keywords ]
    
    list_delimiters = [","]  # values containing any delimiters in this list will be split into lists based on the delimiter
    space_delimiters = ["&"] # values containing any delimiters in this list will be turned into strings with spaces replacing delimiters
    configs = { i:None for i in keywords }    

    with open(args.config,'r') as f:
        for lines in f:
            fields = lines.split()
            
            # Delete comments
            if "#" in fields:
                del fields[fields.index("#"):]

            # Parse keywords
            l_fields = [ _.lower() for _ in fields ] 
            for i in keywords:
                if i in l_fields:
                    # Parse keyword value pair
                    ind = l_fields.index(i) + 1
                    if len(fields) >= ind + 1:
                        configs[i] = fields[ind]

                        # Handle delimiter parsing of lists
                        for j in space_delimiters:
                            if j in configs[i]:
                                configs[i] = " ".join([ _ for _ in configs[i].split(j) ])
                        for j in list_delimiters:
                            if j in configs[i]:
                                configs[i] = configs[i].split(j)
                                break

                    # Break if keyword is encountered in a non-comment token without an argument
                    else:
                        print("ERROR in python_driver: enountered a keyword ({}) without an argument.".format(i))
                        quit()
                        
    # Check that batch is an acceptable system
    if configs["batch"] not in ["pbs","slurm"]:
        print("ERROR in locate_TS: only pbs and slurm are acceptable arguments for the batch variable.")
        quit()
    elif configs["batch"] == "pbs":
        configs["sub_cmd"] = "qsub"
    elif configs["batch"] == "slurm":
        configs["sub_cmd"] = "sbatch"

    # Check that dispersion option is valid
    if configs["dispersion"].lower() == 'none': 
        configs["dispersion"] = None
    elif configs["dispersion"] not in ["D2", "D3", "D3BJ"]:
        print("Gaussian only has D2, D3 and D3BJ Empirical Dispersion, check inputs...")
        quit()

    # set default value for IRC at DFT level
    if configs["dft-irc"] is None: configs["dft-irc"] = False
    elif configs["dft-irc"].lower() == 'false': configs["dft-irc"] = False
    else: configs["dft-irc"] = True
    
    return configs

class Logger(object):
    def __init__(self,folder):
        self.terminal = sys.stdout
        self.log = open(folder+"/result.log", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass

if __name__ == "__main__":
    main(sys.argv[1:])

