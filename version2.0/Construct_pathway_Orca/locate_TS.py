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

    parser = argparse.ArgumentParser(description='Driver script for submitting YARP jobs with Orca engine.')

    #optional arguments                                             
    parser.add_argument('-c', dest='config', default='config.txt',
                        help = 'The program expects a configuration file from which to assign various run conditions. (default: config.txt in the current working directory)')

    parser.add_argument('-o', dest='outputname', default='result',
                        help = 'Controls the output folder name for the result')

    # parse configuration dictionary (c)
    print("parsing calculation configuration...")
    args=parser.parse_args()
    c = parse_configuration(parser.parse_args())
    sys.stdout = Logger(args.outputname)
    run_YARP_Orca(c)

    return

def run_YARP_Orca(c):

    # create folders
    if os.path.isdir(c["output_path"]) is False: os.mkdir(c["output_path"])

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
    product_smiles  = []

    # if input_type is 0, exploring reactions from scratch. Should have all input reactants in a reactant dictionary
    if c['input_type'] == 0:
        
        # load in reactant dictionary
        with open(c["reactant_dict"],'rb') as f:
            reactant_dict = pickle.load(f)

        # Analyze the reactant
        input_list = []
        with open(c["input_react"],"r") as f:
            for lines in f:
                if lines.split()[0] == '#': continue
                input_list.append(lines.split()[0])

        for inchi_ind in input_list:

            if inchi_ind not in reactant_dict.keys() or "possible_products" not in reactant_dict[inchi_ind].keys():
                print("{} missing possible products, first run reaction enumeration for it...".format(inchi_ind))
                input_list.remove(inchi_ind)

            else:
                prop  = reactant_dict[inchi_ind]["prop"]
                reactant_smiles += prop["smiles"].split('.')

    # if input_type is 1/2, parse reactant information from input files
    else:
        input_xyzs = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(c['input_xyz']) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])
        reactant_list= []

        # iterate over each xyz file to identify 
        for xyz in sorted(input_xyzs):

            # obtain input file name (please avoid random naming style)
            react = '_'.join(xyz.split('/')[-1].split('.xyz')[0].split('_')[:-1])

            # add smiles into reactant_smiles
            if react not in reactant_list: 
                reactant_list += [react]
                RE,RG,PG = parse_input(xyz)
                smi = return_smi(RE,RG,Table_generator(RE, RG))
                reactant_smiles += smi.split('.')
                if c['compute_product']:
                    Psmi = return_smi(RE,PG,Table_generator(RE, PG))
                    product_smiles += Psmi.split('.')

    # determine output folder for CREST jobs
    if c['c_path'] is None: conf_path = c['output_path']+'/conformer'
    else: conf_path =  c['c_path']

    # check whether DFT energy is given
    if c['compute_product']: reactant_smiles = list(set(reactant_smiles+product_smiles))
    else: reactant_smiles = list(set(reactant_smiles))
    energy_config = {"parallel": c["parallel"], "procs": c["high_procs"], "njobs": c["high_njobs"], "ppn": c["ppn"], "queue": c["high_queue"], "memory": c["memory"], "sched": c["sched"],\
                     "wt": c["high_wt"], "batch": c["batch"], "e_dict": c["e_dict"]}
    DFT_finish,tmp = check_DFT_energy(reactant_smiles,E_dict,conf_path,functional=c["functional"],basis=c["basis"],dispersion=c["dispersion"],config=energy_config,package='Orca')

    if DFT_finish:
        E_dict = tmp
        NR_list = []
    else:
        NR_list = tmp

    ##########################################################
    # perfrom CREST on each reactant if input_type is 0 or 1 #
    ##########################################################
    R_list = []
    if c['input_type']==0 or c['input_type']==1 or len(NR_list) > 0:

        # Make conformer folder for CREST working folder
        if os.path.isdir(c['output_path']+'/xTB-folder') is False: os.mkdir(c['output_path']+'/xTB-folder')
        if os.path.isdir(c['output_path']+'/opt-folder') is False: os.mkdir(c['output_path']+'/opt-folder')
        if os.path.isdir(conf_path) is False: os.mkdir(conf_path)

    if c['input_type']==0:

        # initialize reactant list for CREST
        for inchi_ind in input_list:

            # load in properties
            prop  = reactant_dict[inchi_ind]["prop"]
            E,G,qt,unpair=prop['E'],prop['G'],prop['charge'],prop['unpair'] 

            # generate input files
            reactant_file  = c["output_path"]+'/opt-folder/'+'{}.xyz'.format(inchi_ind)
            
            # perform xtb opt on reactant, check whether geo-opt changes the reactant. If so, geo-opt fails
            oinchi = return_inchikey(E,G,separate=True)
            xyz_write(reactant_file, E, G)
            Energy, opted_geo, finish=xtb_geo_opt(reactant_file,charge=qt,unpair=unpair,namespace=oinchi,workdir=(c['output_path']+'/xTB-folder'),level='normal',output_xyz=reactant_file,cleanup=False)
            if not finish: continue
            NE,NG  = xyz_parse(reactant_file)
            ninchi = return_inchikey(NE,NG,separate=True)
            if oinchi[:14] != ninchi[:14]: continue

            # add reactant file into a list
            R_list += [reactant_file]

    elif c['input_type']==1:

        # set charge and unpair
        qt,unpair = int(c['charge']),int(c['unpair'])

        # initialize reactant list for CREST
        R_inchi= []
        R_list_refer = {}

        # iterate over input files to parse unique smiles
        input_files=[i for i in os.listdir(c['input_xyz']) if fnmatch.fnmatch(i, '*.xyz')]
        for count_i, i in enumerate(input_files):
            E,RG,PG = parse_input('{}/{}'.format(c['input_xyz'], i))
            oinchi  = return_inchikey(E,RG,separate=True)
            R_list_refer[i] = oinchi

            if oinchi not in R_inchi: 

                R_inchi += [oinchi]
                reactant_file=c['output_path']+'/opt-folder/'+'{}.xyz'.format(oinchi)
                xyz_write(reactant_file, E, RG)
                Energy, opted_geo, finish=xtb_geo_opt(reactant_file,charge=qt,unpair=unpair,namespace=oinchi,workdir=(c['output_path']+'/xTB-folder'),level='normal',output_xyz=reactant_file,cleanup=False)
                if not finish: continue
                NRE,NRG = xyz_parse(reactant_file)
                ninchi  = return_inchikey(NRE,NRG,separate=True)

                # Check whether geo-opt changes the geometry. If so, geo-opt fails
                if oinchi[:14] != ninchi[:14]: continue

                # add reactant file into a list
                R_list += [reactant_file]

    else:
        qt,unpair = int(c['charge']),int(c['unpair'])

    # merge two R_list
    current_compounds = [filei.split('/')[-1] for filei in R_list]
    for filei in NR_list:
        if filei.split('/')[-1] not in current_compounds:
            R_list += [filei]

    ###################################################
    ####   Submit CREST jobs if input type is 0/1  ####
    ###################################################
    if c['input_type']==0 or c['input_type']==1 or len(R_list) > 0:

        # check if conformer searching are already done
        finished = os.listdir(conf_path)
        R_list = [i for i in R_list if i.split('/')[-1].split('.xyz')[0] not in finished]

        # submit crest jobs
        output_list = submit_crest(R_list,c["output_path"],int(c["c_njobs"]),conf_path=conf_path,Wt=c["c_wt"],sched=c["sched"],queue=c["c_queue"],nprocs=c["c_nprocs"],charge=qt,unpair=unpair)
        substring = "python {}/utilities/job_submit.py -f 'CREST.*.submit' -sched {}".format('/'.join(os.getcwd().split('/')[:-1]),c["sched"])
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')

        if c["batch"] == 'pbs': print("\t running {} CREST jobs...".format(int(len(output.split()))))
        elif c["batch"] == 'slurm': print("\t running {} CREST jobs...".format(int(len(output.split())/4)))
    
        monitor_jobs(output.split())

    #################################################################
    #    Re-generate DFT dictionary is previous check_DFT fail      #
    #################################################################
    if DFT_finish is False:
        DFT_finish,tmp = check_DFT_energy(reactant_smiles,E_dict,conf_path,functional=c["functional"],basis=c["basis"],dispersion=c["dispersion"],config=energy_config,package='Orca')
        if DFT_finish:
            E_dict = tmp
        else:
            print("Error appears when checking DFT level energies, check it...")
            quit()

    #################################################################
    # Analyze the CREST output and prepare a input folder with conf #
    #################################################################
    # create folders 
    if os.path.isdir(c["output_path"]+'/input_files_conf') is False: os.mkdir(c["output_path"]+'/input_files_conf')
    if os.path.isdir(c["output_path"]+'/products_init') is False: os.mkdir(c["output_path"]+'/products_init')

    # input_type is 0, generate input files for GSM based on reactants and crest results
    if c['input_type']==0:

        # conf_pair stores the information of CREST output folder, product xyz file and name for each reaction pathway
        conf_pair = []
        for inchi_ind in input_list:
            
            # load in properties
            prop  = reactant_dict[inchi_ind]["prop"]
            E,G,qt,unpair,adj_mat,hash_list,inchi_list=prop['E'],prop['G'],prop['charge'],prop['unpair'],prop["adj_mat"],prop["hash_list"],prop["inchi_list"] 
            N_element= len(E)

            # if apply tcit is true, parse in TCIT result for reactant and product
            if c["apply_tcit"]:

                # load in TCIT result
                if os.path.isfile(c["tcit_result"]) is True:
                    with open(c["tcit_result"],"r") as g: 
                        TCIT_dict= json.load(g)      

                if "TCIT_Hf" not in reactant_dict[inchi_ind]["prop"].keys():
                    smile_list = prop["smiles"].split('.')
                    R_Hf_298 = sum([TCIT_dict[smi]["Hf_298"] for smi in smile_list])
                    reactant_dict[inchi_ind]["prop"]["TCIT_Hf"] = R_Hf_298

                # calculate enthalpy change
                for pind,pp in reactant_dict[inchi_ind]["possible_products"].items():
                    if "TCIT_Hf" not in pp.keys():
                        smile_list = pp["name"].split('.')
                        try:
                            P_Hf_298   = sum([TCIT_dict[smi]["Hf_298"] for smi in smile_list])
                            Reaction_Hf= P_Hf_298 - R_Hf_298 
                            reactant_dict[inchi_ind]["possible_products"][pind]["TCIT_Hf"]    = P_Hf_298
                            reactant_dict[inchi_ind]["possible_products"][pind]["Reaction_Hf"]= Reaction_Hf
                        except:
                            print("TCIT result for {}_{} is missing, add this into potential products...".format(inchi_ind,pind))
                            reactant_dict[inchi_ind]["possible_products"][pind]["Reaction_Hf"]=0
                    else:
                        reactant_dict[inchi_ind]["possible_products"][pind]["Reaction_Hf"]= pp["TCIT_Hf"] - reactant_dict[inchi_ind]["prop"]["TCIT_Hf"]

            # reactant xyz file
            conf_folder = '{}/{}/results'.format(conf_path,inchi_ind)
            
            # loop over all products
            for pind,pp in reactant_dict[inchi_ind]["possible_products"].items():

                # if apply tcit is true, only keep those satisfies Delta Hf criteria 
                if c["apply_tcit"]:
                    if pp["Reaction_Hf"] > float(c["hf_cut"]):
                        print("The enthalpy of reaction for product {} (smiles: {}) is {}, which is larger than Delta Hf criteria, exclude this compound".format(pind,pp['name'],pp["Reaction_Hf"]))
                        continue
                    else:
                        print("The enthalpy of reaction for product {} (smiles: {}) is {}, which is smaller than Delta Hf criteria, keep this compound".format(pind,pp['name'],pp["Reaction_Hf"]))

                # check whether this reaction pathway is already in reaction database, if so, skip this compounds
                Rsmiles_list=prop["smiles"].split('.')
                Psmiles_list=pp["name"].split('.')
                
                # exclude same smiles string, they serve as spectater
                Rsmiles='.'.join(sorted([smi for smi in Rsmiles_list if smi not in Psmiles_list]))
                Psmiles='.'.join(sorted([smi for smi in Psmiles_list if smi not in Rsmiles_list]))
                reaction_index,_,_ = return_Rindex(prop['E'],prop['G'],pp['G_list'][0],prop['adj_mat'],pp['adj_list'][0])

                if reaction_index in reaction_dict.keys():
                    '''
                    if reaction_dict[reaction_index]['status'] == 'Finish':
                        print("Reaction pathway between {} and {} is already constructed".format(reaction_index[0],reaction_index[1]))
                    else:
                        print("Reaction pathway between {} and {} fails to be constructed at the given level of theory".format(reaction_index[0],reaction_index[1]))
                    '''
                    F_R = sum([E_dict[inchi[:14]]['F'] for inchi in prop["inchi_list"]])
                    DG = [(F_TS-F_R)*627.5 for F_TS in reaction_dict[reaction_index]['TS_Energy']]
                    print("Reaction pathway between {} and {} is already constructed, with free energy of activation of {} kcal/mol".format(Rsmiles,Psmiles,min(DG)))
                    continue

                for ind in range(len(pp['G_list'])):
                            
                    # load in product prop
                    PG = pp['G_list'][ind]
                    Padj_mat = pp['adj_list'][ind]

                    # apply joint-opt if is needed
                    if '-' in inchi_ind:
                        inchi_ind = '-'.join([inchi[:5] for inchi in inchi_ind.split('-')])

                    if c["add-joint"]:
                        pre_align(E,qt,unpair,G,PG,Radj_mat=adj_mat,Padj_mat=Padj_mat,working_folder=c["output_path"],ff=c["ff"],Pname='{}_{}_{}'.format(inchi_ind,pind,ind))

                    # write product xyz file
                    product_xyz = c["output_path"]+'/products_init/{}_{}_{}.xyz'.format(inchi_ind,pind,ind)
                    xyz_write(product_xyz,E,PG)
                    conf_pair  += [(conf_folder,product_xyz,"{}_{}_{}".format(inchi_ind,pind,ind))]

    # input_type is 1, generate input files for GSM from input folder and crest results
    elif c['input_type']==1:

        input_xyzs = sorted([i for i in os.listdir(c['input_xyz']) if fnmatch.fnmatch(i, '*.xyz')])

        # conf_pair stores the information of CREST output folder, product xyz file and name for each reaction pathway
        conf_pair = []

        # iterate over each xyz file to identify 
        for xyz in input_files:
            
            # reactant xyz files
            conf_folder = '{}/{}/results'.format(conf_path,R_list_refer[xyz])

            # parse product info
            PE,_,PG = parse_input(c['input_xyz']+'/'+xyz)
            product_xyz = c["output_path"]+'/products_init/{}'.format(xyz)
            xyz_write(product_xyz,PE,PG)

            # write product xyz file
            conf_pair  += [(conf_folder,product_xyz,xyz.split('.xyz')[0])]

    ######################################################
    ####   Submit conf_gen jobs if input type is 0/1  ####
    ######################################################
    if c['input_type'] == 0 or c['input_type'] == 1:

        input_folder = c["output_path"]+'/input_files_conf'
        submited = submit_select(conf_pair,input_folder,Njobs=c["low_njobs"],Nmax=c["n_max"],ff=c["ff"],sched=c["sched"],queue=c["low_queue"],rank_by_energy=False)

        substring="python {}/utilities/job_submit.py -f 'CONF_GEN.*.submit' -sched {}".format('/'.join(os.getcwd().split('/')[:-1]),c["sched"])
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        if c["batch"] == 'pbs': print("\t running {} CONF_GEN jobs...".format(int(len(output.split()))))
        elif c["batch"] == 'slurm': print("\t running {} CONF_GEN jobs...".format(int(len(output.split())/4)))
    
        monitor_jobs(output.split())

    else:
        input_folder = c['input_xyz']
    
    # find all xyz files in final input folder ( if input_types==2, directly jumps to this step)
    final_products = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(input_folder) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])
    if c["select_conf"]:
        N_dict,F_dict,adj_dict,hash_dict = {},{},{},{}
        conf_list = {}
        for i in sorted(final_products):
            name = i.split('/')[-1].split('.xyz')[0]
            if '_'.join(name.split('_')[:-1]) not in conf_list.keys():
                conf_list['_'.join(name.split('_')[:-1])] = name
                NN_dict,NF_dict,Nadj_dict,Nhash_dict = parse_pyGSM_inputs([i],E_dict)
                N_dict[name],F_dict[name],adj_dict[name],hash_dict[name] = NN_dict[name],NF_dict[name],Nadj_dict[name],Nhash_dict[name]
            else:
                pname = conf_list['_'.join(name.split('_')[:-1])]
                N_dict[name],F_dict[name],adj_dict[name],hash_dict[name] = N_dict[pname],F_dict[pname],adj_dict[pname],hash_dict[pname]

    else:
        N_dict,F_dict,adj_dict,hash_dict = parse_pyGSM_inputs(final_products,E_dict)

    ##################################################
    ##    Run GSM  calculation for given products   ##
    ##################################################
    qt,unpair = int(c["charge"]),int(c["unpair"])

    # Generate GSM submit files
    output_list = submit_GSM(final_products,c["output_path"],int(c["low_njobs"]),c["pygsm_path"],level=c["level"],procs=c["low_procs"],Wt=c["low_wt"],sched=c["sched"],queue=c["low_queue"],\
                             charge=qt,unpair=unpair,Nimage=int(c["nimage"]),conv_tor=c["conv-tor"],add_tor=c["add-tor"],relax_end=c["relax_end"],package='Orca')

    # submit GSM jobs if is needed
    if c["restart_step"] == "None":

        substring="python {}/utilities/job_submit.py -f 'GSM.*.submit' -sched {}".format('/'.join(os.getcwd().split('/')[:-1]),c["sched"])
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
    
        if c["batch"] == 'pbs': print("\t running {} GSM jobs...".format(int(len(output.split()))))
        elif c["batch"] == 'slurm': print("\t running {} GSM jobs...".format(int(len(output.split())/4)))
    
        monitor_jobs(output.split())
        os.system('rm GSM.*.submit')

    ############################################################
    # Analyze the output files and get the initial guess of TS #
    ############################################################
    # make a initial TS guess folder
    ini_folder = c["output_path"] + '/Initial_TS'
    if os.path.isdir(ini_folder) is False: os.mkdir(ini_folder)

    # generate input file list
    TSinp_list = []

    # loop over all of result folders
    for output_folder in output_list:

        # read pyGSM output file and return product name
        pname = read_pyGSM_output(output_folder,ini_folder,N_dict)
        
        if pname:

            # use xyz_to_Orca to tranfer a xyz file to inp file for low level TS-opt
            if len(c["level"].split('/')) == 1:
                low_level = '-f {}'.format(c["level"])
            else:
                low_level = '-f {} -b {}'.format(c["level"].split('/')[0],c["level"].split('/')[1])

            substring = "python {}/utilities/xyz_to_Orca.py {}/{}-TS.xyz -p {} -o {}/{}/{}-TS -q {} -m {} {} -hs {} -ty \"OptTS Freq \" "
            substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),ini_folder,pname,c["low_procs"],c["output_path"],pname,pname,qt,unpair+1,low_level,c["hess-step"])
            os.system(substring)

            # add input file to the list
            TSinp_list += [c["output_path"]+'/{}/{}-TS.in'.format(pname,pname)]
            
        else:
            print("pyGSM for {} fails / locates to a wrong TS".format(output_folder.split('/')[-1]))

    # submit low level TS-opt jobs if is needed
    if c["restart_step"] in ["None","GSM"]:

        substring="python {}/utilities/Orca_submit.py -f '*.in' -ff \"{}\" -d {} -para perpendicular -p {} -n {} -ppn {} -q {} -mem {} -sched {} -t {} -o TS --silent"
        substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),TSinp_list,c["output_path"],c["low_procs"],c["low_njobs"],c["low_procs"],c["low_queue"],c["memory"],c["sched"],c["low_wt"]) 
        os.system(substring)

        # submit all the jobs
        substring="python {}/utilities/job_submit.py -f 'TS.*.submit' -sched {}".format('/'.join(os.getcwd().split('/')[:-1]),c["sched"])
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
                    
        if c["batch"] == 'pbs': print("\t running {} low level TS_opt jobs...".format(int(len(output.split()))))
        elif c["batch"] == 'slurm': print("\t running {} low level TS_opt jobs...".format(int(len(output.split())/4)))
        
        monitor_jobs(output.split())
        os.system('rm *.submit')

    # remove duplicate if needed
    if c["select_conf"]: TSinp_list = select_conf(TSinp_list,package='Orca')

    ###############################################
    # Analyze the TS and perform IRC at low level #
    ###############################################
    # Initialize input files for IRC 
    IRCinp_list = []

    # check job status, if TS successfully found, put into IRC list
    for TSinp in TSinp_list:

        # change .in to .out will return TS geo-opt output file
        TSout = TSinp.replace('.in','.out')
        TSxyz = TSinp.replace('.in','.xyz')

        # return file name
        pname = TSinp.split('/')[-1].split('-TS')[0]

        # imag_flag refers whether there is an imaginary frequency in TS output file; finish_flag refers to whether TS geo-opt normally finished
        finish_flag,imag_flag,SPE,zero_E,H_298,F_298,_ = read_Orca_output(TSout)

        # for the success tasks, generate optimized TS geometry
        if imag_flag and finish_flag:

            # obtain level of theory
            if len(c["level"].split('/')) == 1:
                low_level = '-f {}'.format(c["level"])
            else:
                low_level = '-f {} -b {}'.format(c["level"].split('/')[0],c["level"].split('/')[1])

            substring = "python {}/utilities/xyz_to_Orca.py {} -p {} -o {}/{}/{}-IRC -q {} -m {} {} -ty \"IRC \" -s {}"
            substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),TSxyz,c["low_procs"],c["output_path"],pname,pname,qt,unpair+1,low_level,c["irc-image"])
            os.system(substring)

            # add inp file into the list
            IRCinp_list += [c["output_path"]+'/{}/{}-IRC.in'.format(pname,pname)]
                
        else:
            print("TS for reaction payway to {} fails (either no imag freq or geo-opt fails)".format(pname))
            continue

    # submit low level IRC jobs if is needed
    if c["restart_step"] in ["None","GSM","low-TS"]:

        # Generate IRC calculation job and wait for the result
        substring="python {}/utilities/Orca_submit.py -f '*.in' -ff \"{}\" -d {} -para perpendicular -p {} -n {} -ppn {} -q {} -mem {} -sched {} -t {} -o IRC --silent"
        substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),IRCinp_list,c["output_path"],c["low_procs"],c["low_njobs"],c["low_procs"],c["low_queue"],c["memory"],c["sched"],c["low_wt"]) 
        os.system(substring)

        # submit all the jobs
        substring="python {}/utilities/job_submit.py -f 'IRC.*.submit' -sched {}".format('/'.join(os.getcwd().split('/')[:-1]),c["sched"])
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        
        if c["batch"] == 'pbs': print("\t running {} low level IRC jobs...".format(int(len(output.split()))))
        elif c["batch"] == 'slurm': print("\t running {} low level IRC jobs...".format(int(len(output.split())/4)))

        monitor_jobs(output.split())
        os.system('rm IRC.*.submit')

    # analyze low IRC result, find valuable intended and unintended channels. 
    TSinp_list,low_IRC_dict = read_IRC_outputs(c["output_path"],IRCinp_list,adj_dict,hash_dict,N_dict,reaction_dict,select=True,folder_name='low-IRC-result',package='Orca',dg_thresh=c["dg_thresh"])

    ##############################################
    # Analyze the TS and perform DFT level calc  #
    ##############################################
    # make a TS folder
    TS_folder = c["output_path"] + '/TS-folder'
    if os.path.isdir(TS_folder) is False: os.mkdir(TS_folder)

    # generate in (Orca input file) list 
    TS_DFT_inp = []

    # check job status, if TS successfully found, put into TS_DFT
    for TSinp in TSinp_list:

        # change .in to .out will return TS geo-opt output file
        TSout = TSinp.replace('.in','.out')
        TSxyz = TSinp.replace('.in','.xyz')

        # return file name
        pname = TSinp.split('/')[-1].split('-TS')[0]

        # imag_flag refers whether there is an imaginary frequency in TS output file; finish_flag refers to whether TS geo-opt normally finished
        finish_flag,imag_flag,SPE,zero_E,H_298,F_298,_ = read_Orca_output(TSout)

        # for the success tasks, generate optimized TS geometry
        if imag_flag and finish_flag:

            # obtain level of theory
            high_level = '-f {}'.format(c["functional"])
            if c["dispersion"] is not None: high_level += ' -d {}'.format(c["dispersion"])
            if c["basis"] is not None: high_level += ' -b {}'.format(c["basis"])

            substring = "python {}/utilities/xyz_to_Orca.py {} -p {} -o {}/{}-TS -q {} -m {} {} -hs {} -ty \"OptTS Freq \" "
            substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),TSxyz,c["high_procs"],TS_folder,pname,qt,unpair+1,high_level,c["hess-step"])
            os.system(substring)
            
            # add inp file to the list
            TS_DFT_inp += [TS_folder+'/{}-TS.in'.format(pname)]

        else:
            print("TS for reaction payway to {} fails (either no imag freq or geo-opt fails)".format(pname))
            continue

    # Generate TS location job and wait for the result
    if c["restart_step"] in ["None","GSM","low-TS","low-IRC"]:

        substring="python {}/utilities/Orca_submit.py -f '*.in' -ff \"{}\" -d {} -para {} -p {} -n {} -ppn {} -q {} -mem {} -sched {} -t {} -o TS --silent"
        substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),TS_DFT_inp,TS_folder,c["parallel"],c["high_procs"],c["high_njobs"],c["ppn"],c["high_queue"],c["memory"],c["sched"],c["high_wt"]) 
        os.system(substring)

        # submit all the jobs
        substring="python {}/utilities/job_submit.py -f 'TS.*.submit' -sched {}".format('/'.join(os.getcwd().split('/')[:-1]),c["sched"])
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
                    
        if c["batch"] == 'pbs': print("\t running {} high level TS_opt jobs...".format(int(len(output.split()))))
        elif c["batch"] == 'slurm': print("\t running {} high level TS_opt jobs...".format(int(len(output.split())/4)))

        monitor_jobs(output.split())
        os.system('rm TS.*.submit')

    ########################################
    # Analyze the TS and classify channels #
    ########################################
    N_grad,reaction_dict,TS_DFT_inp = channel_classification(output_path=c["output_path"],initial_xyz=input_folder,inp_list=TS_DFT_inp,low_IRC_dict=low_IRC_dict,N_dict=N_dict,adj_dict=adj_dict,F_dict=F_dict,reaction_dict=reaction_dict,\
                                                             append=False,thresh=0.5,model=c["irc-model"],package='Orca',dg_thresh=c["dg_thresh"])

    print("Total number of gradient calls is {}".format(N_grad))

    ###############################################
    # Analyze the TS and perform IRC at DFT level #
    ###############################################
    if c["dft-irc"]:
    
        # make a IRC folder
        IRC_folder = c["output_path"] + '/IRC-folder'
        if os.path.isdir(IRC_folder) is False: os.mkdir(IRC_folder)

        # Initialize input files for IRC 
        IRCinp_list = []

        # TS gibbs free energy dictionary
        TSEne_dict={}

        # check job status, if TS successfully found, put into IRC list
        for TSinp in TS_DFT_inp:

            # change .in to .out will return TS geo-opt output file
            TSout = TSinp.replace('.in','.out')
            TSxyz = TSinp.replace('.in','.xyz')

            # return file name
            pname = TSinp.split('/')[-1].split('-TS')[0]

            # imag_flag refers whether there is an imaginary frequency in TS output file; finish_flag refers to whether TS geo-opt normally finished
            finish_flag,imag_flag,SPE,zero_E,H_298,F_298,_ = read_Orca_output(TSout)

            # for the success tasks, generate optimized TS geometry
            if imag_flag and finish_flag:

                TSEne_dict[pname]=F_298
                print("TS for reaction payway to {} is found with Energy barrier {}".format(pname,(F_298-F_dict[pname])*627.5))

                # obtain level of theory
                high_level = '-f {}'.format(c["functional"])
                if c["dispersion"] is not None: high_level += ' -d {}'.format(c["dispersion"])
                if c["basis"] is not None: high_level += ' -b {}'.format(c["basis"])

                # generate Orca IRC input files
                substring = "python {}/utilities/xyz_to_Orca.py {} -p {} -o {}/{}-IRC -q {} -m {} {} -ty \"IRC \" -s {}"
                substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),TSxyz,c["high_procs"],IRC_folder,pname,qt,unpair+1,high_level,c["irc-image"])
                os.system(substring)

                # add gjf file into the list
                IRCinp_list += [IRC_folder+'/{}-IRC.in'.format(pname)]

            else:
                print("TS for reaction payway to {} fails (either no imag freq or geo-opt fails)".format(pname))
                continue

        if c["restart_step"] in ["None","GSM","low-TS","low-IRC","DFT-TS"]:

            if len(IRCinp_list) > 0:
                substring="python {}/utilities/Orca_submit.py -f '*.in' -ff \"{}\" -d {} -para {} -p {} -n {} -ppn {} -q {} -mem {} -sched {} -t {} -o IRC --silent"
                substring = substring.format('/'.join(os.getcwd().split('/')[:-1]),IRCinp_list,IRC_folder,c["parallel"],c["high_procs"],c["high_njobs"],c["ppn"],c["high_queue"],c["memory"],c["sched"],c["high_wt"]) 
                os.system(substring)
                
                # submit all the jobs
                substring="python {}/utilities/job_submit.py -f 'IRC.*.submit' -sched {}".format('/'.join(os.getcwd().split('/')[:-1]),c["sched"])
                output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        
                if c["batch"] == 'pbs': print("\t running {} high level IRC jobs...".format(int(len(output.split()))))
                elif c["batch"] == 'slurm': print("\t running {} high level IRC jobs...".format(int(len(output.split())/4)))

                monitor_jobs(output.split())
                os.system('rm *.submit')

        reaction_dict = read_IRC_outputs(c["output_path"],IRCinp_list,adj_dict,hash_dict,N_dict,reaction_dict,F_dict,TSEne_dict,append=False,package='Orca')

    ##########################################################
    # Analyze the IRC result and return product and reactant #
    ##########################################################
    # write updated reactant and rection dictionries into db file
    if c["apply_tcit"] and c["input_type"] == 0:
        with open(c["reactant_dict"],'wb') as fp:
            pickle.dump(reactant_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

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
    keywords = ["input_type","reactant_dict","reaction_dict","input_xyz","output_path","input_react","pygsm_path","e_dict","ff","apply_tcit","tcit_result","hf_cut","charge","unpair","hess-step",\
                "batch","sched","restart_step","select_conf","dft-irc","memory","irc-image","irc-model","dg_thresh","compute_product","level","nimage","add-tor","conv-tor","relax_end","low_wt","low_njobs",\
                "low_queue","low_procs","c_wt","c_njobs","c_nprocs","n_max","c_queue","add-joint","c_path","functional","basis","dispersion","high_wt","ppn","high_procs","high_njobs","high_queue","parallel"]

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
    elif configs["dispersion"] not in ["D2", "D3", "D3ZERO", "D3BJ", "D4"]:
        print("Orca only supports D2, D3 and D4 Empirical Dispersion, check inputs...")
        quit()

    if configs["basis"].lower() == 'none': 
        configs["basis"] = None

    # Check that relax end is an acceptable system
    if configs["restart_step"] not in ["None","GSM","low-TS","DFT-TS","low-IRC","IRC"]:
        print("ERROR in locate_TS: only None, GSM, low-TS, DFT-TS and IRC are acceptable")
        quit()

    # set default value
    # input_type (default:0)
    if configs["input_type"] is None:
        configs["input_type"] = 0
    else:
        configs["input_type"] = int(configs["input_type"])
    
    # apply TCIT
    if configs["apply_tcit"] is None:
        configs["apply_tcit"] = True

    elif configs["apply_tcit"].lower() == 'false':
        configs["apply_tcit"] = False

    else:
        configs["apply_tcit"] = True

    # fixed two end nodes
    if configs["relax_end"] is None:
        configs["relax_end"] = True

    elif configs["relax_end"].lower() == 'false':
        configs["relax_end"] = False

    else:
        configs["relax_end"] = True
    
    # apply joint optimization
    if configs["add-joint"] is None:
        configs["add-joint"] = True

    elif configs["add-joint"].lower() == 'false':
        configs["add-joint"] = False

    else:
        configs["add-joint"] = True
        
    # set default value for dg_thresh 
    if configs["dg_thresh"] is None or configs["dg_thresh"].lower() == 'none':
        configs["dg_thresh"] = None
    else:
        configs["dg_thresh"] = int(configs["dg_thresh"])

    # apply product optimization 
    if configs["compute_product"] is None:
        configs["compute_product"] = False

    elif configs["compute_product"].lower() == 'true':
        configs["compute_product"] = True

    else:
        configs["compute_product"] = False

    # remove duplicate TSs or not
    if configs["select_conf"] is None:
        configs["select_conf"] = False

    elif configs["select_conf"].lower() == 'false':
        configs["select_conf"] = False

    else:
        configs["select_conf"] = True

    # IRC at DFT level
    if configs["dft-irc"] is None:
        configs["dft-irc"] = False

    elif configs["dft-irc"].lower() == 'false':
        configs["dft-irc"] = False

    else:
        configs["dft-irc"] = True
    
    # Nimage (default: 9)
    if configs["nimage"] is None:
        configs["nimage"] = 9

    else:
        configs["nimage"] = int(configs["nimage"])

    # add-tor (default: 0.01)
    if configs["add-tor"] is None:
        configs["add-tor"] = 0.01
    else:
        configs["add-tor"] = float(configs["add-tor"])

    # conv-tor (default: 0.0005)
    if configs["conv-tor"] is None:
        configs["conv-tor"] = 0.0005
    else:
        configs["conv-tor"] = float(configs["conv-tor"])

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

