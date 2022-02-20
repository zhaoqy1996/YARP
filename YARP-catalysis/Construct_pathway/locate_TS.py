import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
from operator import add
import numpy as np
import ast,collections

# import ase modules
from ase import io
from ase.build import minimize_rotation_and_translation

# Load modules in same folder        
sys.path.append('../utilities')
from xtb_functions import xtb_energy,xtb_geo_opt
from taffi_functions import *
from job_submit import *

def main(argv):

    parser = argparse.ArgumentParser(description='Driver script for submitting GSM-xTB jobs. This program can take one reactant xyz file and \
                                     a product xyz file folder and apply GSM to search for reaction pathways.')

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

    # run GSM calculation
    run_GSM(c)

    return

def run_GSM(c):

    # create folders
    if os.path.isdir(c["output_path"]) is False:
        os.mkdir(c["output_path"])

    # parse energy dictionary
    E_dict=parse_Energy(c["e_dict"])

    # Analyze the reactant
    E,G      = xyz_parse(c["reactant_xyz"])
    qt       = parse_q(c["reactant_xyz"])      
    adj_mat  = Table_generator(E,G)
    hash_list= return_hash(E,G,adj_mat)

    # check fragments of the reactant, if this is a multi-molecular reaction, seperate each reactant
    gs = graph_seps(adj_mat)
    groups = []   # initialize the list for index of all reactants
    loop_ind = []
    for i in range(len(gs)):
        if i not in loop_ind:
            new_group = [count_j for count_j,j in enumerate(gs[i,:]) if j >= 0]
            loop_ind += new_group
            groups   += [new_group]

    # get inchikey for each compound seperately
    inchi_list = []
    group_dict = {} 
    for group in groups:
        # numver of atoms in this reactant
        N_atom = len(group)
        # Elements and geometry of this reactant 
        frag_E = [E[ind] for ind in group]
        frag_G = np.zeros([N_atom,3])
        for count_i,i in enumerate(group):
            frag_G[count_i,:] = G[i,:]
        # Determine the inchikey of this reactant
        inchikey   = return_inchikey(frag_E,frag_G)
        inchi_list+= [inchikey]
        group_dict[inchikey]=group

    # load in reactant geometry
    reactant  = io.read(c["reactant_xyz"])

    # initialize dictionaries
    N_element= len(reactant.get_atomic_numbers())
    F_dict   = {}    # Gibbs free energy dictionary
    adj_dict = {}    # adj_mat dictionary     
    hash_dict= {}    # hash_list dictionary
    N_dict   = {}
    N_heavy  = {}

    # input_type is 0, generate input for GSM
    if c["input_type"] == 0:

        # create folders 
        if os.path.isdir(c["output_path"]+'/opt-folder') is False:
            os.mkdir(c["output_path"]+'/opt-folder')

        if os.path.isdir(c["output_path"]+'/input_files') is False:
            os.mkdir(c["output_path"]+'/input_files')

        if os.path.isdir(c["output_path"]+'/xTB-folder') is False:
            os.mkdir(c["output_path"]+'/xTB-folder')
    
        if c["pre-opt"] is True:
            
            # reactant xyz file
            reactant_opt = c["output_path"]+'/opt-folder/'+(c["reactant_xyz"].split('/')[-1].split('.')[0]+'-opt.xyz')

            # apply geometry optimization for products
            final_products= []

            # determine fixed atoms
            fixed_atoms = [ind + 1 for ind in c["surface"]]

            # loop over all products
            for lp,product_xyz in enumerate(c["products"]):

                # load in product xyz file and generate adj_mat
                PE,PG   = xyz_parse(product_xyz)
                Padj_mat= Table_generator(PE,PG)

                # Similar as avove, use graph_seps to seperate all of the product(s)
                P_gs    = graph_seps(Padj_mat)
                P_groups = []
                loop_ind = []

                # loop over all of components in this product
                for i in range(len(P_gs)):
                    if i not in loop_ind:
                        new_group = [count_j for count_j,j in enumerate(P_gs[i,:]) if j >= 0]
                        loop_ind += new_group
                        P_groups   += [new_group]

                # return inchikey of all components
                P_inchi_list = []
                for group in P_groups:
                    N_atom = len(group)
                    frag_E = [PE[ind] for ind in group]
                    frag_G = np.zeros([N_atom,3])
                    for count_i,i in enumerate(group):
                        frag_G[count_i,:] = PG[i,:]

                    inchikey  = return_inchikey(frag_E,frag_G)
                    P_inchi_list+= [inchikey]

                # Determine the inchikey before xTB geo-opt
                oinchi = return_inchikey(PE,PG)

                # Apply xtb geo-opt on the product
                product_opt = c["output_path"]+'/opt-folder/'+(product_xyz.split('/')[-1].split('.')[0]+'-opt.xyz')
                Energy,opted_geo,finish = xtb_geo_opt(product_xyz,charge=c["charge"],unpair=c["unpair"],namespace='product_{}'.format(lp),fixed_atoms=fixed_atoms,\
                                                      workdir=c["output_path"]+'/xTB-folder',level='normal',output_xyz=product_opt,cleanup=False)

                # If geo-opt fails to converge, skip this product...
                if not finish:
                    continue

                # Determine the inchikey of the product after xTB geo-opt
                PE,PG=xyz_parse(opted_geo)
                ninchi = return_inchikey(PE,PG)

                # Check whether geo-opt changes the product. If so, geo-opt fails
                if oinchi[:14] == ninchi[:14]:

                    # Determine the adj_mat and hash list of this product
                    Padj_mat= Table_generator(PE,PG)
                    _,_,Phash_list=canon_geo(PE,Padj_mat)

                    # take the product geometry and apply ff-opt to regenerate reactant geo
                    new_G = ob_geo_opt(PE,PG,adj_mat,ff='uff',fixed_atoms=fixed_atoms,step=500)
                    tmpxyz=c["output_path"]+'/xTB-folder/opt-reactant.xyz'
                    xyz_write(tmpxyz,E,new_G)

                    # obtain the inchikey before geo_opt
                    NRE,NRG=xyz_parse(tmpxyz)
                    oinchi = return_inchikey(NRE,NRG)

                    # apply xtb geo-opt on reactant
                    Energy,reactant_opt_geo,finish = xtb_geo_opt(tmpxyz,charge=c["charge"],unpair=c["unpair"],namespace='reactant',workdir=c["output_path"]+'/xTB-folder',\
                                                                 fixed_atoms=fixed_atoms,level='normal',output_xyz=reactant_opt,cleanup=False)
                    if not finish:
                        continue

                    NRE,NRG=xyz_parse(reactant_opt_geo)
                    ninchi = return_inchikey(NRE,NRG)

                    # Check whether geo-opt changes the product. If so, geo-opt fails
                    if oinchi[:14] != ninchi[:14]:
                        continue

                    # Apply ase minimize_rotation_and_translation to optinize the reaction pathway
                    #reactant= io.read(reactant_opt_geo)
                    #product = io.read(opted_geo)
                    #minimize_rotation_and_translation(reactant,product)
                    #io.write(opted_geo,product)

                    # generate input files for GSM
                    product_name = opted_geo.split('/')[-1].split('-opt')[0]+'.xyz'
                    F_dict[opted_geo.split('/')[-1].split('-opt')[0]] = sum([E_dict[inchi]['F'] for inchi in inchi_list])
                    N_dict[opted_geo.split('/')[-1].split('-opt')[0]] = len(reactant.get_atomic_numbers())
                    N_heavy[opted_geo.split('/')[-1].split('-opt')[0]] = len([Ei for Ei in PE if Ei != 'H'])
                    adj_dict[opted_geo.split('/')[-1].split('-opt')[0]] = {}
                    adj_dict[opted_geo.split('/')[-1].split('-opt')[0]]["reactant"]= adj_mat
                    adj_dict[opted_geo.split('/')[-1].split('-opt')[0]]["product"] = Padj_mat
                    hash_dict[opted_geo.split('/')[-1].split('-opt')[0]] = {}
                    hash_dict[opted_geo.split('/')[-1].split('-opt')[0]]["reactant"]= hash_list
                    hash_dict[opted_geo.split('/')[-1].split('-opt')[0]]["product"] = Phash_list
                    
                    # cat reactant and product together
                    command_line = "cd {}/{};mv {} {};cat {} >> {}".format(c["output_path"],'input_files',reactant_opt_geo,product_name,opted_geo,product_name)
                    os.system(command_line)
                    final_products+= ["{}/input_files/{}".format(c["output_path"],product_name)]

                else:
                    print("xtb geo-opt fails for {}, remove this one from the list...".format(opted_geo))

        else:
            final_products= []
            for lp,product_xyz in enumerate(c["products"]):
                
                # get the name of file
                product_name = product_xyz.split('/')[-1]

                # cat reactant and product together
                command_line = "cd {}/{};cp {} {};cat {} >> {}".format(c["output_path"],'input_files',c["reactant_xyz"],product_name,product_xyz,product_name)
                os.system(command_line)
                final_products+= ["{}/input_files/{}".format(c["output_path"],product_name)]

    else:
        print("Directly take xyz_files ready for GSM...")
        # find all xyz files in the input folder
        final_products = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(c["input_xyz"]) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])

        # loop over all pairs of reactant & product
        for count_i,i in enumerate(sorted(final_products)):
            # get the name of xyz_file
            name = i.split('/')[-1].split('.xyz')[0]
            
            # create dictionary of adj_mat and hash list
            adj_dict[name] ={}
            hash_dict[name]={}

            # initialize list for reactant/product geometry
            xyz = ['','']
            count=0

            # read in pairs of xyz file
            with open(i,"r") as f:
                for lc,lines in enumerate(f):
                    fields = lines.split()
                    if lc == 0: 
                        N = int(fields[0])
                        xyz[0] += lines
                        continue

                    if len(fields) == 1 and int(fields[0]) == N:
                        count+=1

                    xyz[count]+=lines

            with open('reactant.xyz',"w") as f:
                f.write(xyz[0])

            with open('product.xyz',"w") as f:
                f.write(xyz[1])

            # parse reactant info
            RE,RG   = xyz_parse('reactant.xyz')
            Radj_mat= Table_generator(RE,RG)
            _,_,Rhash_list=canon_geo(RE,Radj_mat)

            # parse product info
            PE,PG   = xyz_parse('product.xyz')
            Padj_mat= Table_generator(PE,PG)
            _,_,Phash_list=canon_geo(PE,Padj_mat)

            # reomve xyz files
            try:
                os.remove('reactant.xyz')
                os.remove('product.xyz')
            except:
                pass

            # Seperate reactant(s)
            R_gs    = graph_seps(Radj_mat)
            Rgroups = []
            loop_ind= []
            for i in range(len(R_gs)):
                if i not in loop_ind:
                    new_group = [count_j for count_j,j in enumerate(R_gs[i,:]) if j >= 0]
                    loop_ind += new_group
                    Rgroups   += [new_group]
            
            # Determine the inchikey of all components in the reactant
            Rinchi_list = []
            for group in Rgroups:
                N_atom = len(group)
                frag_E = [RE[ind] for ind in group]
                frag_G = np.zeros([N_atom,3])
                for count_i,i in enumerate(group):
                    frag_G[count_i,:] = RG[i,:]
                
                inchikey = return_inchikey(frag_E,frag_G)
                Rinchi_list+= [inchikey]

            # write info into dictionaies
            N_dict[name] = len(RE)
            N_heavy[name]= len([Ei for Ei in PE if Ei != 'H'])
            F_dict[name] = sum([E_dict[inchi]['F'] for inchi in Rinchi_list])
            adj_dict[name]["reactant"]= Radj_mat                
            adj_dict[name]["product"] = Padj_mat
            hash_dict[name]["reactant"]= Rhash_list              
            hash_dict[name]["product"] = Phash_list


    ##################################################
    ##  Run GSM-xTB calculation for given products  ##
    ##################################################
    #os.system('rm *.submit')
    # Generate submit files
    output_list = submit_GSM(final_products,c["output_path"],int(c["njobs"]),Wt=c["wt"],sched=c["sched"],queue=c["queue"],charge=c["charge"],unpair=c["unpair"],\
                             multiplicity=c["multiplicity"],Nimage=int(c["nimage"]),conv_tor=c["conv-tor"],add_tor=c["add-tor"])
    '''
    # submit jobs
    substring="python ../utilities/job_submit.py -f '*.submit' -sched {}".format(c["sched"])
    output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
    
    if c["batch"] == 'pbs':
        print("\t running {} GSM-xTB jobs...".format(len(output.split())))
    elif c["batch"] == 'slurm':
        print("\t running {} GSM-xTB jobs...".format(int(len(output.split())/4)))
    
    monitor_jobs(output.split())
    '''
    os.system('rm *.submit')

    ############################################################
    # Analyze the output files and get the initial guess of TS #
    ############################################################
    # make a TS folder
    TS_folder = c["output_path"] + '/TS-folder'
    if os.path.isdir(TS_folder) is False:
        os.mkdir(TS_folder)
        
    # generate gjf (Gaussian input file) list
    TSgjf_list = []

    # loop over all of result folders
    for output_folder in output_list:

        # load in GSM output log file
        with open(output_folder+'/log','r') as f:
            lines = f.readlines()

        # check success
        if 'Finished GSM!\n' in lines:

            pname = output_folder.split('/')[-1]
            print("pyGSM-xTB for {} finished".format(pname))

            # obtain the location of TS
            for line in lines:
                fields = line.split()
                if len(fields) == 12 and fields[0]=='min' and fields[2]=='node:' and fields[8]=='TS':
                    N_TS = int(fields[-1])
                    
                # obatin GSM-xTB barrier height
                if len(fields) == 3 and fields[0]=='TS' and fields[1]=='energy:':
                    E_TS = float(fields[2])

            # if TS energy is so high, we assert GSM locates to a wrong TS
            if E_TS < 1000:

                # find the geometry of TS
                with open(output_folder+'/scratch/opt_converged_000_000.xyz','r') as g:
                    lines = g.readlines()
                    count = 0
                    write_lines = []
                    for lc,line in enumerate(lines):
                        fields = line.split()
                        if len(fields)==1 and fields[0] == str(N_dict[pname]):
                            count += 1
                        
                        if count == N_TS + 1:
                            write_lines += [line]
                        
                # write the TS initial guess xyz file 
                output_xyz = TS_folder+'/{}-TS.xyz'.format(pname)
                with open(output_xyz,'w') as g:
                    for line in write_lines:
                        g.write(line)

                # use xyz_to_Gaussian to tranfer a xyz file to gjf
                substring = "python ../utilities/xyz_to_Gaussian.py {}/{}-TS.xyz -o {}/{}-TS.gjf -q {} -m {} -c False" + \
                            " -ty \"{}/{} OPT=(TS, CALCALL, NOEIGEN, maxcycles=100) Freq \" -t \"{} TS\" "
                substring = substring.format(TS_folder,pname,TS_folder,pname,c["charge"],c["multiplicity"],c["functional"],c["basis"],pname)
                os.system(substring)
                
                # add gjf file to the list
                TSgjf_list += [TS_folder+'/{}-TS.gjf'.format(pname)]

            else:
                print("pyGSM-xTB for {} locates to a wrong TS".format(output_folder.split('/')[-1]))

        else:
            print("pyGSM-xTB for {} failed".format(output_folder.split('/')[-1]))

    # Generate TS location job and wait for the result
    #'''
    substring="python ../utilities/Gaussian_submit.py -f '*.gjf' -ff \"{}\" -d {} -para {} -p {} -n {} -ppn {} "+\
              "-q {} -mem {} -sched {} -t {} -o TS --silent"
    substring = substring.format(TSgjf_list,TS_folder,c["parallel"],c["procs"],c["njobs2"],c["ppn"],c["queue2"],c["memory"],c["sched"],c["wt2"]) 
    os.system(substring)

    # submit all the jobs
    substring="python ../utilities/job_submit.py -f '*.submit' -sched {}".format(c["sched"])
    output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        
    if c["batch"] == 'pbs':
        print("\t running {} Gaussian TS_opt jobs...".format(len(output.split())))
    elif c["batch"] == 'slurm':
        print("\t running {} Gaussian TS_opt jobs...".format(int(len(output.split())/4)))
    monitor_jobs(output.split())
    #'''
    os.system('rm *.submit')

    ##############################################
    # Analyze the TS and perform IRC calculation #
    ##############################################
    # make a IRC folder
    IRC_folder = c["output_path"] + '/IRC-folder'
    if os.path.isdir(IRC_folder) is False:
        os.mkdir(IRC_folder)

    # Initialize gjf files for IRC 
    IRCgjf_list = []

    # count Number of gradient calls
    N_grad = 0

    # check job status, if TS successfully found, put into IRC list
    for TSgjf in TSgjf_list:

        # change .gjf to .out will return TS geo-opt output file
        TSout = TSgjf.replace('.gjf','.out')

        # return file name
        pname = TSgjf.split('/')[-1].split('-TS')[0]

        # imag_flag refers whether there is an imaginary frequency in TS output file; finish_flag refers to whether TS geo-opt normally finished
        imag_flag  = False
        finish_flag= False

        # parse Gaussian TS geo-opt output file
        with open(TSout,'r') as f:
            for lc,lines in enumerate(f):
                fields = lines.split()
                if 'imaginary' in fields and 'frequencies' in fields and '(negative' in fields:
                    imag_flag=True

                if 'Normal' in fields and 'termination' in fields and 'of' in fields and 'Gaussian' in fields:
                    finish_flag=True

                if len(fields) == 8 and fields[0] == 'Sum' and fields[2] == 'electronic' and fields[5] == 'Free' and fields[6] == 'Energies=':
                    E_TS = float(fields[7])

        # for the success tasks, generate optimized TS geometry
        if imag_flag and finish_flag:
            print("TS for reaction payway to {} is found with Energy barrier {}".format(pname,(E_TS-F_dict[pname])*630.0))
            command='python ../utilities/read_Gaussian_output.py -t geo-opt -i {} -o {}/{}-TS.xyz -n {} --count'
            # apply read_Gaussian_output.py to obatin optimized TS geometry
            os.system(command.format(TSout,IRC_folder,pname,N_dict[pname]))

            # get number of gradient calls for each TS geo-opt jobs
            with open('{}/{}-TS.xyz'.format(IRC_folder,pname)) as ff:
                for lc,lines in enumerate(ff):
                    if lc == 1:
                        fields = lines.split()
                        if len(fields) == 6:
                            N_g = int(fields[-1])

            # add gradient calls to Total gradient calls 
            N_grad += N_g

            # generate IRC Gaussian input files
            substring = "python ../utilities/xyz_to_Gaussian.py {}/{}-TS.xyz -o {}/{}-IRC.gjf -q {} -m {} -c False" + \
                        " -ty \"{}/{} IRC=(LQA, recorrect=never, CalcFC, maxpoints={}, StepSize={}, maxcyc=100, Report=Cartesians)\" -t \"{} IRC\" "
            substring = substring.format(IRC_folder,pname,IRC_folder,pname,c["charge"],c["multiplicity"],c["functional"],c["basis"],c["irc-image"],c["stepsize"],pname)
            os.system(substring)

            # add gjf file into the list
            IRCgjf_list += [IRC_folder+'/{}-IRC.gjf'.format(pname)]

        else:
            print("TS for reaction payway to {} fails (either no imag freq or geo-opt fails)".format(pname))
            continue

    print("Total number of gradient calls is {}".format(N_grad))

    #'''
    # Generate IRC calculation job and wait for the result
    substring="python ../utilities/Gaussian_submit.py -f '*.gjf' -ff \"{}\" -d {} -para {} -p {} -n {} -ppn {} "+\
              "-q {} -mem {} -sched {} -t {} -o IRC --silent"
    substring = substring.format(IRCgjf_list,IRC_folder,c["parallel"],c["procs"],c["njobs2"],c["ppn"],c["queue2"],c["memory"],c["sched"],c["wt2"]) 
    os.system(substring)

    # submit all the jobs
    substring="python ../utilities/job_submit.py -f '*.submit' -sched {}".format(c["sched"])
    output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        
    if c["batch"] == 'pbs':
        print("\t running {} Gaussian IRC jobs...".format(len(output.split())))
    elif c["batch"] == 'slurm':
        print("\t running {} Gaussian IRC jobs...".format(int(len(output.split())/4)))

    monitor_jobs(output.split())
    #'''

    ##########################################################
    # Analyze the IRC result and return product and reactant #
    ##########################################################
    # make a IRC result folder
    IRC_result = c["output_path"] + '/IRC-result'
    if os.path.isdir(IRC_result) is False:
        os.mkdir(IRC_result)

    # create record.txt to write the IRC result
    with open(IRC_result+'/record.txt','w') as g:
        g.write('{:<10s} {:<40s} {:<40s} \n'.format('product','Node1','Node2'))

    # initilize IRC result list
    IRC_result_list = []
    
    # loop over IRC output files
    for IRCgjf in IRCgjf_list:

        # change .gjf to .out will return IRC output file  
        IRCout = IRCgjf.replace('.gjf','.out')

        # obtain file name
        pname = IRCgjf.split('/')[-1].split('-IRC')[0] 

        # create finish_flag to check whether IRC task is normally finished
        finish_flag=False

        # read the IRC output file, chech whether it is finished and what is final image number
        with open(IRCout,'r') as f:
            for lc,lines in enumerate(f):
                fields = lines.split()
                if 'Normal' in fields and 'termination' in fields and 'of' in fields and 'Gaussian' in fields:
                    finish_flag=True
                
                if len(fields)== 5 and fields[0]=='Total' and fields[1]=='number' and fields[2]=='of' and fields[3]=='points:':
                    N_image = int(fields[4]) + 1

        # If IRC task finished, parse IRC output file
        if finish_flag:
            print("IRC for reaction payway to {} is finished".format(pname))
            # apply read_Gaussian_output.py to generate IRC pathway file
            command='python ../utilities/read_Gaussian_output.py -t IRC -i {} -o {}/{}-IRC.xyz -n {}'
            os.system(command.format(IRCout,IRC_folder,pname,N_dict[pname]))

            # find the geometry of reactant & product
            with open("{}/{}-IRC.xyz".format(IRC_folder,pname),'r') as g:
                lines = g.readlines()
                count = 0
                write_reactant= []
                write_product = []
                for lc,line in enumerate(lines):
                    fields = line.split()
                    if len(fields)==1 and fields[0] == str(N_dict[pname]):
                        count += 1
                        
                    if count == 1:
                        write_reactant+= [line]
                        
                    if count == N_image:
                        write_product += [line]

            # write the reactant and product
            with open(IRC_result+'/{}-start.xyz'.format(pname),'w') as g:
                for line in write_reactant:
                    g.write(line)

            # parse IRC start point xyz file
            NE,NG  = xyz_parse('{}/{}-start.xyz'.format(IRC_result,pname))
            N_adj_1= Table_generator(NE,NG)
            _,_,Nhash_list1=canon_geo(NE,N_adj_1)

            # get smile string
            ssmile = return_smi(NE,NG,N_adj_1)

            # generate end point of IRC
            with open(IRC_result+'/{}-end.xyz'.format(pname),'w') as g:
                for line in write_product:
                    g.write(line)
            
            # parse IRC start point xyz file
            NE,NG  = xyz_parse('{}/{}-end.xyz'.format(IRC_result,pname))
            N_adj_2= Table_generator(NE,NG)
            _,_,Nhash_list2=canon_geo(NE,N_adj_2)

            # get smile string
            esmile = return_smi(NE,NG,N_adj_2)

            o_adj_1  = adj_dict[pname]['reactant']
            o_adj_2  = adj_dict[pname]['product']
            o_hash1  = np.array(hash_dict[pname]['reactant'])
            o_hash2  = np.array(hash_dict[pname]['product'])
            adj_diff = np.abs((N_adj_1+N_adj_2) - (o_adj_1+o_adj_2))
            hash_diff= np.abs((np.array(Nhash_list1)+np.array(Nhash_list2)) - (o_hash1+o_hash2))
            
            if adj_diff.sum() == 0 or hash_diff.sum() == 0:
                words = "Intended"

            elif adj_diff.sum() == 2:
                words = "Intended (1 bond diff)"

            else:
                words = "Unintended"

            with open(IRC_result+'/record.txt','a') as g:
                g.write('{:<10s} {:<40s} {:<40s} {:20s}\n'.format(pname,ssmile,esmile,words))

        else:
            print("IRC calculation for reaction payway to {} fails".format(pname))
            continue

    return

# Function for keeping tabs on the validity of the user supplied inputs
def parse_configuration(args):
    
    # Convert inputs to the proper data type
    if os.path.isfile(args.config) is False:
        print("ERROR in python_driver: the configuration file {} does not exist.".format(args.config))
        quit()
    
    # Process configuration file for keywords
    keywords = ["input_type","reactant_xyz","product_path","input_xyz","output_path","e_dict","charge","unpair","multiplicity",\
                "surface","pre-opt","mode","nimage","add-tor","conv-tor","batch","sched","wt","njobs","queue",\
                "functional","basis","wt2","ppn","procs","njobs2","queue2","parallel","memory","irc-image","stepsize"]

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
        print("ERROR in FOM_driver: only pbs and slurm are acceptable arguments for the batch variable.")
        quit()
    elif configs["batch"] == "pbs":
        configs["sub_cmd"] = "qsub"
    elif configs["batch"] == "slurm":
        configs["sub_cmd"] = "sbatch"

    # set default value
    # input_type (default:0)
    if configs["input_type"] is None:
        configs["input_type"] = 0
    else:
        configs["input_type"] = int(configs["input_type"])

    # pre-opt (default: True)
    if configs["pre-opt"] is None:
        configs["pre-opt"] = True
    elif configs["pre-opt"].lower() == 'false':
        configs["pre-opt"] = False
    else:
        configs["pre-opt"] = True

    # surface atom list
    if configs["surface"] is not None:
        slist = configs["surface"].split('-')
        configs["surface"] = np.arange(int(slist[0]),int(slist[1])+1)
    else:
        configs["surface"] = []

    # charge (default: 0)
    if configs["charge"] is None:
        configs["charge"] = 0
    else:
        configs["charge"] = int(configs["charge"])

    # unpair (default: 0)
    if configs["unpair"] is None:
        configs["unpair"] = 0
    else:
        configs["unpair"] = int(configs["unpair"])

    # Nimage (default: 9)
    if configs["nimage"].lower() == 'auto':
        configs["nimage"] == 'Auto'

    elif configs["nimage"] is None:
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

    # search for product xyz files 
    configs["products"] = [ os.path.join(dp, f) for dp, dn, filenames in os.walk(configs["product_path"]) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )]
    configs["products"] = sorted(configs["products"])

    return configs

# Function to parse the energy dictionary
def parse_Energy(db_files,E_dict={}):
    with open(db_files,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if lc == 0: continue
            if len(fields) ==0: continue
            if len(fields) == 4:
                if fields[0] not in E_dict.keys():
                    E_dict[fields[0]] = {}
                    E_dict[fields[0]]["E_0"]= float(fields[1])
                    E_dict[fields[0]]["H"]  = float(fields[2])
                    E_dict[fields[0]]["F"]  = float(fields[3])

    return E_dict

# Return smiles string 
def return_smi(E,G,adj_mat):
    mol_write("obabel_input.mol",E,G,adj_mat)
    substring = "obabel -imol obabel_input.mol -ocan"
    output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0]
    output = output.decode('utf-8')
    smile  = str(output.split()[0])
    os.system("rm obabel_input.mol")

    return smile

# Return smiles string 
def return_inchikey(E,G):
    xyz_write("obabel_input.xyz",E,G)
    substring = "obabel -ixyz obabel_input.xyz -oinchikey" # make sure obabel will call openbabel correctly, otherwise, specify a pathway to bin/obabel
    output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0]
    output = output.decode('utf-8')
    inchi  = str(output.split()[0])
    os.system("rm obabel_input.xyz")
    return inchi

# return hash value function
def return_hash(elements,geo,adj_mat):
    # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
    mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                 'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                 'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                 'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                 'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                 'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                 'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                 'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    # Canonicalize by sorting the elements based on hashing
    masses = [ mass_dict[i] for i in elements ]
    hash_list = [ atom_hash(i,adj_mat,masses) for i in range(len(geo)) ]

    return hash_list

# Function that sleeps the script until jobids are no longer in a running or pending state in the queue
def monitor_jobs(jobids):
    
    current_jobs = check_queue()
    while True in [ i in current_jobs for i in jobids ]:
        time.sleep(60)
        current_jobs = check_queue()  
    return

# Returns the pending and running jobids for the user as a list
def check_queue():

    # The first time this function is executed, find the user name and scheduler being used. 
    if not hasattr(check_queue, "user"):

        # Get user name
        check_queue.user = subprocess.check_output("echo ${USER}", shell=True).decode('utf-8').strip("\r\n")

        # Get batch system being used
        squeue_tmp = subprocess.Popen(['which', 'squeue'], stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').strip("\r\n")
        qstat_tmp = subprocess.Popen(['which', 'qstat'], stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').strip("\r\n")
        check_queue.sched =  None
        if "no squeue in" not in squeue_tmp:
            check_queue.sched = "slurm"
        elif "no qstat in" not in qstat_tmp:
            check_queue.sched = "pbs"
        else:
            print("ERROR in check_queue: neither slurm or pbs schedulers are being used.")
            quit()

    # Get running and pending jobs using the slurm scheduler
    if check_queue.sched == "slurm":

        # redirect a squeue call into output
        output = subprocess.check_output("squeue -l", shell=True).decode('utf-8')

        # Initialize job information dictionary
        jobs = []
        id_ind = None
        for count_i,i in enumerate(output.split('\n')):            
            fields = i.split()
            if len(fields) == 0: continue                
            if id_ind is None and "JOBID" in fields:
                id_ind = fields.index("JOBID")
                if "STATE" not in fields:
                    print("ERROR in check_queue: Could not identify STATE column in squeue -l output.")
                    quit()
                else:
                    state_ind = fields.index("STATE")
                if "USER" not in fields:
                    print("ERROR in check_queue: Could not identify USER column in squeue -l output.")
                    quit()
                else:
                    user_ind = fields.index("USER")
                continue

            # If this job belongs to the user and it is pending or running, then add it to the list of active jobs
            if id_ind is not None and fields[user_ind] == check_queue.user and fields[state_ind] in ["PENDING","RUNNING"]:
                jobs += [fields[id_ind]]

    # Get running and pending jobs using the pbs scheduler
    elif check_queue.sched == "pbs":

        # redirect a qstat call into output
        output = subprocess.check_output("qstat -f", shell=True).decode('utf-8')

        # Initialize job information dictionary
        jobs = []
        job_dict = {}
        current_key = None
        for count_i,i in enumerate(output.split('\n')):
            fields = i.split()
            if len(fields) == 0: continue
            if "Job Id" in i:

                # Check if the previous job belongs to the user and needs to be added to the pending or running list. 
                if current_key is not None:
                    if job_dict[current_key]["State"] in ["R","Q"] and job_dict[current_key]["User"] == check_queue.user:
                        jobs += [current_key]
                current_key = i.split()[2]
                job_dict[current_key] = { "State":"NA" , "Name":"NA", "Walltime":"NA", "Queue":"NA", "User":"NA"}
                continue
            if "Job_Name" == fields[0]:
                job_dict[current_key]["Name"] = fields[2]
            if "job_state" == fields[0]:
                job_dict[current_key]["State"] = fields[2]
            if "queue" == fields[0]:
                job_dict[current_key]["Queue"] = fields[2]
            if "Resource_List.walltime" == fields[0]:
                job_dict[current_key]["Walltime"] = fields[2]        
            if "Job_Owner" == fields[0]:
                job_dict[current_key]["User"] = fields[2].split("@")[0]

        # Check if the last job belongs to the user and needs to be added to the pending or running list. 
        if current_key is not None:
            if job_dict[current_key]["State"] in ["R","Q"] and job_dict[current_key]["User"] == check_queue.user:
                jobs += [current_key]

    return jobs

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

