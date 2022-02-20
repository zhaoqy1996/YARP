import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
import numpy as np

# Load modules in same folder        
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/utilities')
from xtb_functions import xtb_energy,xtb_geo_opt
from taffi_functions import *
from job_submit import *
from utility import *

def main(argv):

    parser = argparse.ArgumentParser(description='This script takes a folder of reactant/product alignments and generate conformations .')

    #optional arguments    
    parser.add_argument('coord_files', help = 'The program performs on a folder of given xyz file of reactants and products')
                                         
    parser.add_argument('-w', dest='working_folder', default='scratch',help = 'Working folder for performing xTB calculations')

    parser.add_argument('-conf', dest='conf_path', default=None,help = 'pathway to crest output folder')

    parser.add_argument('-crest', dest='crest_path', default=None,help = 'pathway to executable crest')

    parser.add_argument('-Njob', dest='N_job', default=2, help = 'Number of jobs in CONF_GEN step')

    parser.add_argument('-xtb', dest='xtb_path', default=None,help = 'pathway to executable xtb')

    parser.add_argument('-N', dest='N_max', default=10, help = 'maximum number of conformation')

    parser.add_argument('-ff', dest='force_field', default='uff',help = 'force field applied to derive product geometries')

    parser.add_argument('-c', dest='charge', default=0, help = 'charge of input systems')

    parser.add_argument('-u', dest='unpair', default=0, help = 'unpair electron of input systems')

    parser.add_argument('-s', dest='strategy', default=0, help = 'sampling strategy: 0 refers to sampling on the reactant, 1 refers to sampling on the product, 2 refers to sampling on both side')

    parser.add_argument('--product_opt', dest='popt', default=False, action='store_const',const=True,
                        help = 'When this flag is on, optimize the product at xTB level')

    parser.add_argument('--rank_by_energy', dest='rbe', default=False, action='store_const',const=True,
                        help = 'When this flag is on, the conformation will be ranked by xTB energy rather than RF probability (only valids for conformation rich cases)')

    parser.add_argument('--remove_constraints', dest='remove_constraints', default=False, action='store_const',const=True,
                        help = 'When this is flag is on, no hard constraint for indicators will be applied')

    # parse configuration dictionary (c)
    args=parser.parse_args()
    N_max = int(args.N_max)
    N_job = int(args.N_job)
    ff    = args.force_field
    crest = args.crest_path
    xtb   = args.xtb_path
    file_folder = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    try:
        start = int(args.strategy)
    except:
        print("Receiving invalid input! Sampling strategy should be 0, 1, or 2. Treate as 0...")
        start = 0
        
    # change into absolute path
    if args.coord_files[0] != '/':
        input_folder = os.getcwd()+'/'+args.coord_files
    else:
        input_folder =args.coord_files

    if args.working_folder[0] != '/':
        work_folder = os.getcwd()+'/'+args.working_folder
    else:
        work_folder = args.working_folder
        
    # make working folders
    if os.path.isdir(work_folder) is False: os.mkdir(work_folder)
    if os.path.isdir(work_folder+'/xTB-folder') is False: os.mkdir(work_folder+'/xTB-folder')
    if os.path.isdir(work_folder+'/opt-folder') is False: os.mkdir(work_folder+'/opt-folder')
    if os.path.isdir(work_folder+'/ini-inputs') is False: os.mkdir(work_folder+'/ini-inputs')

    # specify the conformer folder
    if args.conf_path is None:
        conf_path = work_folder+'/conformer'
        if os.path.isdir(work_folder+'/conformer') is False: os.mkdir(work_folder+'/conformer')
    else:
        conf_path = args.conf_path

    # set default charge and unpair
    q,unpair = args.charge,args.unpair
    if start not in [0,1,2]:
        print("Receiving invalid input! Sampling strategy should be 0, 1, or 2. Treate as 0...")
        start = 0

    # initialize reactant list for CREST
    opt_list   = []
    index_list = []
    list_refer = {}

    # iterate over input files to parse unique smiles
    input_files = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(input_folder) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])
    for i in input_files:

        E,RG,PG     = parse_input(i)
        R_adj,P_adj = Table_generator(E,RG),Table_generator(E,PG)
        R_index     = i.split('/')[-1].split('.xyz')[0]+'-R'
        P_index     = i.split('/')[-1].split('.xyz')[0]+'-P'
        #R_index     = i.split('/')[-1].split('.xyz')[0].split('_')[0]+'-R'
        #P_index     = '_'.join(i.split('/')[-1].split('.xyz')[0].split('_')[:3])+'-P'
        if start == 0: list_refer[i] = [R_index]
        if start == 1: list_refer[i] = [P_index]
        if start == 2: list_refer[i] = [R_index,P_index]

        if start != 1 and R_index not in index_list: 

            index_list += [R_index]
            reactant_file = work_folder+'/opt-folder/'+'{}.xyz'.format(R_index)
            xyz_write(reactant_file, E, RG)
            Energy, opted_geo, finish=xtb_geo_opt(reactant_file,charge=q,unpair=unpair,namespace=R_index,workdir=(work_folder+'/xTB-folder'),level='normal',output_xyz=reactant_file,cleanup=False)
            if not finish: continue
            
            # Check whether geo-opt changes the geometry. If so, geo-opt fails
            _,NRG = xyz_parse(reactant_file)
            if sum(sum(R_adj-Table_generator(E,NRG))) != 0: continue

            # add reactant file into a list
            opt_list += [reactant_file]

        if start != 0 and P_index not in index_list: 

            index_list += [P_index]
            product_file = work_folder+'/opt-folder/'+'{}.xyz'.format(P_index)
            xyz_write(product_file, E, PG)
            Energy, opted_geo, finish=xtb_geo_opt(product_file,charge=q,unpair=unpair,namespace=P_index,workdir=(work_folder+'/xTB-folder'),level='normal',output_xyz=product_file,cleanup=False)
            if not finish: continue
            
            # Check whether geo-opt changes the geometry. If so, geo-opt fails
            _,NPG = xyz_parse(product_file)
            if sum(sum(P_adj-Table_generator(E,NPG))) != 0: continue

            # add product file into a list
            opt_list += [product_file]

    # check if conformer searching are already done
    finished = os.listdir(conf_path)
    opt_list = [ind for ind in opt_list if ind.split('/')[-1].split('.xyz')[0] not in finished]

    # Submit CREST jobs 
    # modify the parameters if is needed
    os.chdir(file_folder)
    output_list = submit_crest(opt_list,work_folder,Njobs=N_job,conf_path=conf_path,Wt=4,sched='slurm',queue='standby',nprocs=8,charge=q,unpair=unpair,crest_path=crest,xtb_path=xtb)
    substring   = "python {}/utilities/job_submit.py -f 'CREST.*.submit' -sched {}".format('/'.join(os.getcwd().split('/')[:-1]),'slurm')
    output      = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
    print("\t running {} CREST jobs...".format(int(len(output.split())/4)))  # for slurm 
    monitor_jobs(output.split())

    #################################################################
    # Analyze the CREST output and prepare a input folder with conf #
    #################################################################
    # create folders 
    if os.path.isdir(work_folder+'/input_files_conf') is False:
        os.mkdir(work_folder+'/input_files_conf')

    if os.path.isdir(work_folder+'/products_init') is False:
        os.mkdir(work_folder+'/products_init')

    # conf_pair stores the information of CREST output folder, product xyz file and name for each reaction pathway
    conf_pair = []

    # iterate over each xyz file to identify 
    input_xyzs = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(input_folder) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])
    for xyz in input_files:
            
        # reactant xyz files
        name = xyz.split('/')[-1].split('.xyz')[0]
        conf_folder = ['{}/{}/results'.format(conf_path,oindex) for oindex in list_refer[xyz]]

        # parse product info
        E,RG,PG = parse_input(xyz)
        reactant_xyz = work_folder+'/products_init/{}_R.xyz'.format(xyz.split('/')[-1].split('.xyz')[0])
        product_xyz = work_folder+'/products_init/{}_P.xyz'.format(xyz.split('/')[-1].split('.xyz')[0])
        xyz_write(reactant_xyz,E,RG)
        xyz_write(product_xyz,E,PG)

        # generate initial alignments
        ini_folder = work_folder+'/ini-inputs/{}'.format(name)
        if os.path.isdir(ini_folder) is False: os.mkdir(ini_folder)
        if start == 0:
            xyz_inds   = sorted([int(f.split('.xyz')[0]) for dp, dn, filenames in os.walk(conf_folder[0]) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])
            for ind in xyz_inds:
                E,RG = xyz_parse(conf_folder[0]+'/{}.xyz'.format(ind))
                xyz_write(ini_folder+'/{}.xyz'.format(ind), E, RG)
                os.system('cat {} >> {}/{}.xyz'.format(product_xyz,ini_folder,ind))

        elif start == 1:
            xyz_inds   = sorted([int(f.split('.xyz')[0]) for dp, dn, filenames in os.walk(conf_folder[0]) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])
            for ind in xyz_inds:
                E,PG = xyz_parse(conf_folder[0]+'/{}.xyz'.format(ind))
                xyz_write(ini_folder+'/{}.xyz'.format(ind), E, PG, comment = 'input_type: backward')
                os.system('cat {} >> {}/{}.xyz'.format(reactant_xyz,ini_folder,ind))

        else:
            Rxyz_inds   = sorted([int(f.split('.xyz')[0]) for dp, dn, filenames in os.walk(conf_folder[0]) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])
            for ind in Rxyz_inds:
                E,RG = xyz_parse(conf_folder[0]+'/{}.xyz'.format(ind))
                xyz_write(ini_folder+'/{}.xyz'.format(ind), E, RG)
                os.system('cat {} >> {}/{}.xyz'.format(product_xyz,ini_folder,ind))

            Pxyz_inds   = sorted([int(f.split('.xyz')[0]) for dp, dn, filenames in os.walk(conf_folder[1]) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])
            for ind in Pxyz_inds:
                E,PG = xyz_parse(conf_folder[1]+'/{}.xyz'.format(ind))
                xyz_write(ini_folder+'/{}.xyz'.format(ind+len(Rxyz_inds)), E, PG, comment = 'input_type: backward')
                os.system('cat {} >> {}/{}.xyz'.format(reactant_xyz,ini_folder,ind+len(Rxyz_inds)))

        conf_pair  += [(ini_folder,name)]

    # Submit conf_gen jobs 
    input_folder = work_folder+'/input_files_conf'
    submited = submit_select_new(conf_pair,input_folder,Njobs=int(N_job),Nmax=N_max,ff=ff,sched='slurm',queue='standby',product_opt=args.popt,rank_by_energy=args.rbe,remove_constraints=args.remove_constraints)

    substring="python {}/utilities/job_submit.py -f 'CONF_GEN.*.submit' -sched {}".format('/'.join(os.getcwd().split('/')[:-1]),'slurm')
    output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
    print("\t running {} CONF_GEN jobs...".format(int(len(output.split())/4)))
    
    monitor_jobs(output.split())

    print("Finishing CONF_GEN !!!")
    return

if __name__ == "__main__":
    main(sys.argv[1:])

