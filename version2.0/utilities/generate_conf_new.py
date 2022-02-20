import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
import numpy as np
import pickle

from ase import io
from ase.build import minimize_rotation_and_translation

# Load modules in same folder        
from taffi_functions import *
from utility import check_bimolecule,return_RMSD,return_indicator
from xtb_functions import xtb_energy,xtb_geo_opt

def main(argv):

    parser = argparse.ArgumentParser(description='Use to generate a set of conformation of reactant-product alignments')

    #optional arguments                                             
    parser.add_argument('-i', dest='inputF',
                        help = 'The program expects a folder of reactant xyz files')

    parser.add_argument('-o', dest='outputname',
                        help = 'An outputname is needed to name input files, the file will be named as outputname_i.xyz')

    parser.add_argument('-oF', dest='output_folder',
                        help = 'A folder to store input xyz files')

    parser.add_argument('-N', default = 10, dest='N_max',
                        help = 'Maximun number of conformations (default:10)')

    parser.add_argument('-ff', default='uff', dest='force_fields',
                        help = 'Force fields applied to derive product geometry')

    parser.add_argument('--product_opt', dest='popt', default=False, action='store_const',const=True,
                        help = 'When this flag is on, optimize the product at xTB level')

    parser.add_argument('--rank_by_energy', dest='rbe', default=False, action='store_const',const=True,
                        help = 'When this flag is on, the conformation will be ranked by xTB energy rather than RF probability (only valids for conformation rich cases)')

    parser.add_argument('--remove_constraints', dest='remove_constraints', default=False, action='store_const',const=True,
                        help = 'When this is flag is on, no hard constraint for indicators will be applied')
    
    # parse configuration and apply main function
    args       = parser.parse_args()
    xyz_inds   = sorted([int(f.split('.xyz')[0]) for dp, dn, filenames in os.walk(args.inputF) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])
    ff         = args.force_fields
    if args.N_max is None:
        N_max=10
    else:
        N_max = int(args.N_max)

    if os.path.isdir(args.output_folder) is False: os.mkdir(args.output_folder)

    # iterate over each xyz file to generate product alignments
    ind_list,back_up,pass_obj_values = [],[],[]
    print("Working on {}".format(args.inputF))

    # if number of conformers larger than 3 times N_max, it will be a conformer rich case
    if len(xyz_inds) > 3 * N_max:

        # load in random forest models
        model = pickle.load(open('rich_model.sav', 'rb'))
        count = 1

        # iterate xyz files
        for xyz_ind in xyz_inds:
        
            xyz   = '{}/{}.xyz'.format(args.inputF,xyz_ind)
            
            # derive reactant and product adj_mat
            E,RG,PG,Radj_mat,Padj_mat,input_type = parse_input(xyz)
            PG = opt_geo(RG,Padj_mat,E,ff=ff,step=500,filename=args.outputname)
            tmp_reactant = "{}/{}_reactant.xyz".format(os.getcwd(),args.outputname)
            tmp_product  = "{}/{}_product.xyz".format(os.getcwd(),args.outputname)
            xyz_write(tmp_reactant,E,RG)
            xyz_write(tmp_product,E,PG)

            # apply xtb level geo-opt
            if args.popt:
                Energy, opted_geo, finish=xtb_geo_opt(tmp_product,namespace=args.outputname,workdir=('/'.join(args.output_folder.split('/')[:-1])+'/xTB-folder'),level='normal',output_xyz=tmp_product,cleanup=False)
                if not finish:
                    os.remove(tmp_product)
                    continue
                _,PG = xyz_parse(opted_geo)

                # Check whether geo-opt changes the geometry. If so, geo-opt fails
                if sum(sum(abs(Padj_mat-Table_generator(E,PG)))) != 0: 
                    os.remove(tmp_product)
                    continue

            # Directly compute the indicators 
            indicators = return_indicator(E,RG,PG,namespace=args.outputname)
            #print(xyz,model.predict_proba([indicators])[0][1])
            if args.remove_constraints or (indicators[0] < 1.80 and indicators[1] < 4.8 and indicators[3] > 0.03 and indicators[5] > 0.28):
                
                # apply ase minimize_rotation_and_translation to optimize the reaction pathway 
                reactant= io.read(tmp_reactant)  
                product = io.read(tmp_product)
                minimize_rotation_and_translation(reactant,product) 
                io.write(tmp_product,product) 
                _,PG_opt = xyz_parse(tmp_product)
                indicators_opt = return_indicator(E,RG,PG_opt,namespace=args.outputname)
                
                # if applying ase minimize_rotation_and_translation will increase the intended probability, use the rotated geometry
                if model.predict_proba([indicators])[0][1] < model.predict_proba([indicators_opt])[0][1]:
                    indicators = indicators_opt
                    PG = PG_opt

                # check whether the channel is classified as intended and check uniqueness 
                if model.predict([indicators])[0] == 1 and check_duplicate(indicators,ind_list,thresh=0.025):

                    ind_list.append(indicators)                    
                    
                    if args.rbe:

                        # generate input file
                        tmpxyz = '{}_product.xyz'.format(args.outputname)
                        xyz_write(tmpxyz,E,PG)
                        if input_type == 'forward':
                            command_line = "cp {} {}/{}_{}.xyz;cat {} >> {}/{}_{}.xyz;rm {} {}".format(tmp_reactant,args.output_folder,args.outputname,count,tmpxyz,args.output_folder,args.outputname,count,tmpxyz,tmp_reactant)
                        else:
                            command_line = "cp {} {}/{}_{}.xyz;cat {} >> {}/{}_{}.xyz;rm {} {}".format(tmpxyz,args.output_folder,args.outputname,count,tmp_reactant,args.output_folder,args.outputname,count,tmpxyz,tmp_reactant)
                            
                        os.system(command_line)
                        count += 1
                    
                        # if reach N_max, break the loop
                        if count > N_max: 
                            if os.path.isfile(tmp_product): 
                                os.remove(tmp_product)
                            return
                        
                    else:
                        pass_obj_values.append((model.predict_proba([indicators])[0][0],RG,PG,input_type))
                        

                elif model.predict_proba([indicators])[0][1] > 0.4 and check_duplicate(indicators,ind_list,thresh=0.025):

                    ind_list.append(indicators)                    
                    back_up.append((model.predict_proba([indicators])[0][0],RG,PG,input_type))
            
            # remove tmp file
            if os.path.isfile(tmp_reactant): os.remove(tmp_reactant)
            if os.path.isfile(tmp_product): os.remove(tmp_product)

        # rank by RF prob
        if not args.rbe:
            pass_obj_values = sorted(pass_obj_values, key=lambda x: x[0])

            # generate conformers based on prob
            for count,item in enumerate(pass_obj_values):

                input_type = item[3]
                tmp_reactant = '{}_reactant.xyz'.format(args.outputname)
                tmp_product  = '{}_product.xyz'.format(args.outputname)
                xyz_write(tmp_reactant,E,item[1])
                xyz_write(tmp_product,E,item[2])

                if input_type == 'forward':
                    command_line = "cp {} {}/{}_{}.xyz;cat {} >> {}/{}_{}.xyz;rm {} {}".format(tmp_reactant,args.output_folder,args.outputname,count,tmp_product,args.output_folder,args.outputname,count,tmp_reactant,tmp_product)
                else:
                    command_line = "cp {} {}/{}_{}.xyz;cat {} >> {}/{}_{}.xyz;rm {} {}".format(tmp_product,args.output_folder,args.outputname,count,tmp_reactant,args.output_folder,args.outputname,count,tmp_reactant,tmp_product)

                os.system(command_line)
                if count+1 >= N_max: return

        # In the end, if so few conformations are generated, select at most Nmax / 2 backup conformations for instead
        if count - 1 < N_max:

            N_need = min(len(back_up),N_max-count+1)

            # sort by prob
            back_up = sorted(back_up, key=lambda x: x[0])
            if N_need > 0:

                # generate conformers based on prob
                for count_i in range(N_need):

                    item = back_up[count_i]
                    input_type = item[3]
                    tmp_reactant = '{}_reactant.xyz'.format(args.outputname)
                    tmp_product  = '{}_product.xyz'.format(args.outputname)
                    xyz_write(tmp_reactant,E,item[1])
                    xyz_write(tmp_product,E,item[2])

                    if input_type == 'forward':
                        command_line = "cp {} {}/{}_{}.xyz;cat {} >> {}/{}_{}.xyz;rm {} {}".format(tmp_reactant,args.output_folder,args.outputname,count,tmp_product,args.output_folder,args.outputname,count,tmp_reactant,tmp_product)
                    else:
                        command_line = "cp {} {}/{}_{}.xyz;cat {} >> {}/{}_{}.xyz;rm {} {}".format(tmp_product,args.output_folder,args.outputname,count,tmp_reactant,args.output_folder,args.outputname,count,tmp_reactant,tmp_product)

                    os.system(command_line)

    # if number of conformers less or equal to 2N_max, it will be a conformer poor case
    else:
        # load in random forest models
        model = pickle.load(open('poor_model.sav', 'rb'))
        obj_values = []

        # iterate xyz files
        for xyz_ind in xyz_inds:
        
            xyz   = '{}/{}.xyz'.format(args.inputF,xyz_ind)
            
            # derive reactant and product adj_mat
            E,RG,PG,Radj_mat,Padj_mat,input_type = parse_input(xyz)
            PG = opt_geo(RG,Padj_mat,E,ff=ff,step=500,filename=args.outputname)
            tmp_reactant = "{}/{}_reactant.xyz".format(os.getcwd(),args.outputname)
            tmp_product  = "{}/{}_product.xyz".format(os.getcwd(),args.outputname)
            xyz_write(tmp_reactant,E,RG)
            xyz_write(tmp_product,E,PG)

            # apply xtb level geo-opt
            if args.popt:
                Energy, opted_geo, finish=xtb_geo_opt(tmp_product,namespace=args.outputname,workdir=('/'.join(args.output_folder.split('/')[:-1])+'/xTB-folder'),level='normal',output_xyz=tmp_product,cleanup=False)
                if not finish: 
                    os.remove(tmp_product)
                    continue
                _,PG = xyz_parse(opted_geo)

                # Check whether geo-opt changes the geometry. If so, geo-opt fails
                if sum(sum(abs(Padj_mat-Table_generator(E,PG)))) != 0: 
                    os.remove(tmp_product)
                    continue

            # Directly compute the indicators 
            indicators = return_indicator(E,RG,PG,namespace=args.outputname)
            if args.remove_constraints or (indicators[0] < 1.80 and indicators[1] < 4.8 and indicators[3] > 0.03 and indicators[5] > 0.28):

                # apply ase minimize_rotation_and_translation to optimize the reaction pathway 
                reactant= io.read(tmp_reactant)  
                product = io.read(tmp_product)
                minimize_rotation_and_translation(reactant,product) 
                io.write(tmp_product,product) 
                _,PG_opt = xyz_parse(tmp_product)
                os.remove(tmp_product)
                indicators_opt = return_indicator(E,RG,PG_opt,namespace=args.outputname)

                # if applying ase minimize_rotation_and_translation will increase the intended probability, use the rotated geometry
                if model.predict_proba([indicators])[0][1] < model.predict_proba([indicators_opt])[0][1]:
                    indicators = indicators_opt
                    PG = PG_opt

                # calculate prob
                prob = model.predict_proba([indicators])[0][1]
                if check_duplicate(indicators,ind_list,thresh=0.01):
                    ind_list.append(indicators)
                    if prob > 0.4: pass_obj_values.append((1-prob,RG,PG,input_type))
                    else: obj_values.append((1-prob,RG,PG,input_type))

            # remove tmp file
            if os.path.isfile(tmp_product): os.remove(tmp_product)
            if os.path.isfile(tmp_reactant): os.remove(tmp_reactant)

        # sort by RF prob
        obj_values = sorted(obj_values, key=lambda x: x[0])
        pass_obj_values = sorted(pass_obj_values, key=lambda x: x[0])
        count = -1
            
        # generate conformers based on prob
        for count,item in enumerate(pass_obj_values):

            input_type = item[3]
            tmp_reactant = '{}_reactant.xyz'.format(args.outputname)
            tmp_product  = '{}_product.xyz'.format(args.outputname)
            xyz_write(tmp_reactant,E,item[1])
            xyz_write(tmp_product,E,item[2])

            if input_type == 'forward':
                command_line = "cp {} {}/{}_{}.xyz;cat {} >> {}/{}_{}.xyz;rm {} {}".format(tmp_reactant,args.output_folder,args.outputname,count,tmp_product,args.output_folder,args.outputname,count,tmp_reactant,tmp_product)
            else:
                command_line = "cp {} {}/{}_{}.xyz;cat {} >> {}/{}_{}.xyz;rm {} {}".format(tmp_product,args.output_folder,args.outputname,count,tmp_reactant,args.output_folder,args.outputname,count,tmp_reactant,tmp_product)
            os.system(command_line)

            if count+1 >= N_max: return

        # add a joint-opt alignment if too few alignments pass the criteria
        if len(pass_obj_values) < N_max - 2 and len(obj_values) > 0:

            RG,PG = obj_values[0][1],obj_values[0][2]
            tmp_reactant = '{}_reactant.xyz'.format(args.outputname)
            tmp_product  = '{}_product.xyz'.format(args.outputname)
            xyz_write(tmp_product,E,PG)
            RG = opt_geo(PG,Radj_mat,E,ff=ff,step=500,filename=args.outputname)

            if input_type == 'forward':
                xyz_write('{}/{}_{}.xyz'.format(args.output_folder,args.outputname,count+2),E,RG)
                command_line = "cat {} >> {}/{}_{}.xyz;rm {}".format(tmp_product,args.output_folder,args.outputname,count+2,tmp_product)
            else:
                xyz_write(tmp_reactant,E,RG)
                command_line = "cp {} {}/{}_{}.xyz;cat {} >> {}/{}_{}.xyz;rm {} {}".format(tmp_product,args.output_folder,args.outputname,count+2,tmp_reactant,args.output_folder,args.outputname,count+2,tmp_reactant,tmp_product)

            os.system(command_line)

    return

# Function to parse input files
def parse_input(input_xyz):

    name = input_xyz.split('/')[-1].split('xyz')[0]
    # also use the last folder to name the xyz file
    name = '{}_{}'.format(input_xyz.split('/')[-2],name)

    xyz  = ['','']
    count= 0
    input_type = 'forward'

    # read in pairs of xyz file
    with open(input_xyz,"r") as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if lc == 0: 
                N = int(fields[0])
                xyz[0] += lines
                continue
            if 'input_type:' in fields: input_type = fields[-1]
            if len(fields) == 1 and float(fields[0]) == float(N):
                count+=1

            xyz[count]+=lines

    with open('{}_reactant.xyz'.format(name),"w") as f:
        f.write(xyz[0])

    with open('{}_product.xyz'.format(name),"w") as f:
        f.write(xyz[1])

    # parse reactant info
    E,RG   = xyz_parse('{}_reactant.xyz'.format(name))

    # parse product info
    _,PG   = xyz_parse('{}_product.xyz'.format(name))
                
    try:
        os.remove('{}_reactant.xyz'.format(name))
        os.remove('{}_product.xyz'.format(name))
    except:
        pass

    # generate adj_mat if is needed
    Radj_mat = Table_generator(E, RG)
    Padj_mat = Table_generator(E, PG)
    return E,RG,PG,Radj_mat,Padj_mat,input_type

# Function to check duplicate indicators, return True if unique
def check_duplicate(i,total_i,thresh=0.05):

    if len(total_i) == 0: return True

    min_dis = min([np.linalg.norm(np.array(i)-np.array(j)) for j in total_i])

    # if rmsd > 0.1, this will be a unique conformation
    if min_dis > thresh: return True
    else: return False

if __name__ == "__main__":
    main(sys.argv[1:])
