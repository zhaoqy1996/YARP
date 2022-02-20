import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
import numpy as np
import pickle

from ase import io
from ase.build import minimize_rotation_and_translation

# Load modules in same folder        
from taffi_functions import *
from utility import check_bimolecule,return_RMSD
from xtb_functions import xtb_energy,xtb_geo_opt

def main(argv):

    parser = argparse.ArgumentParser(description='Use to generate a set of conformation of reactant-product alignments')

    #optional arguments                                             
    parser.add_argument('-i', dest='inputF',
                        help = 'The program expects a folder of reactant xyz files')

    parser.add_argument('-p', dest='product',
                        help = 'A product xyz file is needed')

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

    # derive reactant and product adj_mat
    E,RG = xyz_parse('{}/{}.xyz'.format(args.inputF,xyz_inds[0]))
    Radj_mat = Table_generator(E,RG)
    PE,PG = xyz_parse(args.product)
    Padj_mat = Table_generator(E,PG)
    if E != PE:
        print("Check reactant and product info, elements are not matching!")
        exit()

    # iterate over each xyz file to generate product alignments
    ind_list,back_up,pass_obj_values = [],[],[]

    # if number of conformers larger than 2 times N_max, it will be a conformer rich case
    if len(xyz_inds) > 3 * N_max:

        # load in random forest models
        model = pickle.load(open('rich_model.sav', 'rb'))
        count = 1

        # iterate xyz files
        for xyz_ind in xyz_inds:
        
            xyz   = '{}/{}.xyz'.format(args.inputF,xyz_ind)
            _,RGi = xyz_parse(xyz)
            PGi   = opt_geo(RGi,Padj_mat,E,ff=ff,step=1000,filename=args.outputname)
            tmp_product = "{}/{}_product.xyz".format(os.getcwd(),args.outputname)
            xyz_write(tmp_product,E,PGi)

            # apply xtb level geo-opt
            if args.popt:
                Energy, opted_geo, finish=xtb_geo_opt(tmp_product,namespace=args.outputname,workdir=('/'.join(args.output_folder.split('/')[:-1])+'/xTB-folder'),level='normal',output_xyz=tmp_product,cleanup=False)
                if not finish:
                    os.remove(tmp_product)
                    continue
                _,PGi = xyz_parse(opted_geo)

                # Check whether geo-opt changes the geometry. If so, geo-opt fails
                if sum(sum(abs(Padj_mat-Table_generator(E,PGi)))) != 0: 
                    os.remove(tmp_product)
                    continue

            # Directly compute the indicators 
            indicators = return_indicator(E,RGi,PGi,namespace=args.outputname)
            if args.remove_constraints or (indicators[0] < 1.80 and indicators[1] < 4.8 and indicators[3] > 0.03 and indicators[5] > 0.28):

                # apply ase minimize_rotation_and_translation to optimize the reaction pathway 
                reactant= io.read(xyz)  
                product = io.read(tmp_product)
                minimize_rotation_and_translation(reactant,product) 
                io.write(tmp_product,product) 
                _,PGi_opt = xyz_parse(tmp_product)
                indicators_opt = return_indicator(E,RGi,PGi_opt,namespace=args.outputname)
                
                # if applying ase minimize_rotation_and_translation will increase the intended probability, use the rotated geometry
                if model.predict_proba([indicators])[0][1] < model.predict_proba([indicators_opt])[0][1]:
                    indicators = indicators_opt
                    PGi = PGi_opt

                # check whether the channel is classified as intended and check uniqueness 
                if model.predict([indicators])[0] == 1 and check_duplicate(indicators,ind_list,thresh=0.025):

                    ind_list.append(indicators)                    
                    
                    if args.rbe:
                        # generate input file
                        tmpxyz = '{}_product.xyz'.format(args.outputname)
                        xyz_write(tmpxyz,E,PGi)
                        command_line = "cp {} {}/{}_{}.xyz;cat {} >> {}/{}_{}.xyz;rm {}".format(xyz,args.output_folder,args.outputname,count,tmpxyz,args.output_folder,args.outputname,count,tmpxyz)
                        os.system(command_line)
                        count += 1
                    
                        # if reach N_max, break the loop
                        if count > N_max: 
                            if os.path.isfile(tmp_product): 
                                os.remove(tmp_product)
                            return
                        
                    else:
                        pass_obj_values.append((model.predict_proba([indicators])[0][0],xyz,PGi))
                        

                elif model.predict_proba([indicators])[0][1] > 0.4 and check_duplicate(indicators,ind_list,thresh=0.025):

                    ind_list.append(indicators)                    
                    back_up.append((model.predict_proba([indicators])[0][0],xyz,PGi))
            
            # remove tmp file
            if os.path.isfile(tmp_product): os.remove(tmp_product)

        # rank by RF prob
        if not args.rbe:
            pass_obj_values = sorted(pass_obj_values, key=lambda x: x[0])

            # generate conformers based on prob
            for count,item in enumerate(pass_obj_values):

                tmpxyz = '{}_product.xyz'.format(args.outputname)
                xyz_write(tmpxyz,E,item[2])
                command_line = "cp {} {}/{}_{}.xyz;cat {} >> {}/{}_{}.xyz;rm {}".format(item[1],args.output_folder,args.outputname,count+1,tmpxyz,args.output_folder,args.outputname,count+1,tmpxyz)
                os.system(command_line)
                if count+1 >= N_max: return

        # In the end, if so few conformations are generated, select at most Nmax/2  backup conformations for instead
        if count - 1 < N_max:

            N_need = min(len(back_up),N_max-count+1)

            # sort by prob
            back_up = sorted(back_up, key=lambda x: x[0])
            if N_need > 0:

                # generate conformers based on prob
                for count_i in range(N_need):

                    item = back_up[count_i]
                    tmpxyz = '{}_product.xyz'.format(args.outputname)
                    xyz_write(tmpxyz,E,item[2])
                    command_line = "cp {} {}/{}_{}.xyz;cat {} >> {}/{}_{}.xyz;rm {}".format(item[1],args.output_folder,args.outputname,count+count_i,tmpxyz,args.output_folder,args.outputname,count+count_i,tmpxyz)
                    os.system(command_line)

    # if number of conformers less or equal to 2N_max, it will be a conformer poor case
    else:
        # load in random forest models
        model = pickle.load(open('poor_model.sav', 'rb'))
        obj_values,pass_obj_values = [],[]

        # iterate xyz files
        for xyz_ind in xyz_inds:
        
            xyz   = '{}/{}.xyz'.format(args.inputF,xyz_ind)
            _,RGi = xyz_parse(xyz)
            PGi   = opt_geo(RGi,Padj_mat,E,ff=ff,step=1000,filename=args.outputname)
            tmp_product = "{}/{}_product.xyz".format(os.getcwd(),args.outputname)
            xyz_write(tmp_product,E,PGi)

            # apply xtb level geo-opt
            if args.popt:
                Energy, opted_geo, finish=xtb_geo_opt(tmp_product,namespace=args.outputname,workdir=('/'.join(args.output_folder.split('/')[:-1])+'/xTB-folder'),level='normal',output_xyz=tmp_product,cleanup=False)
                if not finish: 
                    os.remove(tmp_product)
                    continue
                _,PGi = xyz_parse(opted_geo)

                # Check whether geo-opt changes the geometry. If so, geo-opt fails
                if sum(sum(abs(Padj_mat-Table_generator(E,PGi)))) != 0: 
                    os.remove(tmp_product)
                    continue

            # Directly compute the indicators 
            indicators = return_indicator(E,RGi,PGi,namespace=args.outputname)

            if args.remove_constraints or (indicators[0] < 1.80 and indicators[1] < 4.8 and indicators[3] > 0.03 and indicators[5] > 0.28):

                # apply ase minimize_rotation_and_translation to optimize the reaction pathway 
                reactant= io.read(xyz)  
                product = io.read(tmp_product)
                minimize_rotation_and_translation(reactant,product) 
                io.write(tmp_product,product) 
                _,PGi_opt = xyz_parse(tmp_product)
                os.remove(tmp_product)
                indicators_opt = return_indicator(E,RGi,PGi_opt,namespace=args.outputname)

                # if applying ase minimize_rotation_and_translation will increase the intended probability, use the rotated geometry
                if model.predict_proba([indicators])[0][1] < model.predict_proba([indicators_opt])[0][1]:
                    indicators = indicators_opt
                    PGi = PGi_opt

                # calculate prob
                prob = model.predict_proba([indicators])[0][1]
                if check_duplicate(indicators,ind_list,thresh=0.01):
                    ind_list.append(indicators)
                    #print(prob)
                    if prob > 0.4: pass_obj_values.append((1-prob,xyz,PGi))
                    else: obj_values.append((1-prob,xyz,PGi))

            # remove tmp file
            if os.path.isfile(tmp_product): os.remove(tmp_product)

        # sort by RF prob
        obj_values = sorted(obj_values, key=lambda x: x[0])
        pass_obj_values = sorted(pass_obj_values, key=lambda x: x[0])
        count = -1
            
        # generate conformers based on prob
        for count,item in enumerate(pass_obj_values):

            tmpxyz = '{}_product.xyz'.format(args.outputname)
            xyz_write(tmpxyz,E,item[2])
            command_line = "cp {} {}/{}_{}.xyz;cat {} >> {}/{}_{}.xyz;rm {}".format(item[1],args.output_folder,args.outputname,count+1,tmpxyz,args.output_folder,args.outputname,count+1,tmpxyz)
            os.system(command_line)
            if count+1 >= N_max: return

        # add a joint-opt alignment if too few alignments pass the criteria
        if len(pass_obj_values) < N_max-2 and len(obj_values) > 0:

            Rxyz,PGi = obj_values[0][1],obj_values[0][2]
            tmpxyz = '{}_product.xyz'.format(args.outputname)
            xyz_write(tmpxyz,E,PGi)
            new_RG = opt_geo(PGi,Radj_mat,E,ff=ff,step=500,filename=args.outputname)
            xyz_write('{}/{}_{}.xyz'.format(args.output_folder,args.outputname,count+2),E,new_RG)
            command_line = "cat {} >> {}/{}_{}.xyz;rm {}".format(tmpxyz,args.output_folder,args.outputname,count+2,tmpxyz)
            os.system(command_line)

    return

# function to find indicators for reactant-product alignments
# Input: 
#         E:   elements
#         RG:  reactant geometry 
#         RG:  reactant geometry 
# Output:
#     RMSD: mass-weighted RMSD between reactant and product, threshold < 1.6
#     max_dis:  maximum bond length change between non-H atoms, threshold < 4.0
#     min_cross_dis: shorted distance between atoms' path (non-H atoms) to original bonds, threshold > 0.6
#     path_cross: if all atoms involved in bond changes are non-H, path_cross refers to the distance between two bond changes, threshold > 0.6 
#     max_Hdis: maximum bond length change if contains H, threshold < 4.5 (* optional)
#     min_Hcross_dis: shorted distance between atoms' path (H atoms involves) to original bonds, threshold > 0.4 (* optional)
#     h = RMSD/1.6 + max_dis/4.0 + 0.6/min_cross_dis + 0.6/path_cross + 0.5 * max_Hdis/4.5 + 0.1/min_cross_dis
#
def return_indicator(E,RG,PG,namespace='node'):

    # calculate adj_mat
    Radj=Table_generator(E, RG)
    Padj=Table_generator(E, PG)
    
    # determine bond changes
    bond_break, bond_form=[], []
    del_adj = Padj - Radj
    for i in range(len(E)):
        for j in range(i+1, len(E)):
            if del_adj[i][j]==-1: bond_break+=[(i, j)]
            if del_adj[i][j]==1: bond_form+=[(i, j)]

    # identify hydrogen atoms, atoms involved in the reactions
    H_index=[i for i, e in enumerate(E) if e=='H']
    involve=list(set(list(sum(bond_break+bond_form, ()))))

    # create observed segments
    bond_seg={i:[] for i in bond_break+bond_form}
    for bond in bond_break:
        bond_seg[bond]=(PG[bond[1]]-PG[bond[0]], np.linalg.norm(PG[bond[1]]-PG[bond[0]]))
    for bond in bond_form:
        bond_seg[bond]=(RG[bond[1]]-RG[bond[0]], np.linalg.norm(RG[bond[1]]-RG[bond[0]]))

    # create bond list to check cross
    bond_dict={i: [] for i in bond_break+bond_form}
    for i in range(len(E)):
        for j in range(i+1, len(E)):
            for bond in bond_break:
                if Padj[i][j]>0 and i not in bond and j not in bond: bond_dict[bond]+=[(i, j)]
            for bond in bond_form:
                if Radj[i][j]>0 and i not in bond and j not in bond: bond_dict[bond]+=[(i, j)]

    # Compute indicator
    rmsd = return_RMSD(E,RG,PG,rotate=False,mass_weighted=True,namespace=namespace)
    Hbond_dis = np.array([i[1] for bond,i in bond_seg.items() if (bond[0] in H_index or bond[1] in H_index)])
    bond_dis  = np.array([i[1] for bond,i in bond_seg.items() if (bond[0] not in H_index and bond[1] not in H_index)])
    if len(Hbond_dis)>0: 
        max_Hdis=max(Hbond_dis)
    else: 
        max_Hdis=2.0
    if len(bond_dis)>0: 
        max_dis=max(bond_dis)
    else: 
        max_dis=2.0

    # Compute "cross" behaviour
    min_cross, min_Hcross=[], []
    for bond in bond_break:
        cross_dis=[]
        for ibond in bond_dict[bond]:
            _,_,dis=closestDistanceBetweenLines(PG[bond[0]], PG[bond[1]], PG[ibond[0]], PG[ibond[1]])
            cross_dis+=[dis]
        if len(cross_dis)>0: 
            min_dis=min(cross_dis)
        else: 
            min_dis=2.0

        if bond[0] in H_index or bond[1] in H_index: 
            min_Hcross+=[min_dis]
        else: 
            min_cross+=[min_dis]

    for bond in bond_form:
        cross_dis=[]
        for ibond in bond_dict[bond]:
            _,_,dis=closestDistanceBetweenLines(RG[bond[0]], RG[bond[1]], RG[ibond[0]], RG[ibond[1]])
            cross_dis+=[dis]
        if len(cross_dis) > 0: 
            min_dis=min(cross_dis)
        else: 
            min_dis=2.0
        if bond[0] in H_index or bond[1] in H_index: 
            min_Hcross+=[min_dis]
        else: 
            min_cross+=[min_dis]

    # Find the smallest bonds distance for each bond, if None, return 2.0
    if len(min_cross) > 0:
        min_cross_dis = min(min_cross)
    else:
        min_cross_dis = 2.0

    if len(min_Hcross) > 0:
        min_Hcross_dis = min(min_Hcross)
    else:
        min_Hcross_dis = 2.0

    # Find the cross distanc ebetween two bond changes
    if len([ind for ind in involve if ind in H_index]) ==0:

        if len(bond_break) == 2:
            _,_,dis = closestDistanceBetweenLines(PG[bond_break[0][0]],PG[bond_break[0][1]],PG[bond_break[1][0]],PG[bond_break[1][1]],clampAll=True)
        else:
            dis = 2.0
        path_cross = dis

        if len(bond_form) == 2:
            _,_,dis = closestDistanceBetweenLines(RG[bond_form[0][0]],RG[bond_form[0][1]],RG[bond_form[1][0]],RG[bond_form[1][1]],clampAll=True)
        else:
            dis = 2.0
        path_cross = min(dis,path_cross)

    else:
        path_cross = 2.0

    return [rmsd, max_dis, max_Hdis, min_cross_dis, min_Hcross_dis, path_cross]

# Function to calculate spatial distance between two segments
def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=True,clampA0=False,clampA1=False,clampB0=False,clampB1=False):

    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    '''

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
   
    _A = A / magA
    _B = B / magB
   
    cross = np.cross(_A, _B);
    denom = np.linalg.norm(cross)**2
   
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A,(b1-a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0,b0,np.linalg.norm(a0-b0)
                    return a0,b1,np.linalg.norm(a0-b1)


            # Is segment B after A?
        elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1)


        # Segments overlap, return distance between parallel segments
        return None,None,np.linalg.norm(((d0*_A)+a0)-b0)

    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0);
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom;
    t1 = detB/denom;

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA,pB,np.linalg.norm(pA-pB)

# objective function for indicators
def obj_fun(x):
    return x[0]/1.8+x[1]/4.0+0.6/x[3]+0.6/x[5]+x[2]/10.0+0.2/x[4]

# Function to check duplicate indicators, return True if unique
def check_duplicate(i,total_i,thresh=0.05):

    if len(total_i) == 0: return True

    min_dis = min([np.linalg.norm(np.array(i)-np.array(j)) for j in total_i])

    # if rmsd > 0.1, this will be a unique conformation
    if min_dis > thresh: return True
    else: return False

if __name__ == "__main__":
    main(sys.argv[1:])
