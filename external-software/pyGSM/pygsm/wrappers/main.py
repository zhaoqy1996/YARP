# standard library imports
import sys
import os
from os import path
import importlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#third party
import argparse
import numpy as np
import textwrap

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from utilities import *
from potential_energy_surfaces import PES
from potential_energy_surfaces import Avg_PES
from potential_energy_surfaces import Penalty_PES
from wrappers import Molecule
from optimizers import *
from growing_string_methods import *
from coordinate_systems import Topology,PrimitiveInternalCoordinates,DelocalizedInternalCoordinates,Distance,Angle,Dihedral,OutOfPlane,TranslationX,TranslationY,TranslationZ,RotationA,RotationB,RotationC



def main():
    parser = argparse.ArgumentParser(
        description="Reaction path transition state and photochemistry tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
                Example of use:
                --------------------------------
                gsm -mode DE_GSM -xyzfile yourfile.xyz -package QChem -lot_inp_file qstart -ID 1
                ''')
        )
    parser.add_argument('-xyzfile', help='XYZ file containing reactant and, if DE-GSM, product.',  required=True)
    parser.add_argument('-isomers', help='driving coordinate file', type=str, required=False)
    parser.add_argument('-mode', default="DE_GSM",help='GSM Type (default: %(default)s)',choices=["DE_GSM","SE_GSM","SE_Cross"], type=str, required=True)
    parser.add_argument('-package',default="QChem",type=str,help="Electronic structure theory package (default: %(default)s)",choices=["QChem","Orca","Molpro","PyTC","TeraChemCloud","OpenMM","DFTB","TeraChem","xTB"])
    parser.add_argument('-lot_inp_file',type=str,default='qstart', help='external file to specify calculation e.g. qstart,gstart,etc. Highly package specific.',required=True)
    parser.add_argument('-ID',default=0, type=int,help='string identification number (default: %(default)s)',required=False)
    parser.add_argument('-num_nodes',type=int,help='number of nodes for string (defaults: 9 DE-GSM, 20 SE-GSM)',required=False)
    parser.add_argument('-pes_type',type=str,default='PES',help='Potential energy surface (default: %(default)s)',choices=['PES','Avg_PES','Penalty_PES'])
    parser.add_argument('-adiabatic_index',nargs="*",type=int,default=[0],help='Adiabatic index (default: %(default)s)',required=False)
    parser.add_argument('-multiplicity',nargs="*",type=int,default=[1],help='Multiplicity (default: %(default)s)')
    parser.add_argument('-FORCE',type=list,default=None,help='Spring force between atoms in AU,e.g. [(1,2,0.1214)]. Negative is tensile, positive is compresive')
    parser.add_argument('-optimizer',type=str,default='eigenvector_follow',help='The optimizer object. (default: %(default)s Recommend LBFGS for large molecules >1000 atoms)',required=False)
    parser.add_argument('-opt_print_level',type=int,default=1,help='Printout for optimization. 2 prints everything in opt.',required=False)
    parser.add_argument('-gsm_print_level',type=int,default=1,help='Printout for gsm. 1 prints ?',required=False)
    parser.add_argument('-linesearch',type=str,default='NoLineSearch',help='default: %(default)s',choices=['NoLineSearch','backtrack'])
    parser.add_argument('-coordinate_type',type=str,default='TRIC',help='Coordinate system (default %(default)s)',choices=['TRIC','DLC','HDLC'])
    parser.add_argument('-ADD_NODE_TOL',type=float,default=0.01,help='Convergence tolerance for adding new node (default: %(default)s)',required=False)
    parser.add_argument('-DQMAG_MAX',type=float,default=0.8,help='Maximum step size in single-ended mode (default: %(default)s)',required=False)
    parser.add_argument('-BDIST_RATIO',type=float,default=0.5,help='Reaction completion convergence in SE modes (default: %(default)s)')
    parser.add_argument('-CONV_TOL',type=float,default=0.0005,help='Convergence tolerance for optimizing nodes (default: %(default)s)',required=False)
    parser.add_argument('-growth_direction',type=int,default=0,help='Direction adding new nodes (default: %(default)s)',choices=[0,1])
    parser.add_argument('-reactant_geom_fixed',action='store_true',help='Fix reactant geometry i.e. do not pre-optimize')
    parser.add_argument('-product_geom_fixed',action='store_true',help='Fix product geometry i.e. do not pre-optimize')
    parser.add_argument('-nproc',type=int,default=1,help='Processors for calculation. Python will detect OMP_NUM_THREADS, only use this if you want to force the number of processors')
    parser.add_argument('-charge',type=int,default=0,help='Total system charge (default: %(default)s)')
    parser.add_argument('-max_gsm_iters',type=int,default=100,help='The maximum number of GSM cycles (default: %(default)s)')
    parser.add_argument('-max_opt_steps',type=int,help='The maximum number of node optimizations per GSM cycle (defaults: 3 DE-GSM, 20 SE-GSM)')
    parser.add_argument('-only_climb',action='store_true',help="Only use climbing image to optimize TS")
    parser.add_argument('-no_climb',action='store_true',help="Don't climb to the TS")
    parser.add_argument('-optimize_mesx',action='store_true',help='optimize to the MESX')
    parser.add_argument('-optimize_meci',action='store_true',help='optimize to the MECI')
    parser.add_argument('-restart_file',help='restart file',type=str)
    parser.add_argument('-use_multiprocessing',action='store_true',help="Use python multiprocessing to parallelize jobs on a single compute node. Set OMP_NUM_THREADS, ncpus accordingly.")
    parser.add_argument('-dont_analyze_ICs',action='store_false',help="Don't post-print the internal coordinates primitives and values") #defaults to true
    parser.add_argument('-hybrid_coord_idx_file',type=str,help="A filename containing a list of  indices to use in hybrid coordinates. 0-Based indexed")
    parser.add_argument('-frozen_coord_idx_file',type=str,help="A filename containing a list of  indices to be frozen. 0-Based indexed")
    parser.add_argument('-conv_Ediff',default=100.,type=float,help='')
    parser.add_argument('-conv_dE',default=1.,type=float,help='')
    parser.add_argument('-conv_gmax',default=100.,type=float,help='')
    parser.add_argument('-DMAX',default=1.,type=float,help='')
    parser.add_argument('-sigma',default=1.,type=float,help='The strength of the difference energy penalty in Penalty_PES')
    parser.add_argument('-prim_idx_file',type=str,help="A filename containing a list of indices to define fragments. 0-Based indexed")


    args = parser.parse_args()

    print_msg()

    if args.nproc>1:
        force_num_procs=True
        print("forcing number of processors to be {}!!!".format(args.nproc))
    else:
        force_num_procs=False
    if force_num_procs:
        nproc = args.nproc
    else:
        #nproc = get_nproc()
        try:
            nproc = int(os.environ['OMP_NUM_THREADS'])
        except: 
            nproc = 1
        print(" Using {} processors".format(nproc))

    #print(args.states)
    #if args.states is not None:
    #    states = [tuple(int(s) for s in my_tup.strip("()").split(",")) for my_tup in args.states ]
    #    print(states)
    #else:
    #    states=None

    inpfileq = {
               # LOT
              'lot_inp_file': args.lot_inp_file,
              'xyzfile' : args.xyzfile,
              'EST_Package': args.package,
              'reactant_geom_fixed' : args.reactant_geom_fixed,
              'nproc': args.nproc,
              'states': None,
              
              #PES
              'PES_type': args.pes_type,
              'adiabatic_index': args.adiabatic_index,
              'multiplicity': args.multiplicity,
              'FORCE': args.FORCE,

              #optimizer
              'optimizer' : args.optimizer,
              'opt_print_level' : args.opt_print_level,
              'linesearch' : args.linesearch,
              'DMAX'    :   args.DMAX,

              #molecule
              'coordinate_type' : args.coordinate_type,
              'hybrid_coord_idx_file' : args.hybrid_coord_idx_file,
              'frozen_coord_idx_file' : args.frozen_coord_idx_file,
              'prim_idx_file' : args.prim_idx_file,

              # GSM
              'gsm_type': args.mode, # SE_GSM, SE_Cross
              'num_nodes' : args.num_nodes,
              'isomers_file': args.isomers,
              'ADD_NODE_TOL': args.ADD_NODE_TOL,
              'CONV_TOL': args.CONV_TOL,
              'conv_Ediff': args.conv_Ediff,
              'conv_dE': args.conv_dE,
              'conv_gmax': args.conv_gmax,
              'BDIST_RATIO':args.BDIST_RATIO,
              'DQMAG_MAX': args.DQMAG_MAX,
              'growth_direction': args.growth_direction,
              'ID':args.ID,
              'product_geom_fixed' : args.product_geom_fixed,
              'gsm_print_level' : args.gsm_print_level,
              'max_gsm_iters' : args.max_gsm_iters,
              'max_opt_steps' : args.max_opt_steps,
              'use_multiprocessing': args.use_multiprocessing,
              'sigma'   :   args.sigma,
              }

    nifty.printcool_dictionary(inpfileq,title='Parsed GSM Keys : Values')

    #LOT
    nifty.printcool("Build the {} level of theory (LOT) object".format(inpfileq['EST_Package']))
    est_package=importlib.import_module("level_of_theories."+inpfileq['EST_Package'].lower())
    lot_class = getattr(est_package,inpfileq['EST_Package'])
    geoms = manage_xyz.read_xyzs(inpfileq['xyzfile'])
    inpfileq['states'] = [ (int(m),int(s)) for m,s in zip(args.multiplicity,args.adiabatic_index)]
    if inpfileq['PES_type']!="PES":
        assert len(args.adiabatic_index)>1, "need more states"
        assert len(args.multiplicity)>1, "need more spins"
    if args.charge != 0:
        print("Warning: charge is not implemented for all level of theories. Make sure this is correct for your package.")
    if inpfileq['num_nodes'] is None:
        if inpfileq['gsm_type']=="DE_GSM":
            inpfileq['num_nodes']=9
        else:
            inpfileq['num_nodes']=20
    do_coupling = True if inpfileq['PES_type']=="Avg_PES" else False
    coupling_states = [ (int(m),int(s)) for m,s in zip(args.multiplicity,args.adiabatic_index)] if inpfileq['PES_type']=="Avg_PES" else []
    
    lot = lot_class.from_options(
            ID = inpfileq['ID'],
            lot_inp_file=inpfileq['lot_inp_file'],
            states=inpfileq['states'],
            gradient_states=inpfileq['states'],
            coupling_states=coupling_states,
            geom=geoms[0],
            nproc=nproc,
            charge=args.charge,
            do_coupling=do_coupling,
            )

    #PES
    if inpfileq['gsm_type'] == "SE_Cross":
        if inpfileq['PES_type']!="Penalty_PES":
            print(" setting PES type to Penalty")
            inpfileq['PES_type']="Penalty_PES"
    if args.optimize_mesx or args.optimize_meci  or inpfileq['gsm_type']=="SE_Cross":
        assert inpfileq['PES_type'] == "Penalty_PES", "Need penalty pes for optimizing MESX/MECI"
    nifty.printcool("Building the {} objects".format(inpfileq['PES_type']))
    pes_class = getattr(sys.modules[__name__], inpfileq['PES_type'])
    if inpfileq['PES_type']=='PES':
        pes = pes_class.from_options(
                lot=lot,
                ad_idx=inpfileq['adiabatic_index'][0],
                multiplicity=inpfileq['multiplicity'][0],
                FORCE=inpfileq['FORCE']
                )
    else:
        pes1 = PES.from_options(
                lot=lot,multiplicity=inpfileq['states'][0][0],
                ad_idx=inpfileq['states'][0][1],
                FORCE=inpfileq['FORCE']
                )
        pes2 = PES.from_options(
                lot=lot,
                multiplicity=inpfileq['states'][1][0],
                ad_idx=inpfileq['states'][1][1],
                FORCE=inpfileq['FORCE']
                )
        if inpfileq['PES_type']=="Avg_PES":
            pes = pes_class(PES1=pes1,PES2=pes2,lot=lot)
        elif inpfileq['PES_type']=="Penalty_PES": 
            pes = pes_class(PES1=pes1,PES2=pes2,lot=lot,sigma=inpfileq['sigma'])
        else:
            raise NotImplementedError

    # Molecule
    nifty.printcool("Building the reactant object with {}".format(inpfileq['coordinate_type']))
    Form_Hessian = True if inpfileq['optimizer']=='eigenvector_follow' else False
    #form_primitives = True if inpfileq['gsm_type']!='DE_GSM' else False

    # hybrid coordinates
    if inpfileq['hybrid_coord_idx_file'] is not None:
        nifty.printcool(" Using Hybrid COORDINATES :)")
        assert inpfileq['coordinate_type']=="TRIC", "hybrid indices won't work (currently) with other coordinate systems"
        with open(inpfileq['hybrid_coord_idx_file']) as f:
            hybrid_indices = f.read().splitlines()
        hybrid_indices = [int(x) for x in hybrid_indices]
    else:
        hybrid_indices = None
    if inpfileq['frozen_coord_idx_file'] is not None:
        with open(inpfileq['frozen_coord_idx_file']) as f:
            frozen_indices = f.read().splitlines()
        frozen_indices = [int(x) for x in frozen_indices]
    else:
        frozen_indices = None


    # prim internal coordinates
    # The start and stop indexes of the primitive internal region, this defines the "fragments" so no large molecule is built        
    if inpfileq['prim_idx_file'] is not None:
        nifty.printcool(" Defining primitive internal region :)")
        assert inpfileq['coordinate_type']=="TRIC", "won't work (currently) with other coordinate systems"
        prim_indices = np.loadtxt(inpfileq['prim_idx_file'])
        prim_indices = [ (int(prim_indices[i,0]), int(prim_indices[i,1])-1) for i in range(len(prim_indices))]
        print(prim_indices)
        #with open(inpfileq['prim_idx_file']) as f:
        #    prim_indices = f.read().splitlines()
        #prim_indices = [int(x) for x in prim_indices]
    else:
        prim_indices = None


    # Build the topology
    nifty.printcool("Building the topology")
    atom_symbols  = manage_xyz.get_atoms(geoms[0])
    ELEMENT_TABLE = elements.ElementData()
    atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atom_symbols]
    xyz1 = manage_xyz.xyz_to_np(geoms[0])
    top1 = Topology.build_topology(
            xyz1,
            atoms,
            hybrid_indices=hybrid_indices,
            prim_idx_start_stop=prim_indices,
            )

    if inpfileq['gsm_type'] == 'DE_GSM':
        # find union bonds
        xyz2 = manage_xyz.xyz_to_np(geoms[-1])
        top2 = Topology.build_topology(
                xyz2,
                atoms,
                hybrid_indices=hybrid_indices,
                prim_idx_start_stop=prim_indices,
                )

        # Add bonds to top1 that are present in top2
        # It's not clear if we should form the topology so the bonds
        # are the same since this might affect the Primitives of the xyz1 (slightly)
        # Later we stil need to form the union of bonds, angles and torsions
        # However, I think this is important, the way its formulated, for identifiyin 
        # the number of fragments and blocks, which is used in hybrid TRIC. 
        for bond in top2.edges():
            if bond in top1.edges:
                pass
            elif (bond[1],bond[0]) in top1.edges():
                pass
            else:
                print(" Adding bond {} to top1".format(bond))
                if bond[0]>bond[1]:
                    top1.add_edge(bond[0],bond[1])
                else:
                    top1.add_edge(bond[1],bond[0])
    elif inpfileq['gsm_type'] == 'SE_GSM' or inpfileq['gsm_type']=='SE_Cross':
        driving_coordinates = read_isomers_file(inpfileq['isomers_file'])

        driving_coord_prims=[]
        for dc in driving_coordinates:
            prim = get_driving_coord_prim(dc)
            if prim is not None:
                driving_coord_prims.append(prim)

        for prim in driving_coord_prims:
            if type(prim)==Distance:
                bond = (prim.atoms[0],prim.atoms[1])
                if bond in top1.edges:
                    pass
                elif (bond[1],bond[0]) in top1.edges():
                    pass
                else:
                    print(" Adding bond {} to top1".format(bond))
                    top1.add_edge(bond[0],bond[1])

    nifty.printcool("Building Primitive Internal Coordinates")
    connect=False
    addtr = False
    addcart=False
    if inpfileq['coordinate_type']=="DLC":
        connect=True
    elif inpfileq['coordinate_type']=="TRIC":
        addtr=True
    elif inpfileq['coordinate_type']=="HDLC":
        addcart=True
    p1 = PrimitiveInternalCoordinates.from_options(
            xyz=xyz1,
            atoms=atoms,
            connect=connect,
            addtr=addtr,
            addcart=addcart,
            topology=top1,
            frozen_atoms=frozen_indices,
            )

    if inpfileq['gsm_type'] == 'DE_GSM':
        nifty.printcool("Building Primitive Internal Coordinates 2")
        p2 = PrimitiveInternalCoordinates.from_options(
                xyz=xyz2,
                atoms=atoms,
                addtr = addtr,
                addcart=addcart,
                connect=connect,
                topology=top1,  # Use the topology of 1 because we fixed it above
                ) 
        nifty.printcool("Forming Union of Primitives")
        # Form the union of primitives
        p1.add_union_primitives(p2)
        print("check {}".format(len(p1.Internals)))

    elif inpfileq['gsm_type'] == 'SE_GSM' or inpfileq['gsm_type']=='SE_Cross':
        for dc in driving_coord_prims:
            if type(dc)!=Distance: # Already handled in topology
                if dc not in p1.Internals:
                    print("Adding driving coord prim {} to Internals".format(dc))
                    p1.append_prim_to_block(dc)

    nifty.printcool("Building Delocalized Internal Coordinates")
    coord_obj1 = DelocalizedInternalCoordinates.from_options(
            xyz=xyz1,
            atoms=atoms,
            addtr = addtr,
            addcart=addcart,
            connect=connect,
            primitives=p1,
            frozen_atoms=frozen_indices,
            ) 

    if inpfileq['gsm_type'] == 'DE_GSM':
        # TMP 
        pass
        #coord_obj2 = DelocalizedInternalCoordinates.from_options(
        #        xyz=xyz2,
        #        atoms=atoms,
        #        addtr = True,
        #        addcart=addcart,
        #        connect=connect,
        #        primitives=p1,
        #        frozen_atoms=frozen_indices,
        #        ) 


    nifty.printcool("Building the reactant")
    reactant = Molecule.from_options(
            geom=geoms[0],
            PES=pes,
            coord_obj = coord_obj1,
            #coordinate_type=inpfileq['coordinate_type'],
            Form_Hessian=Form_Hessian,
            #top_settings = {
            #    'form_primitives': form_primitives,
            #    'hybrid_indices' : hybrid_indices,
            #    },
            )

    if inpfileq['gsm_type']=='DE_GSM':
        nifty.printcool("Building the product object")
        product = Molecule.from_options(
                geom=geoms[1],
                PES=pes,
                #coord_obj = coord_obj2,
                coord_obj = coord_obj1,
                #coordinate_type=inpfileq['coordinate_type'],
                Form_Hessian=Form_Hessian,
                node_id=inpfileq['num_nodes']-1,
                #top_settings = {
                #    'form_primitives': form_primitives,
                #    'hybrid_indices' : hybrid_indices,
                #    },
                copy_wavefunction= False,  
                )
    
    # optimizer
    nifty.printcool("Building the Optimizer object")
    update_hess_in_bg = True
    if args.only_climb or inpfileq['optimizer']=="lbfgs":
        update_hess_in_bg = False
    opt_class = getattr(sys.modules[__name__], inpfileq['optimizer'])
    optimizer = opt_class.from_options(
            print_level=inpfileq['opt_print_level'],
            Linesearch=inpfileq['linesearch'],
            update_hess_in_bg = update_hess_in_bg,
            conv_Ediff = inpfileq['conv_Ediff'],
            conv_dE = inpfileq['conv_dE'],
            conv_gmax = inpfileq['conv_gmax'],
            DMAX = inpfileq['DMAX'],
            opt_climb = True if args.only_climb else False,
            )

    # GSM
    nifty.printcool("Building the GSM object")
    gsm_class = getattr(sys.modules[__name__], inpfileq['gsm_type'])
    
    if inpfileq['gsm_type']=="DE_GSM":
        gsm = gsm_class.from_options(
                reactant=reactant,
                product=product,
                nnodes=inpfileq['num_nodes'],
                CONV_TOL=inpfileq['CONV_TOL'],
                ADD_NODE_TOL=inpfileq['ADD_NODE_TOL'],
                growth_direction=inpfileq['growth_direction'],
                product_geom_fixed=inpfileq['product_geom_fixed'],
                optimizer=optimizer,
                ID=inpfileq['ID'],
                print_level=inpfileq['gsm_print_level'],
                use_multiprocessing=inpfileq['use_multiprocessing'],
                )
    
    else:
        gsm = gsm_class.from_options(
                reactant=reactant,
                nnodes=inpfileq['num_nodes'],
                DQMAG_MAX=inpfileq['DQMAG_MAX'],
                BDIST_RATIO=inpfileq['BDIST_RATIO'],
                CONV_TOL=inpfileq['CONV_TOL'],
                ADD_NODE_TOL=inpfileq['ADD_NODE_TOL'],
                optimizer=optimizer,
                print_level=inpfileq['gsm_print_level'],
                driving_coords=driving_coordinates,
                ID=inpfileq['ID'],
                use_multiprocessing=inpfileq['use_multiprocessing'],
                )

    # For seam calculation
    if inpfileq['gsm_type']!='SE_Cross' and (inpfileq['PES_type'] =="Avg_PES" or inpfileq['PES_type']=="Penalty_PES"):
        optimizer.opt_cross = True

    if not inpfileq['reactant_geom_fixed'] and inpfileq['gsm_type']!='SE_Cross':
        nifty.printcool("REACTANT GEOMETRY NOT FIXED!!! OPTIMIZING")
        optimizer.optimize(
           molecule = reactant,
           refE = reactant.energy,
           opt_steps=100,
           )

    if not inpfileq['product_geom_fixed'] and inpfileq['gsm_type']=='DE_GSM':
        nifty.printcool("PRODUCT GEOMETRY NOT FIXED!!! OPTIMIZING")
        optimizer.optimize(
           molecule = product,
           refE = reactant.energy,
           opt_steps=100,
           )

    rtype=2
    if args.only_climb:
        rtype=1
    elif args.no_climb:
        rtype=0
    elif args.optimize_meci:
        rtype=0
    elif args.optimize_mesx:
        rtype=1
    elif inpfileq['gsm_type']=="SE_Cross":
        rtype=1

    if inpfileq['max_opt_steps'] is None:
        if inpfileq['gsm_type']=="DE_GSM":
            inpfileq['max_opt_steps']=3
        else:
            inpfileq['max_opt_steps']=20

    if args.restart_file is not None:
        gsm.restart_string(args.restart_file,rtype)
    gsm.go_gsm(inpfileq['max_gsm_iters'],inpfileq['max_opt_steps'],rtype)
    
    if inpfileq['gsm_type']=='SE_Cross':
        post_processing(
                gsm,
                analyze_ICs=args.dont_analyze_ICs,
                have_TS=False,
                )
    else:
        post_processing(
                gsm,
                analyze_ICs=args.dont_analyze_ICs,
                have_TS=True,
                )

    cleanup_scratch(gsm.ID)

    return


def read_isomers_file(isomers_file):
    with open(isomers_file) as f:
        tmp = filter(None, (line.rstrip() for line in f))
        lines=[]
        for line in tmp:
            lines.append(line)

    driving_coordinates=[]
    
    if lines[0] == "NEW":
        start = 1
    else:
        start = 0

    for line in lines[start:]:
        dc = []
        twoInts=False
        threeInts=False
        fourInts=False
        for i,elem in enumerate(line.split()):
            if i==0:
                dc.append(elem)
                if elem=="ADD" or elem=="BREAK":
                    twoInts =True
                elif elem=="ANGLE":
                    threeInts =True
                elif elem=="TORSION" or elem=="OOP":
                    fourInts =True
                elif elem=="ROTATE":
                    threeInts =True
            else:
                if twoInts and i>2:
                    dc.append(float(elem))
                elif twoInts and i>3:
                    dc.append(float(elem)) # add break dist
                elif threeInts and i>3:
                    dc.append(float(elem))
                elif fourInts and i>4:
                    dc.append(float(elem))
                else:
                    dc.append(int(elem))
        driving_coordinates.append(dc)

    nifty.printcool("driving coordinates {}".format(driving_coordinates))
    return driving_coordinates


def get_driving_coord_prim(dc):
    prim=None
    if "ADD" in dc or "BREAK" in dc:
        if dc[1]<dc[2]:
            prim = Distance(dc[1]-1,dc[2]-1)
        else:
            prim = Distance(dc[2]-1,dc[1]-1)
    elif "ANGLE" in dc:
        if dc[1]<dc[3]:
            prim = Angle(dc[1]-1,dc[2]-1,dc[3]-1)
        else:
            prim = Angle(dc[3]-1,dc[2]-1,dc[1]-1)
    elif "TORSION" in dc:
        if dc[1]<dc[4]:
            prim = Dihedral(dc[1]-1,dc[2]-1,dc[3]-1,dc[4]-1)
        else:
            prim = Dihedral(dc[4]-1,dc[3]-1,dc[2]-1,dc[1]-1)
    elif "OOP" in dc:
        if dc[1]<dc[4]:
            prim = OutOfPlane(dc[1]-1,dc[2]-1,dc[3]-1,dc[4]-1)
        else:
            prim = OutOfPlane(dc[4]-1,dc[3]-1,dc[2]-1,dc[1]-1)
    return prim


def read_force_file(force_file):
    raise NotImplementedError
    with open(force_file) as f:
        lines=f.readlines()
    force=[]
    return

def cleanup_scratch(ID):
    cmd = "rm scratch/growth_iters_{:03d}_*.xyz".format(ID)
    os.system(cmd)
    cmd = "rm scratch/opt_iters_{:03d}_*.xyz".format(ID)
    os.system(cmd)
    ##cmd = "rm scratch/initial_ic_reparam_{:03d}_{:03d}.xyz".format()
    #if inpfileq['EST_Package']=="DFTB":
    #    for i in range(self.gsm.nnodes):
    #        cmd = 'rm -rf scratch/{}'.format(i)
    #        os.system(cmd)

def plot(fx,x,title):
    plt.figure(1)
    plt.title("String {:04d}".format(title))
    plt.plot(x,fx,color='b', label = 'Energy',linewidth=2,marker='o',markersize=12)
    plt.xlabel('Node Number')
    plt.ylabel('Energy (kcal/mol)')
    plt.legend(loc='best')
    plt.savefig('{:04d}_string.png'.format(title),dpi=600)

def get_nproc():
    # THIS FUNCTION DOES NOT RETURN "USABLE" CPUS 
    try:
        return os.cpu_count()
    except (ImportError, NotImplementedError):
        pass
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass
    raise Exception('Can not determine number of CPUs on this system')

def post_processing(gsm,analyze_ICs=False,have_TS=True):
    plot(fx=gsm.energies,x=range(len(gsm.energies)),title=gsm.ID)

    ICs = []
    ICs.append(gsm.nodes[0].primitive_internal_coordinates)

    # TS energy
    if have_TS:
        minnodeR = np.argmin(gsm.energies[:gsm.TSnode])
        TSenergy = gsm.energies[gsm.TSnode] - gsm.energies[minnodeR]
        print(" TS energy: %5.4f" % TSenergy)
        print(" absolute energy TS node %5.4f" % gsm.nodes[gsm.TSnode].energy)
        minnodeP = gsm.TSnode + np.argmin(gsm.energies[gsm.TSnode:])
        print(" min reactant node: %i min product node %i TS node is %i" % (minnodeR,minnodeP,gsm.TSnode))

        # ICs
        ICs.append(gsm.nodes[minnodeR].primitive_internal_values)
        ICs.append(gsm.nodes[gsm.TSnode].primitive_internal_values)
        ICs.append(gsm.nodes[minnodeP].primitive_internal_values)
        with open('IC_data_{:04d}.txt'.format(gsm.ID),'w') as f:
            f.write("Internals \t minnodeR: {} \t TSnode: {} \t minnodeP: {}\n".format(minnodeR,gsm.TSnode,minnodeP))
            for x in zip(*ICs):
                f.write("{0}\t{1}\t{2}\t{3}\n".format(*x))

    else:
        minnodeR = 0
        minnodeP = gsm.nR
        print(" absolute energy end node %5.4f" % gsm.nodes[gsm.nR].energy)
        print(" difference energy end node %5.4f" % gsm.nodes[gsm.nR].difference_energy)
        # ICs
        ICs.append(gsm.nodes[minnodeR].primitive_internal_values)
        ICs.append(gsm.nodes[minnodeP].primitive_internal_values)
        with open('IC_data_{}.txt'.format(gsm.ID),'w') as f:
            f.write("Internals \t Beginning: {} \t End: {}".format(minnodeR,gsm.TSnode,minnodeP))
            for x in zip(*ICs):
                f.write("{0}\t{1}\t{2}\n".format(*x))

    # Delta E
    deltaE = gsm.energies[minnodeR] - gsm.energies[minnodeP]
    print(" Delta E is %5.4f" % deltaE)

#def go_gsm(gsm,max_iters=50,opt_steps=3,rtype=2):
#    gsm.go_gsm(max_iters=max_iters,opt_steps=opt_steps,rtype=rtype)

def print_msg():
    msg="""
    __        __   _                            _        
    \ \      / /__| | ___ ___  _ __ ___   ___  | |_ ___  
     \ \ /\ / / _ \ |/ __/ _ \| '_ ` _ \ / _ \ | __/ _ \ 
      \ V  V /  __/ | (_| (_) | | | | | |  __/ | || (_) |
       \_/\_/ \___|_|\___\___/|_| |_| |_|\___|  \__\___/ 
                                    ____ ____  __  __ 
                       _ __  _   _ / ___/ ___||  \/  |
                      | '_ \| | | | |  _\___ \| |\/| |
                      | |_) | |_| | |_| |___) | |  | |
                      | .__/ \__, |\____|____/|_|  |_|
                      |_|    |___/                    
#==========================================================================#
#| If this code has benefited your research, please support us by citing: |#
#|                                                                        |# 
#| Aldaz, C.; Kammeraad J. A.; Zimmerman P. M. "Discovery of conical      |#
#| intersection mediated photochemistry with growing string methods",     |#
#| Phys. Chem. Chem. Phys., 2018, 20, 27394                               |#
#| http://dx.doi.org/10.1039/c8cp04703k                                   |#
#|                                                                        |# 
#| Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with |#
#| translation and rotation coordinates", J. Chem, Phys. 144, 214108.     |#
#| http://dx.doi.org/10.1063/1.4952956                                    |#
#==========================================================================#


    """
    print(msg)



if __name__=='__main__':
    main()

