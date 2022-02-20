import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
# Load modules in same folder        
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/utilities')
from taffi_functions import *
from utility import check_bimolecule

def main(argv):

    # parse configuration and apply main function
    target_dic = argv[0]

    # load in resulting conformations
    inf=open(target_dic+'/crest_conformers.xyz')
    lines=inf.readlines()
    inf.close()

    # create xyz files
    count=1
    if os.path.isdir(target_dic+'/results') is False: os.mkdir(target_dic+'/results')
    for j in range(2, len(lines)):
        fields=lines[j].split()
        if len(fields)==4 and len(lines[j-1].split())!=4:
            out=open(target_dic+'/results/{}.xyz'.format(count),'w+')
            out.write(lines[j-2])
            out.write(lines[j-1])
            out.write(lines[j])
            count=count+1
        elif len(fields)==4:
            out.write(lines[j])
        elif len(fields)!=4 and len(lines[j-1].split())!=4:
            out.close()
    out.close()

    # analyze each conformation and remove useless ones
    target_dic = target_dic +'/results'
    xyz_files  = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(target_dic) for f in filenames if (fnmatch.fnmatch(f,"*.xyz") )])
    for j in xyz_files:
        E,G=xyz_parse(j)
        adj=Table_generator(E,G)
        if not check_bimolecule(adj,G,factor=2.0): 
            print('YARP removes {}/{} due to far bimolecular product'.format(target_dic,j))
            os.system('rm {}'.format(j))

if __name__ == "__main__":
    main(sys.argv[1:])
