import sys,os,argparse,subprocess,shutil,time,glob,scandir,fnmatch
import numpy as np

output_folder = '/scratch/halstead/z/zhao922/pyGSM/Energy/KHP/all-product'
output_files = [ os.path.join(dp, f) for dp, dn, filenames in scandir.walk(output_folder) for f in filenames if (fnmatch.fnmatch(f,"*.out") )]
Energy_dict = {}
for output in output_files:
    name = output.split('/')[-1].split('.')[0]
    Energy_dict[name] = {}
    with open(output,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if 'Normal' in fields and 'termination' in fields and 'of' in fields and 'Gaussian' in fields:
                finish=True

            if len(fields) == 7 and fields[0] == 'Sum' and fields[2] == 'electronic' and fields[4] == 'zero-point' and fields[5] == 'Energies=':
                E = float(fields[6])
                Energy_dict[name]["E"] = E

            if len(fields) == 7 and fields[0] == 'Sum' and fields[2] == 'electronic' and fields[4] == 'thermal' and fields[5] == 'Enthalpies=':
                H = float(fields[6])
                Energy_dict[name]["H"] = H

            if len(fields) == 8 and fields[0] == 'Sum' and fields[2] == 'electronic' and fields[4] == 'thermal' and fields[5] == 'Free':
                F = float(fields[7])
                Energy_dict[name]["F"] = F
                
    '''
    if finish:
        with open('compounds_energy.txt',"a") as g:
            g.write("{:<30s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(name,E,H,F))
            
    '''
#'''
H_KHP = -343.399667
with open("frag_inchi.txt","r") as f:
    for lc,lines in enumerate(f):
        fields = lines.split()
        if lc == 0:continue
        if len(fields) > 1:
            name = fields[0]
            compounds = [fields[i] for i in range(len(fields))[1:] ]
            E = 0
            H = 0
            flag = True
            for i in compounds:
                if i in Energy_dict.keys():
                    E += Energy_dict[i]["E"]
                    H += Energy_dict[i]["H"]
                else:
                    flag = False
                    print("{} for {} is missing".format(i,name))
        if flag:
            if E == 0:
                print("Check this compound {}...".format(name))
            else:
                with open("product_energy.txt","a") as g:
                    g.write("{:<30s} {:< 20.8f} {:< 20.8f}".format(name,E,(H-H_KHP) * 630.0))
                    for i in compounds:
                        g.write("{}\t\t".format(i))
                    g.write("\n")
#'''
