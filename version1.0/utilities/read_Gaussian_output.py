import sys,os,argparse,subprocess,shutil,time,glob,scandir,fnmatch
from operator import add
import numpy as np
# Load taffi modules        
from taffi_functions import xyz_write

def main(argv):

    parser = argparse.ArgumentParser(description='Use to read in Gaussian geo-opt output file and write a xyz file')

    #optional arguments                                             
    parser.add_argument('-i', dest='inputF',
                        help = 'The program expects a Gaussian geo-opt output file')

    parser.add_argument('-n', dest='E_number',
                        help = 'The program expects the number of elements')

    parser.add_argument('-o', dest='output', default='result.xyz',
                        help = 'Controls the output xyz file name for the result')

    parser.add_argument('-t', dest='jobtype', default='geo-opt',
                        help = 'Controls the input Gaussian job type (available:geo-opt,IRC,G4,...)')

    parser.add_argument('-scale', dest='scale', default='0.0',
                        help = 'Controls the rotation scale when facing imaginary frequency, default is 0 means no rotation')

    parser.add_argument('--count', dest='count', default=0, action='store_const', const=1,
                        help = 'When set, number of gradient calls will be counted, (only works for geo-opt job, default: off)')
			       
    # parse configuration and apply main function
    args    = parser.parse_args()
    E_number= int(args.E_number)
    scale   = float(args.scale)
    if args.jobtype not in ["geo-opt","IRC"]:
        print("Error found, not just support geo-opt and IRC output file")
        quit()

    if args.count == 1:
        count_flag = True
    else:
        count_flag = False

    if args.jobtype == "geo-opt":
        read_geoopt(args.inputF,args.output,E_number,count_flag=count_flag,scale=scale)
    else:
        read_IRC(args.inputF,args.output,E_number)

# Function to read Gaussian geometry optimization file
# Inputs:        inputF: input Gaussian output file name
#                output: XXX.xyz which is file name of output geometry
#                Element_num: number of atomns
#                count_flag: when this flag is on, count the number of gradient calls
#                scale: when imaginary frequency appears, rotate the molecule following imaginary frequency direction
# Output:        None  
#
def read_geoopt(inputF,output,Element_num,count_flag=False,scale=0.3):

    # Initialize periodic table
    periodic = { "H": 1,  "He": 2,\
                 "Li":3,  "Be":4,                                                                                                      "B":5,    "C":6,    "N":7,    "O":8,    "F":9,    "Ne":10,\
                 "Na":11, "Mg":12,                                                                                                     "Al":13,  "Si":14,  "P":15,   "S":16,   "Cl":17,  "Ar":18,\
                  "K":19, "Ca":20,   "Sc":21,  "Ti":22,  "V":23,  "Cr":24,  "Mn":25,  "Fe":26,  "Co":27,  "Ni":28,  "Cu":29,  "Zn":30, "Ga":31,  "Ge":32,  "As":33, "Se":34,  "Br":35,   "Kr":36,\
                 "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                 "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}    

    # create an inverse periodic table
    invert_periodic = {}
    for p in periodic.keys():
        invert_periodic[periodic[p]]=p

    count_xyz = []
    E = []
    G = []
    change_G = []

    # find final xyz 
    imag_flag = False
    with open(inputF,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            # identify whether there are imaginary frequency in the out put file.
            if 'imaginary' in fields and 'frequencies' in fields and '(negative' in fields:
                imag_flag = True
                image_number = int(fields[1])
            if len(fields) == 2 and fields[0] == "Standard" and fields[1] == "orientation:":
                count_xyz += [lc]
            if imag_flag == True and len(fields)==5 and fields[0] == 'Frequencies' and float(fields[2]) < 0:
                imag_lc = lc

    N_grad = len(count_xyz) - 2

    with open(inputF,'r') as f:
        for lc,lines in enumerate(f):
            if lc >= count_xyz[-1]+5 and lc < count_xyz[-1]+5+Element_num:
                fields = lines.split()
                E += [invert_periodic[float(fields[1])]]
                G += [[float(fields[3]),float(fields[4]),float(fields[5])]]
        
    #  If imaginary freq found, change the input file
    if imag_flag == False:
        if count_flag:
            xyz_write(output,E,G,comment='Number of gradient calls is {}'.format(N_grad))
        else:
            xyz_write(output,E,G)
        
    else:
        with open(inputF,'r') as f:
            for lc,lines in enumerate(f):
                if lc >= imag_lc+5 and lc < imag_lc+5+Element_num:
                    fields = lines.split()
                    if image_number == 1:
                        change_G += [[float(fields[2]),float(fields[3]),float(fields[4])]]
                    elif image_number == 2:
                        change_G += [[float(fields[2])+float(fields[5]),float(fields[3])+float(fields[6]),\
                                      float(fields[4])+float(fields[7])]]
                    elif image_number == 3:
                        change_G += [[float(fields[2])+float(fields[5])+float(fields[8]),\
                                      float(fields[3])+float(fields[6])+float(fields[9]),\
                                      float(fields[4])+float(fields[7])+float(fields[10])]]
        for l in range(len(change_G)):
            change_G[l] = [scale * ll for ll in change_G[l]]
            G_shift = []
        for l in range(len(G)):
            G_shift += [list( map(add, G[l], change_G[l]) )]

        if count_flag:
            xyz_write(output,E,G_shift,comment='Number of gradient calls is {}'.format(N_grad))
        else:
            xyz_write(output,E,G_shift)

    return

# Function to read Gaussian IRC output file to generate a sequence of xyz files among reaction pathway
# Inputs:        inputF: input Gaussian output file name
#                output: XXX.xyz which is file name of output pathway
#                Element_num: number of atomns
# Output:        None  
#
def read_IRC(inputF,output,Element_num):

    # read gaussian optput file
    with open(inputF,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            # find elements location
            if len(fields) == 2 and fields[0] == 'Symbolic' and fields[1] == 'Z-matrix:':
                E_count = lc + 1 

            # find the summary of IRC
            if len(fields)== 5 and fields[0] == 'Summary' and fields[1] == 'of' and fields[2] == 'reaction':
                count_start = lc + 2
                
            # locate the end of summary 
            if len(fields)== 5 and fields[0]=='Total' and fields[1]=='number' and fields[2]=='of' and fields[3]=='points:':
                N_image = int(fields[4]) + 1
                count_end = lc - 2
    
    # initialize the geometry dictionary
    geo_dict={}
    E = []
    for i in range(N_image+1)[1:]:
        geo_dict[str(i)]=[]

    # re-open the gaussian output file and write info into geo-dict 
    with open(inputF,'r') as f:
        for lc,lines in enumerate(f):
            if lc > E_count and lc <= (E_count + Element_num):
                fields = lines.split()
                E += [fields[0]]

            if lc > count_start and lc < count_end:
                fields = lines.split()
                if fields[0] in geo_dict.keys():
                    geo_dict[fields[0]] += fields[1:]

    # write into output xyz file
    with open(output,'w') as f:
        for i in range(N_image+1)[1:]:
            f.write('{}\n'.format(Element_num))
            f.write('Image {}\n'.format(i))
            geo_list = [float(j) for j in geo_dict[str(i)][2:]]
            for count_k,k in enumerate(E):
                f.write("{:<20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(k,geo_list[count_k*3],geo_list[count_k*3+1],geo_list[count_k*3+2]))

    return

if __name__ == "__main__":
    main(sys.argv[1:])
