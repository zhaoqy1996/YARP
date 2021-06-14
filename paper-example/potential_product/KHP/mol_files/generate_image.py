import sys,os,argparse,subprocess,time,fnmatch

product_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 51, 52, 53, 54, 58, 59, 61, 62, 63, 65, 67, 68, 69, 71, 72, 73, 75, 81, 82, 85, 87, 89, 90, 91, 94, 95, 96, 97, 98, 102, 103, 104, 105, 106, 107, 108, 109, 111, 113, 114, 119, 123, 125, 126, 127, 129, 130, 132, 133, 135, 136, 138, 139, 140, 149, 150, 152, 153, 155, 160, 163, 165]
total_lines = []
for p in product_list:
    with open("pp_{}.mol".format(p),"r") as f:
        for lc,lines in enumerate(f):
            if lc==0:
                lines="P{}\n".format(p)
            total_lines += [lines]

for lines in total_lines:
    with open("product.mol","a") as f:
        f.write(lines)
#substring = "obabel -imol pp_{}.mol -O images/pp_{}.svg -xO molfile -xa -xu -xA -xd".format(p,p)
#output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0]
