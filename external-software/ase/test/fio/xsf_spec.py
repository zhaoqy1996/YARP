import numpy as np

from ase.io import read, write

# This reads and writes XSF examples taken from the
# specification of the XSF format:
#
#   http://www.xcrysden.org/doc/XSF.html
#
# See the data below

def check(name, xsf_text, check_data):
    fname = '%s.xsf' % name
    fd = open(fname, 'w')
    fd.write(xsf_text)
    fd.close()

    print('Read: %s' % fname)
    images = read(fname + '@:', read_data=check_data)
    if check_data:
        array, images = images

    assert isinstance(images, list)
    print('  Images: %s' % len(images))
    for image in images:
        print('    %s' % image)

    # Now write the same system back out:
    outfname = 'out.%s' % fname
    if check_data:
        write(outfname, images, data=array)
    else:
        write(outfname, images)

    # ...and read it back in:
    images2 = read(outfname + '@:', read_data=check_data)
    if check_data:
        array2, images2 = images2

    # It should be the same as the original file.
    assert images == images2
    if check_data:
        print(array)
        print(array2)
        assert np.abs(array - array2).max() < 1e-13

    # In fact, if we write it back out again, it should be
    # byte-wise identical to the other file that we just wrote.
    # So do that:
    outfname2 = 'doubleout.%s' % fname
    if check_data:
        write(outfname2, images2, data=array2)
    else:
        write(outfname2, images2)
    assert open(outfname).read() == open(outfname2).read()


def main():
    files = {'01-comments': f1,
             '02-atoms': f2,
             '03-periodic': f3,
             '04-forces-atoms': f4,
             '05-forces-slab': f5,
             '06-anim-atoms': f6,
             '07-anim-crystal-fixcell': f7,
             '08-anim-crystal-varcell': f8}

    names = list(sorted(files.keys()))

    for name in names:
        check(name, files[name], check_data=False)
        check('%s-ignore-datagrid' % name, files[name] + datagrid,
              check_data=False)
        check('%s-read-datagrid' % name, files[name] + datagrid,
              check_data=True)


f1 = """
 # this is a specification 
 # of ZnS crystal structure

 CRYSTAL

 # these are primitive lattice vectors (in Angstroms)

 PRIMVEC
    2.7100000000    2.7100000000    0.0000000000
    2.7100000000    0.0000000000    2.7100000000
    0.0000000000    2.7100000000    2.7100000000

 # these are convetional lattice vectors (in Angstroms)

 CONVVEC
    5.4200000000    0.0000000000    0.0000000000
    0.0000000000    5.4200000000    0.0000000000
    0.0000000000    0.0000000000    5.4200000000

 # these are atomic coordinates in a primitive unit cell 
 # (in Angstroms)

 PRIMCOORD
 2 1
 16      0.0000000000     0.0000000000     0.0000000000
 30      1.3550000000    -1.3550000000    -1.3550000000
"""

f2 = """
 ATOMS
  6    2.325243   -0.115261    0.031711
  1    2.344577   -0.363301    1.077589
  9    3.131708   -0.909527   -0.638930
  9    2.736189    1.130568   -0.134093
  8    1.079338   -0.265162   -0.526351
  6    0.007719   -0.041269    0.244204
  9    0.064656    1.154700    0.824420
  9   -0.042641   -0.911850    1.255074
  8   -1.071578   -0.152842   -0.539134
  6   -2.310374    0.036537    0.022189
  1   -2.267004    0.230694    1.077874
  9   -2.890949    1.048938   -0.593940
  9   -3.029540   -1.046542   -0.203665
"""

f3 = """
 CRYSTAL
 PRIMVEC
    0.0000000    2.7100000    2.7100000
    2.7100000    0.0000000    2.7100000
    2.7100000    2.7100000    0.0000000
 CONVVEC
    5.4200000    0.0000000    0.0000000
    0.0000000    5.4200000    0.0000000
    0.0000000    0.0000000    5.4200000
 PRIMCOORD
 2 1
 16      0.0000000     0.0000000     0.0000000
 30      1.3550000    -1.3550000    -1.3550000
"""

f4 = """
 ATOMS
  8   0.00000  0.00000  0.00000   -.05164  .00000 -.03999
  1   0.00000  0.00000  1.00000    .01769  .00000  .03049
  1   0.96814  0.00000 -0.25038    .03395  .00000  .00949
"""

f5 = """
SLAB
PRIMVEC
  5.8859828533     0.0000000000     0.0000000000
  0.0000000000     5.8859828533     0.0000000000
  0.0000000000     0.0000000000     1.0000000000
PRIMCOORD
11 1
 6    3.674759   2.942992  -3.493103     -0.021668   0.000000  -0.057324
 1    4.121990   3.816734  -4.007689     -0.000478   0.001204   0.006657
 1    4.121990   2.069250  -4.007689     -0.000478  -0.001204   0.006657
 6    2.211226   2.942992  -3.493103      0.021668   0.000000  -0.057324
 1    1.763995   3.816734  -4.007689      0.000478   0.001204   0.006657
 1    1.763995   2.069250  -4.007689      0.000478  -0.001204   0.006657
 8    0.000000   0.000000  -2.719012      0.000000   0.000000  -0.050242
47    4.448147   4.449892  -1.919011     -0.022812  -0.029123   0.007553
47    4.448147   1.436093  -1.919011     -0.022812   0.029123   0.007553
47    1.437838   4.449892  -1.919011      0.022812  -0.029123   0.007553
47    1.437838   1.436093  -1.919011      0.022812   0.029123   0.007553
"""

f6 = """
ANIMSTEPS 4
ATOMS 1
 8    0.0000  0.0000  0.0000     -0.0516  0.0000 -0.0399
 1    0.0000  0.0000  1.0000      0.0176  0.0000  0.0304
 1    0.9681  0.0000 -0.2503      0.0339  0.0000  0.0094
ATOMS 2
 8   -0.1480  0.0000 -0.1146      0.0020  0.0000  0.0015
 1   -0.0468  0.0000  0.9134     -0.0069  0.0000  0.0069
 1    0.8726  0.0000 -0.2740      0.0049  0.0000 -0.0084
ATOMS 3
 8   -0.1032  0.0000 -0.0799      0.0013  0.0000  0.0010
 1   -0.0319  0.0000  0.9591      0.0011  0.0000 -0.0028
 1    0.9205  0.0000 -0.2710     -0.0025  0.0000  0.0018
ATOMS 4
 8   -0.1102  0.0000 -0.0853      0.0001  0.0000  0.0000
 1   -0.0345  0.0000  0.9503     -0.0000  0.0000 -0.0000
 1    0.9114  0.0000 -0.2714     -0.0000  0.0000 -0.0000
"""

f7 = """
ANIMSTEPS 2
CRYSTAL
PRIMVEC
   0.0000000    2.7100000    2.7100000
   2.7100000    0.0000000    2.7100000
   2.7100000    2.7100000    0.0000000
PRIMCOORD 1
   2 1
   16      0.0000000     0.0000000     0.0000000
   30      1.3550000    -1.3550000    -1.3550000
PRIMCOORD 2
   2 1
   16      0.0000000     0.0000000     0.0000000
   30      1.2550000    -1.2550000    -1.2550000
"""

f8 = """
ANIMSTEPS 2
CRYSTAL
PRIMVEC 1
   2.7100000    2.7100000    0.00000000
   2.7100000    0.0000000    2.71000000
   0.0000000    2.7100000    2.71000000
CONVVEC 1
   5.4200000    0.0000000    0.00000000
   0.0000000    5.4200000    0.00000000
   0.0000000    0.0000000    5.42000000
PRIMCOORD 1
   2 1
   16      0.0000000     0.0000000     0.00000000
   30      1.3550000    -1.3550000    -1.35500000
PRIMVEC 2
   2.9810000    2.9810000    0.00000000
   2.9810000    0.0000000    2.98100000
   0.0000000    2.9810000    2.98100000
CONVVEC 2
   5.9620000    0.0000000    0.00000000
   0.0000000    5.9620000    0.00000000
   0.0000000    0.0000000    5.96200000
PRIMCOORD 2
   2 1
   16      0.0000000     0.0000000     0.00000000
   30      1.5905000    -1.5905000    -1.59050000
"""

datagrid = """BEGIN_BLOCK_DATAGRID_3D                        
   my_first_example_of_3D_datagrid      
   BEGIN_DATAGRID_3D_this_is_3Dgrid#1           
     5  5  5                              
     0.0 0.0 0.0                          
     1.0 0.0 0.0                          
     0.0 1.0 0.0                                
     0.0 0.0 1.0                          
       0.000  1.000  2.000  5.196  8.000        
       1.000  1.414  2.236  5.292  8.062        
       2.000  2.236  2.828  5.568  8.246        
       3.000  3.162  3.606  6.000  8.544        
       4.000  4.123  4.472  6.557  8.944        
                                        
       1.000  1.414  2.236  5.292  8.062        
       1.414  1.732  2.449  5.385  8.124        
       2.236  2.449  3.000  5.657  8.307        
       3.162  3.317  3.742  6.083  8.602        
       4.123  4.243  4.583  6.633  9.000        
                                        
       2.000  2.236  2.828  5.568  8.246        
       2.236  2.449  3.000  5.657  8.307        
       2.828  3.000  3.464  5.916  8.485        
       3.606  3.742  4.123  6.325  8.775        
       4.472  4.583  4.899  6.856  9.165        
                                        
       3.000  3.162  3.606  6.000  8.544        
       3.162  3.317  3.742  6.083  8.602        
       3.606  3.742  4.123  6.325  8.775        
       4.243  4.359  4.690  6.708  9.055        
       5.000  5.099  5.385  7.211  9.434        
                                        
       4.000  4.123  4.472  6.557  8.944        
       4.123  4.243  4.583  6.633  9.000        
       4.472  4.583  4.899  6.856  9.165        
       5.000  5.099  5.385  7.211  9.434        
       5.657  5.745  6.000  7.681  9.798        
   END_DATAGRID_3D                      
 END_BLOCK_DATAGRID_3D          
"""

main()
