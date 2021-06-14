# additional tests of the extended XYZ file I/O
# (which is also included in oi.py test case)
# maintainted by James Kermode <james.kermode@gmail.com>

import os

import numpy as np

import ase.io
from ase.io import extxyz
from ase.atoms import Atoms
from ase.build import bulk

# array data of shape (N, 1) squeezed down to shape (N, ) -- bug fixed
# in commit r4541
at = bulk('Si')
ase.io.write('to.xyz', at, format='extxyz')
at.arrays['ns_extra_data'] = np.zeros((len(at), 1))
assert at.arrays['ns_extra_data'].shape == (2, 1)

ase.io.write('to_new.xyz', at, format='extxyz')
at_new = ase.io.read('to_new.xyz')
assert at_new.arrays['ns_extra_data'].shape == (2,)

os.unlink('to.xyz')
os.unlink('to_new.xyz')

#test comment read/write with vec_cell
at.info['comment'] = 'test comment'
ase.io.write('comment.xyz', at, comment=at.info['comment'], vec_cell=True)
r = ase.io.read('comment.xyz')
assert at == r
os.unlink('comment.xyz')

# write sequence of images with different numbers of atoms -- bug fixed
# in commit r4542
images = [at, at * (2, 1, 1), at * (3, 1, 1)]
ase.io.write('multi.xyz', images, format='extxyz')
read_images = ase.io.read('multi.xyz@:')
assert read_images == images

#test vec_cell writing and reading
images[1].set_pbc([True,True,False])
images[2].set_pbc([True,False,False])
ase.io.write('multi.xyz', images, vec_cell=True)
cell = images[1].get_cell()
cell[-1] = [0.0, 0.0, 0.0]
images[1].set_cell(cell)
cell = images[2].get_cell()
cell[-1] = [0.0, 0.0, 0.0]
cell[-2] = [0.0, 0.0, 0.0]
images[2].set_cell(cell)
read_images = ase.io.read('multi.xyz@:')
assert read_images == images
os.unlink('multi.xyz')
# also test for vec_cell with whitespaces
f = open('structure.xyz', 'w')
f.write("""1
Coordinates
C         -7.28250        4.71303       -3.82016
  VEC1 1.0 0.1 1.1
1

C         -7.28250        4.71303       -3.82016
VEC1 1.0 0.1 1.1
""")
f.close()
a = ase.io.read('structure.xyz',index=0)
b = ase.io.read('structure.xyz',index=1)
assert a == b
os.unlink('structure.xyz')

# read xyz containing trailing blank line
# also test for upper case elements
f = open('structure.xyz', 'w')
f.write("""4
Coordinates
MG        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016

""")
f.close()
a = ase.io.read('structure.xyz')
assert a[0].symbol == 'Mg'
os.unlink('structure.xyz')

# read xyz with / and @ signs in key value
f = open('slash.xyz', 'w')
f.write("""4
key1=a key2=a/b key3=a@b key4="a@b"
Mg        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016
""")
f.close()
a = ase.io.read('slash.xyz')
assert a.info['key1'] == r'a'
assert a.info['key2'] == r'a/b'
assert a.info['key3'] == r'a@b'
assert a.info['key4'] == r'a@b'
os.unlink('slash.xyz')

struct = Atoms('H4', pbc=[True, True, True],
                cell=[[4.00759, 0.0, 0.0], [-2.003795, 3.47067475, 0.0], [3.06349683e-16, 5.30613216e-16, 5.00307]], positions=[[-2.003795e-05, 2.31379473, 0.875437189], [2.00381504, 1.15688001, 4.12763281], [2.00381504, 1.15688001, 3.37697219], [-2.003795e-05, 2.31379473, 1.62609781]])
struct.info = {'key_value_pairs': {'dataset': 'deltatest', 'kpoints': np.array([28, 28, 20]), 'identifier': 'deltatest_H_1.00'}, 'unique_id': '4cf83e2f89c795fb7eaf9662e77542c1'}

ase.io.write('tmp.xyz', struct)
os.unlink('tmp.xyz')

# Complex properties line. Keys and values that break with a regex parser.
# see https://gitlab.com/ase/ase/issues/53 for more info

complex_xyz_string = (
    ' '  # start with a separator
    'str=astring '
    'quot="quoted value" '
    u'quote_special="a_to_Z_$%%^&*\xfc\u2615" '
    r'escaped_quote="esc\"aped" '
    'true_value '
    'false_value = F '
    'integer=22 '
    'floating=1.1 '
    'int_array={1 2 3} '
    'float_array="3.3 4.4" '
    'a3x3_array="1 4 7 2 5 8 3 6 9" '  # fortran ordering
    'Lattice="  4.3  0.0 0.0 0.0  3.3 0.0 0.0 0.0  7.0 " '  # spaces in array
    'scientific_float=1.2e7 '
    'scientific_float_2=5e-6 '
    'scientific_float_array="1.2 2.2e3 4e1 3.3e-1 2e-2" '
    'not_array="1.2 3.4 text" '
    'nested_brackets=[[1,2],[3,4]] '  # gets flattented if not 3x3
    'bool_array={T F T F} '
    'bool_array_2=" T, F, T " ' # leading spaces
    'not_bool_array=[T F S] '
    # read and write
    u'\xfcnicode_key=val\xfce '
    u'unquoted_special_value=a_to_Z_$%%^&*\xfc\u2615 '
    '2body=33.3 '
    'hyphen-ated '
    # parse only
    'many_other_quotes=({[4 8 12]}) '
    'comma_separated="7, 4, -1" '
    'bool_array_commas=[T, T, F, T] '
    'Properties=species:S:1:pos:R:3 '
    'multiple_separators       '
    'double_equals=abc=xyz '
    'trailing'
)

expected_dict = {
    'str': 'astring',
    'quot': "quoted value",
    'quote_special': u"a_to_Z_$%%^&*\xfc\u2615",
    'escaped_quote': r"esc\"aped",
    'true_value': True,
    'false_value': False,
    'integer': 22,
    'floating': 1.1,
    'int_array': np.array([1, 2, 3]),
    'float_array': np.array([3.3, 4.4]),
    'a3x3_array': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    'Lattice': np.array([[4.3, 0.0, 0.0], [0.0, 3.3, 0.0], [0.0, 0.0, 7.0]]),
    'scientific_float': 1.2e7,
    'scientific_float_2': 5e-6,
    'scientific_float_array': np.array([1.2, 2200, 40, 0.33, 0.02]),
    'not_array': "1.2 3.4 text",
    'nested_brackets': np.array([1, 2, 3, 4]),
    'bool_array': np.array([True, False, True, False]),
    'bool_array_2': np.array([True, False, True]),
    'not_bool_array': 'T F S',
    u'\xfcnicode_key': u'val\xfce',
    'unquoted_special_value': u'a_to_Z_$%%^&*\xfc\u2615',
    '2body': 33.3,
    'hyphen-ated': True,
    'many_other_quotes': np.array([4, 8, 12]),
    'comma_separated': np.array([7, 4, -1]),
    'bool_array_commas': np.array([True, True, False, True]),
    'Properties': 'species:S:1:pos:R:3',
    'multiple_separators': True,
    'double_equals': 'abc=xyz',
    'trailing': True
}

parsed_dict = extxyz.key_val_str_to_dict(complex_xyz_string)
np.testing.assert_equal(parsed_dict, expected_dict)

# Round trip through a file with complex line.
# Create file with the complex line and re-read it afterwards.
# Test is disabled as it requires that file io defaults to utf-8 encoding
# which is not guaranteed on Python 2 and varies with LC_ variables
# on linux. Test can be enabled if ase ever strongly enforces utf-8
# everywhere.
if False:
    with open('complex.xyz', 'w', encoding='utf-8') as f_out:
        f_out.write('1\n{}\nH 1.0 1.0 1.0'.format(complex_xyz_string))
    complex_atoms = ase.io.read('complex.xyz')

    # test all keys end up in info, as expected
    for key, value in expected_dict.items():
        if key in ['Properties']:
            continue  # goes elsewhere
        else:
            np.testing.assert_equal(complex_atoms.info[key], value)

    os.unlink('complex.xyz')

#write multiple atoms objects to one xyz
frames = [at, at * (2, 1, 1), at * (3, 1, 1)]
for atoms in frames:
    atoms.write('append.xyz',append=True)
    atoms.write('append.xyz.gz',append=True)
    atoms.write('not_append.xyz',append=False)
readFrames = ase.io.read('append.xyz',index=slice(0,None))
assert readFrames == frames
readFrames = ase.io.read('append.xyz.gz',index=slice(0,None))
assert readFrames == frames
singleFrame = ase.io.read('not_append.xyz',index=slice(0,None))
assert singleFrame[-1] == frames[-1]
os.unlink('append.xyz')
os.unlink('append.xyz.gz')
os.unlink('not_append.xyz')
