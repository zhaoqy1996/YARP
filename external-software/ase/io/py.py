
def write_py(fileobj, images):
    """Write to ASE-compatible python script."""
    fileobj.write('from ase import Atoms\n\n')
    fileobj.write('import numpy as np\n\n')
    
    if hasattr(images, 'get_positions'):
        images = [images]
    fileobj.write('images = [\n')

    for image in images:
        fileobj.write("    Atoms(symbols='%s',\n"
                      "          pbc=np.%s,\n"
                      "          cell=np.array(\n      %s,\n"
                      "          positions=np.array(\n      %s),\n" % (
                          image.get_chemical_formula(mode='reduce'),
                          repr(image.pbc),
                          repr(image.cell)[6:],
                          repr(image.positions)[6:]))
        
    fileobj.write(']')
