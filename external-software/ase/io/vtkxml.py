import numpy as np


fast = False

def write_vti(filename, atoms, data=None):
    from vtk import vtkStructuredPoints, vtkDoubleArray, vtkXMLImageDataWriter

    #if isinstance(fileobj, basestring):
    #    fileobj = paropen(fileobj, 'w')

    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError('Can only write one configuration to a VTI file!')
        atoms = atoms[0]

    if data is None:
        raise ValueError('VTK XML Image Data (VTI) format requires data!')

    data = np.asarray(data)

    if data.dtype == complex:
        data = np.abs(data)

    cell = atoms.get_cell()

    assert np.all(cell==np.diag(cell.diagonal())), 'Unit cell must be orthogonal'

    bbox = np.array(list(zip(np.zeros(3),cell.diagonal()))).ravel()

    # Create a VTK grid of structured points
    spts = vtkStructuredPoints()
    spts.SetWholeBoundingBox(bbox)
    spts.SetDimensions(data.shape)
    spts.SetSpacing(cell.diagonal() / data.shape)
    #spts.SetSpacing(paw.gd.h_c * Bohr)

    #print 'paw.gd.h_c * Bohr=',paw.gd.h_c * Bohr
    #print 'atoms.cell.diagonal() / data.shape=', cell.diagonal()/data.shape
    #assert np.all(paw.gd.h_c * Bohr==cell.diagonal()/data.shape)

    #s = paw.wfs.kpt_u[0].psit_nG[0].copy()
    #data = paw.get_pseudo_wave_function(band=0, kpt=0, spin=0, pad=False)
    #spts.point_data.scalars = data.swapaxes(0,2).flatten()
    #spts.point_data.scalars.name = 'scalars'

    # Allocate a VTK array of type double and copy data
    da = vtkDoubleArray()
    da.SetName('scalars')
    da.SetNumberOfComponents(1)
    da.SetNumberOfTuples(np.prod(data.shape))

    for i,d in enumerate(data.swapaxes(0,2).flatten()):
        da.SetTuple1(i,d)

    # Assign the VTK array as point data of the grid
    spd = spts.GetPointData() # type(spd) is vtkPointData
    spd.SetScalars(da)

    """
    from vtk.util.vtkImageImportFromArray import vtkImageImportFromArray
    iia = vtkImageImportFromArray()
    #iia.SetArray(Numeric_asarray(data.swapaxes(0,2).flatten()))
    iia.SetArray(Numeric_asarray(data))
    ida = iia.GetOutput()
    ipd = ida.GetPointData()
    ipd.SetName('scalars')
    spd.SetScalars(ipd.GetScalars())
    """

    # Save the ImageData dataset to a VTK XML file.
    w = vtkXMLImageDataWriter()

    if fast:
        w.SetDataModeToAppend()
        w.EncodeAppendedDataOff()
    else:
        w.SetDataModeToAscii()

    w.SetFileName(filename)
    w.SetInput(spts)
    w.Write()


def write_vtu(filename, atoms, data=None):
    from vtk import VTK_MAJOR_VERSION, vtkUnstructuredGrid, vtkPoints, vtkXMLUnstructuredGridWriter
    from vtk.util.numpy_support import numpy_to_vtk

    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError('Can only write one configuration to a VTI file!')
        atoms = atoms[0]

    # Create a VTK grid of structured points
    ugd = vtkUnstructuredGrid()

    # add atoms as vtk Points
    p = vtkPoints()
    p.SetNumberOfPoints(len(atoms))
    p.SetDataTypeToDouble()
    for i,pos in enumerate(atoms.get_positions()):
        p.InsertPoint(i,pos[0],pos[1],pos[2])
    ugd.SetPoints(p)

    # add atomic numbers
    numbers = numpy_to_vtk(atoms.get_atomic_numbers(), deep=1)
    ugd.GetPointData().AddArray(numbers)
    numbers.SetName("atomic numbers")

    # add tags
    tags = numpy_to_vtk(atoms.get_tags(), deep=1)
    ugd.GetPointData().AddArray(tags)
    tags.SetName("tags")

    # add covalent radii
    from ase.data import covalent_radii
    radii = numpy_to_vtk(np.array([covalent_radii[i] for i in atoms.get_atomic_numbers()]), deep=1)
    ugd.GetPointData().AddArray(radii)
    radii.SetName("radii")

    # Save the UnstructuredGrid dataset to a VTK XML file.
    w = vtkXMLUnstructuredGridWriter()

    if fast:
        w.SetDataModeToAppend()
        w.EncodeAppendedDataOff()
    else:
        w.GetCompressor().SetCompressionLevel(0)
        w.SetDataModeToAscii()

    if isinstance(filename, str):
        w.SetFileName(filename)
    else:
        w.SetFileName(filename.name)
    if VTK_MAJOR_VERSION <= 5:
        w.SetInput(ugd)
    else:
        w.SetInputData(ugd)
    w.Write()
