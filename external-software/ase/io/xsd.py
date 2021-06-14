import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

from ase import Atoms


def read_xsd(fd):
    tree = ET.parse(fd)
    root = tree.getroot()

    atomtreeroot = root.find('AtomisticTreeRoot')
    # if periodic system
    if atomtreeroot.find('SymmetrySystem') is not None:
        symmetrysystem = atomtreeroot.find('SymmetrySystem')
        mappingset = symmetrysystem.find('MappingSet')
        mappingfamily = mappingset.find('MappingFamily')
        system = mappingfamily.find('IdentityMapping')

        coords = list()
        cell = list()
        formula = str()

        for atom in system:
            if atom.tag == 'Atom3d':
                symbol = atom.get('Components')
                formula += symbol

                xyz = atom.get('XYZ')
                if xyz:
                    coord = [float(coord) for coord in xyz.split(',')]
                else:
                    coord = [0.0, 0.0, 0.0]
                coords.append(coord)
            elif atom.tag == 'SpaceGroup':
                avec = [float(vec) for vec in atom.get('AVector').split(',')]
                bvec = [float(vec) for vec in atom.get('BVector').split(',')]
                cvec = [float(vec) for vec in atom.get('CVector').split(',')]

                cell.append(avec)
                cell.append(bvec)
                cell.append(cvec)

        atoms = Atoms(formula, cell=cell, pbc=True)
        atoms.set_scaled_positions(coords)
        return atoms
        # if non-periodic system
    elif atomtreeroot.find('Molecule') is not None:
        system = atomtreeroot.find('Molecule')

        coords = list()
        formula = str()

        for atom in system:
            if atom.tag == 'Atom3d':
                symbol = atom.get('Components')
                formula += symbol

                xyz = atom.get('XYZ')
                coord = [float(coord) for coord in xyz.split(',')]
                coords.append(coord)

        atoms = Atoms(formula, pbc=False)
        atoms.set_scaled_positions(coords)
        return atoms


def CPK_or_BnS(element):
    """Determine how atom is visualized"""
    if element in ['C', 'H', 'O', 'S', 'N']:
        visualization_choice = 'Ball and Stick'
    else:
        visualization_choice = 'CPK'
    return visualization_choice


def write_xsd(fd, atoms, connectivity=None):
    """Takes Atoms object, and write materials studio file
    atoms: Atoms object
    filename: path of the output file
    connectivity: number of atoms by number of atoms matrix for connectivity
    between atoms (0 not connected, 1 connected)

    note: material studio file cannot use a partial periodic system. If partial
    perodic system was inputted, full periodicity was assumed.
    """

    natoms = atoms.get_number_of_atoms()
    atom_element = atoms.get_chemical_symbols()
    atom_cell = atoms.get_cell()
    atom_positions = atoms.get_positions()

    XSD = ET.Element('XSD')
    XSD.set('Version', '6.0')

    AtomisticTreeRootElement = ET.SubElement(XSD, 'AtomisticTreeRoot')
    AtomisticTreeRootElement.set('ID', '1')
    AtomisticTreeRootElement.set('NumProperties', '40')
    AtomisticTreeRootElement.set('NumChildren', '1')

    Property1 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property1.set('DefinedOn', 'ClassicalEnergyHolder')
    Property1.set('Name', 'AngleEnergy')
    Property1.set('Type', 'Double')

    Property2 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property2.set('DefinedOn', 'ClassicalEnergyHolder')
    Property2.set('Name', 'BendBendEnergy')
    Property2.set('Type', 'Double')

    Property3 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property3.set('DefinedOn', 'ClassicalEnergyHolder')
    Property3.set('Name', 'BendTorsionBendEnergy')
    Property3.set('Type', 'Double')

    Property4 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property4.set('DefinedOn', 'ClassicalEnergyHolder')
    Property4.set('Name', 'BondEnergy')
    Property4.set('Type', 'Double')

    Property5 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property5.set('DefinedOn', 'Atom')
    Property5.set('Name', 'EFGAsymmetry')
    Property5.set('Type', 'Double')

    Property6 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property6.set('DefinedOn', 'Atom')
    Property6.set('Name', 'EFGQuadrupolarCoupling')
    Property6.set('Type', 'Double')

    Property7 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property7.set('DefinedOn', 'ClassicalEnergyHolder')
    Property7.set('Name', 'ElectrostaticEnergy')
    Property7.set('Type', 'Double')

    Property8 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property8.set('DefinedOn', 'GrowthFace')
    Property8.set('Name', 'FaceMillerIndex')
    Property8.set('Type', 'MillerIndex')

    Property9 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property9.set('DefinedOn', 'GrowthFace')
    Property9.set('Name', 'FacetTransparency')
    Property9.set('Type', 'Float')

    Property10 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property10.set('DefinedOn', 'Bondable')
    Property10.set('Name', 'Force')
    Property10.set('Type', 'CoDirection')

    Property11 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property11.set('DefinedOn', 'ClassicalEnergyHolder')
    Property11.set('Name', 'HydrogenBondEnergy')
    Property11.set('Type', 'Double')

    Property12 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property12.set('DefinedOn', 'Bondable')
    Property12.set('Name', 'ImportOrder')
    Property12.set('Type', 'UnsignedInteger')

    Property13 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property13.set('DefinedOn', 'ClassicalEnergyHolder')
    Property13.set('Name', 'InversionEnergy')
    Property13.set('Type', 'Double')

    Property14 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property14.set('DefinedOn', 'Atom')
    Property14.set('Name', 'IsBackboneAtom')
    Property14.set('Type', 'Boolean')

    Property15 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property15.set('DefinedOn', 'Atom')
    Property15.set('Name', 'IsChiralCenter')
    Property15.set('Type', 'Boolean')

    Property16 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property16.set('DefinedOn', 'Atom')
    Property16.set('Name', 'IsOutOfPlane')
    Property16.set('Type', 'Boolean')

    Property17 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property17.set('DefinedOn', 'BestFitLineMonitor')
    Property17.set('Name', 'LineExtentPadding')
    Property17.set('Type', 'Double')

    Property18 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property18.set('DefinedOn', 'Linkage')
    Property18.set('Name', 'LinkageGroupName')
    Property18.set('Type', 'String')

    Property19 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property19.set('DefinedOn', 'PropertyList')
    Property19.set('Name', 'ListIdentifier')
    Property19.set('Type', 'String')

    Property20 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property20.set('DefinedOn', 'Atom')
    Property20.set('Name', 'NMRShielding')
    Property20.set('Type', 'Double')

    Property21 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property21.set('DefinedOn', 'ClassicalEnergyHolder')
    Property21.set('Name', 'NonBondEnergy')
    Property21.set('Type', 'Double')

    Property22 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property22.set('DefinedOn', 'Bondable')
    Property22.set('Name', 'NormalMode')
    Property22.set('Type', 'Direction')

    Property23 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property23.set('DefinedOn', 'Bondable')
    Property23.set('Name', 'NormalModeFrequency')
    Property23.set('Type', 'Double')

    Property24 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property24.set('DefinedOn', 'Bondable')
    Property24.set('Name', 'OrbitalCutoffRadius')
    Property24.set('Type', 'Double')

    Property25 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property25.set('DefinedOn', 'BestFitPlaneMonitor')
    Property25.set('Name', 'PlaneExtentPadding')
    Property25.set('Type', 'Double')

    Property26 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property26.set('DefinedOn', 'ClassicalEnergyHolder')
    Property26.set('Name', 'PotentialEnergy')
    Property26.set('Type', 'Double')

    Property27 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property27.set('DefinedOn', 'ScalarFieldBase')
    Property27.set('Name', 'QuantizationValue')
    Property27.set('Type', 'Double')

    Property28 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property28.set('DefinedOn', 'ClassicalEnergyHolder')
    Property28.set('Name', 'RestraintEnergy')
    Property28.set('Type', 'Double')

    Property29 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property29.set('DefinedOn', 'ClassicalEnergyHolder')
    Property29.set('Name', 'SeparatedStretchStretchEnergy')
    Property29.set('Type', 'Double')

    Property30 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property30.set('DefinedOn', 'Trajectory')
    Property30.set('Name', 'SimulationStep')
    Property30.set('Type', 'Integer')

    Property31 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property31.set('DefinedOn', 'ClassicalEnergyHolder')
    Property31.set('Name', 'StretchBendStretchEnergy')
    Property31.set('Type', 'Double')

    Property32 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property32.set('DefinedOn', 'ClassicalEnergyHolder')
    Property32.set('Name', 'StretchStretchEnergy')
    Property32.set('Type', 'Double')

    Property33 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property33.set('DefinedOn', 'ClassicalEnergyHolder')
    Property33.set('Name', 'StretchTorsionStretchEnergy')
    Property33.set('Type', 'Double')

    Property34 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property34.set('DefinedOn', 'ClassicalEnergyHolder')
    Property34.set('Name', 'TorsionBendBendEnergy')
    Property34.set('Type', 'Double')

    Property35 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property35.set('DefinedOn', 'ClassicalEnergyHolder')
    Property35.set('Name', 'TorsionEnergy')
    Property35.set('Type', 'Double')

    Property36 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property36.set('DefinedOn', 'ClassicalEnergyHolder')
    Property36.set('Name', 'TorsionStretchEnergy')
    Property36.set('Type', 'Double')

    Property37 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property37.set('DefinedOn', 'ClassicalEnergyHolder')
    Property37.set('Name', 'ValenceCrossTermEnergy')
    Property37.set('Type', 'Double')

    Property38 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property38.set('DefinedOn', 'ClassicalEnergyHolder')
    Property38.set('Name', 'ValenceDiagonalEnergy')
    Property38.set('Type', 'Double')

    Property39 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property39.set('DefinedOn', 'ClassicalEnergyHolder')
    Property39.set('Name', 'VanDerWaalsEnergy')
    Property39.set('Type', 'Double')

    Property40 = ET.SubElement(AtomisticTreeRootElement, 'Property')
    Property40.set('DefinedOn', 'SymmetrySystem')
    Property40.set('Name', '_Stress')
    Property40.set('Type', 'Matrix')
    # Set up bonds
    bonds = list()
    if connectivity is not None:
        for i in range(0,connectivity.shape[0]):
            for j in range(i+1,connectivity.shape[0]):
                if connectivity[i,j]:
                    bonds.append([i,j])

    # non-periodic system
    if not atoms.pbc.all():
        Molecule = ET.SubElement(AtomisticTreeRootElement, 'Molecule')
        Molecule.set('ID', '2')
        Molecule.set('NumChildren', str(natoms+len(bonds)))
        Molecule.set('Name', 'Lattice=&quot1.0')

        # writing atoms
        for x in range(0, natoms):
            NewAtom = ET.SubElement(Molecule, 'Atom3d')
            NewAtom.set('ID', str(x + 3))
            NewAtom.set('Name', (atom_element[x] + str(x + 1)))
            NewAtom.set('UserID', str(x + 1))
            NewAtom.set('DisplayStyle', CPK_or_BnS(atom_element[x]))
            tmpstr = ''
            for y in range(3):
                tmpstr += '%1.16f,' % atom_positions[x, y]
            NewAtom.set('XYZ', tmpstr[0:-1])
            NewAtom.set('Components', atom_element[x])
            tmpstr = ''
            for ibond in range(0,len(bonds)):
                if x in bonds[ibond]:
                    tmpstr += '%i,' % (ibond + 3 + natoms)
            if tmpstr != '':
                NewAtom.set('Connections', tmpstr[0:-1])
        for x in range(0, len(bonds)):
            NewBond = ET.SubElement(Molecule, 'Bond')
            NewBond.set('ID', str(x + 3 + natoms))
            tmpstr = '%i,%i'%(bonds[x][0] + 3,bonds[x][1] + 3)
            NewBond.set('Connects', tmpstr)
    # periodic system
    else:
        atom_positions = np.dot(atom_positions, np.linalg.inv(atom_cell))
        SymmSys = ET.SubElement(AtomisticTreeRootElement, 'SymmetrySystem')
        SymmSys.set('ID', '2')
        SymmSys.set('Mapping', '3')
        tmpstr = ''
        for x in range(4, natoms + len(bonds) + 4):
            tmpstr += '%1.0f,' % (x)
        tmpstr += str(natoms + len(bonds) + 4)
        SymmSys.set('Children', tmpstr)
        SymmSys.set('Normalized', '1')
        SymmSys.set('Name', 'SymmSys')
        SymmSys.set('UserID', str(natoms + 18))
        SymmSys.set('XYZ',
                    '0.00000000000000,0.00000000000000,0.000000000000000')
        SymmSys.set('OverspecificationTolerance', '0.05')
        SymmSys.set('PeriodicDisplayType', 'Original')

        MappngSet = ET.SubElement(SymmSys, 'MappingSet')
        MappngSet.set('ID', str(natoms + len(bonds) + 5))
        MappngSet.set('SymmetryDefinition', str(natoms + 4))
        MappngSet.set('ActiveSystem', '2')
        MappngSet.set('NumFamilies', '1')
        MappngSet.set('OwnsTotalConstraintMapping', '1')
        MappngSet.set('TotalConstraintMapping', '3')

        MappngFamily = ET.SubElement(MappngSet, 'MappingFamily')
        MappngFamily.set('ID', str(natoms + len(bonds) + 6))
        MappngFamily.set('NumImageMappings', '0')

        IdentMappng = ET.SubElement(MappngFamily, 'IdentityMapping')
        IdentMappng.set('ID', str(natoms + len(bonds) + 7))
        IdentMappng.set('Element', '1,0,0,0,0,1,0,0,0,0,1,0')
        IdentMappng.set('Constraint', '1,0,0,0,0,1,0,0,0,0,1,0')
        tmpstr = ''
        for x in range(4, natoms + len(bonds) + 4):
            tmpstr += '%1.0f,' % (x)
        IdentMappng.set('MappedObjects', tmpstr[0:-1])
        tmpstr = str(natoms + len(bonds) + 4) + ',' + str(natoms + len(bonds) + 8)
        IdentMappng.set('DefectObjects', tmpstr)
        IdentMappng.set('NumImages', str(natoms + len(bonds)))
        IdentMappng.set('NumDefects', '2')

        MappngRepairs = ET.SubElement(MappngFamily, 'MappingRepairs')
        MappngRepairs.set('NumRepairs', '0')

        # writing atoms
        for x in range(natoms):
            NewAtom = ET.SubElement(IdentMappng, 'Atom3d')
            NewAtom.set('ID', str(x + 4))
            NewAtom.set('Mapping', str(natoms + len(bonds) + 7))
            NewAtom.set('Parent', '2')
            NewAtom.set('Name', (atom_element[x] + str(x + 1)))
            NewAtom.set('UserID', str(x + 1))
            NewAtom.set('DisplayStyle', CPK_or_BnS(atom_element[x]))
            tmpstr = ''
            for y in range(3):
                tmpstr += '%1.16f,' % atom_positions[x, y]
            NewAtom.set('XYZ', tmpstr[0:-1])
            NewAtom.set('Components', atom_element[x])
            tmpstr = ''
            for ibond in range(0,len(bonds)):
                if x in bonds[ibond]:
                    tmpstr += '%i,' % (ibond + 4 + natoms + 1)
            if tmpstr != '':
                NewAtom.set('Connections', tmpstr[0:-1])
        for x in range(0, len(bonds)):
            NewBond = ET.SubElement(IdentMappng, 'Bond')
            NewBond.set('ID', str(x + 4 + natoms + 1))
            NewBond.set('Mapping', str(natoms + len(bonds) + 7))
            NewBond.set('Parent', '2')
            tmpstr = '%i,%i'%(bonds[x][0] + 4,bonds[x][1] + 4)
            NewBond.set('Connects', tmpstr)

        SpaceGrp = ET.SubElement(IdentMappng, 'SpaceGroup')
        SpaceGrp.set('ID', str(natoms + 4))
        SpaceGrp.set('Parent', '2')
        SpaceGrp.set('Children', str(natoms + len(bonds) + 8))
        SpaceGrp.set('DisplayStyle', 'Solid')
        SpaceGrp.set('XYZ', '0.00,0.00,0.00')
        SpaceGrp.set('Color', '0,0,0,0')
        tmpstr = ''
        for x in range(3):
            tmpstr += '%1.16f,' % atom_cell[0, x]
        SpaceGrp.set('AVector', tmpstr[0:-1])
        tmpstr = ''
        for x in range(3):
            tmpstr += '%1.16f,' % atom_cell[1, x]
        SpaceGrp.set('BVector', tmpstr[0:-1])
        tmpstr = ''
        for x in range(3):
            tmpstr += '%1.16f,' % atom_cell[2, x]
        SpaceGrp.set('CVector', tmpstr[0:-1])
        SpaceGrp.set('OrientationBase', 'C along Z, B in YZ plane')
        SpaceGrp.set('Centering', '3D Primitive-Centered')
        SpaceGrp.set('Lattice', '3D Triclinic')
        SpaceGrp.set('GroupName', 'GroupName')
        SpaceGrp.set('Operators', '1,0,0,0,0,1,0,0,0,0,1,0')
        SpaceGrp.set('DisplayRange', '0,1,0,1,0,1')
        SpaceGrp.set('LineThickness', '2')
        SpaceGrp.set('CylinderRadius', '0.2')
        SpaceGrp.set('LabelAxes', '1')
        SpaceGrp.set('ActiveSystem', '2')
        SpaceGrp.set('ITNumber', '1')
        SpaceGrp.set('LongName', 'P 1')
        SpaceGrp.set('Qualifier', 'Origin-1')
        SpaceGrp.set('SchoenfliesName', 'C1-1')
        SpaceGrp.set('System', 'Triclinic')
        SpaceGrp.set('Class', '1')

        RecLattc = ET.SubElement(IdentMappng, 'ReciprocalLattice3D')
        RecLattc.set('ID', str(natoms + len(bonds) + 8))
        RecLattc.set('Parent', str(natoms + 4))

        InfiniteMappng = ET.SubElement(MappngSet, 'InfiniteMapping')
        InfiniteMappng.set('ID', '3')
        InfiniteMappng.set('Element', '1,0,0,0,0,1,0,0,0,0,1,0')
        InfiniteMappng.set('MappedObjects', '2')

    # Return a pretty-printed XML string for the Element.
    rough_string = ET.tostring(XSD, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    Document = reparsed.toprettyxml(indent='\t')

    fd.write(Document)
