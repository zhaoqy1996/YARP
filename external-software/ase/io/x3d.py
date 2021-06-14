"""
Output support for X3D and X3DOM file types.
See http://www.web3d.org/x3d/specifications/
X3DOM outputs to html pages that should display 3-d manipulatable atoms in
modern web browsers.
"""

from __future__ import print_function
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.utils import basestring


def write_x3d(filename, atoms, format=None):
    """Writes to html using X3DOM.
    
    Args: 
        filename - str or file-like object, filename or output file object
        atoms - Atoms object to be rendered
        format - str, either 'X3DOM' for web-browser compatibility or 'X3D'
            to be readable by Blender. `None` to detect format based on file extension
            ('.html' -> 'X3DOM', '.x3d' -> 'X3D')"""
    X3D(atoms).write(filename, datatype=format)

def write_html(filename, atoms):
    """Writes to html using X3DOM
    
    Args: 
        filename - str or file-like object, filename or output file object
        atoms - Atoms object to be rendered"""
    write_x3d(filename, atoms, format='X3DOM')


class X3D:
    """Class to write either X3D (readable by open-source rendering
    programs such as Blender) or X3DOM html, readable by modern web
    browsers.
    """

    def __init__(self, atoms):
        self._atoms = atoms

    def write(self, filename, datatype=None):
        """Writes output to either an 'X3D' or an 'X3DOM' file, based on
        the extension. For X3D, filename should end in '.x3d'. For X3DOM,
        filename should end in '.html'.
        
        Args:
            filename - str or file-like object, output file name or writer
            datatype - str, output format. 'X3D' or 'X3DOM'. If `None`, format 
                will be determined from the filename"""
        
        # Detect the format, if not stated
        if datatype is None:
            if filename.endswith('.x3d'):
                datatype = 'X3D'
            elif filename.endswith('.html'):
                datatype = 'X3DOM'
            else:
                raise ValueError("filename must end in '.x3d' or '.html'.")
                
        # Write the header
        w = WriteToFile(filename, 'w')
        if datatype == 'X3DOM':
            w(0, '<html>')
            w(1, '<head>')
            w(2, '<title>ASE atomic visualization</title>')
            w(2, '<link rel="stylesheet" type="text/css"')
            w(2, ' href="https://www.x3dom.org/x3dom/release/x3dom.css">')
            w(2, '</link>')
            w(2, '<script type="text/javascript"')
            w(2, ' src="https://www.x3dom.org/x3dom/release/x3dom.js">')
            w(2, '</script>')
            w(1, '</head>')
            w(1, '<body>')
            w(2, '<X3D style="margin:0; padding:0; width:100%; height:100%;'
                 ' border:none;">')
        elif datatype == 'X3D':
            w(0, '<?xml version="1.0" encoding="UTF-8"?>')
            w(0, '<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.2//EN" '
              '"http://www.web3d.org/specifications/x3d-3.2.dtd">')
            w(0, '<X3D profile="Interchange" version="3.2" '
              'xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance" '
              'xsd:noNamespaceSchemaLocation='
              '"http://www.web3d.org/specifications/x3d-3.2.xsd">')
        else:
            raise ValueError("datatype not supported: " + str(datatype))

        w(3, '<Scene>')

        for atom in self._atoms:
            for indent, line in atom_lines(atom):
                w(4 + indent, line)

        w(3, '</Scene>')
        
        if datatype == 'X3DOM':
            w(2, '</X3D>')
            w(1, '</body>')
            w(0, '</html>')
        elif datatype == 'X3D':
            w(0, '</X3D>')


class WriteToFile:
    """Creates convenience function to write to a file."""

    def __init__(self, filename, mode='w'):
        if isinstance(filename, basestring):
            self._f = open(filename, mode)
        else:
            self._f = filename

    def __call__(self, indent, line):
        text = ' ' * indent
        print('%s%s\n'%(text,line), file=self._f)

    def close(self):
        self._f.close()


def atom_lines(atom):
    """Generates a segment of X3D lines representing an atom."""
    x, y, z = atom.position
    lines = [(0, '<Transform translation="%.2f %.2f %.2f">' % (x, y, z))]
    lines += [(1, '<Shape>')]
    lines += [(2, '<Appearance>')]
    color = tuple(jmol_colors[atom.number])
    color = 'diffuseColor="%.3f %.3f %.3f"' % color
    lines += [(3, '<Material %s specularColor="0.5 0.5 0.5">' % color)]
    lines += [(3, '</Material>')]
    lines += [(2, '</Appearance>')]
    lines += [(2, '<Sphere radius="%.2f">' % covalent_radii[atom.number])]
    lines += [(2, '</Sphere>')]
    lines += [(1, '</Shape>')]
    lines += [(0, '</Transform>')]
    return lines
