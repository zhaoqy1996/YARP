import numpy as np
from math import sqrt
from ase.utils import rotate
from ase.data import covalent_radii, atomic_numbers
from ase.data.colors import jmol_colors
from ase.utils import basestring


def generate_writer_variables(writer, atoms, rotation='', show_unit_cell=0,
                              radii=None, bbox=None, colors=None, scale=20,
                              maxwidth=500, extra_offset=(0., 0.)):
    writer.numbers = atoms.get_atomic_numbers()
    writer.colors = colors
    if colors is None:
        writer.colors = jmol_colors[writer.numbers]

    if radii is None:
        radii = covalent_radii[writer.numbers]
    elif isinstance(radii, float):
        radii = covalent_radii[writer.numbers] * radii
    else:
        radii = np.array(radii)

    natoms = len(atoms)

    if isinstance(rotation, basestring):
        rotation = rotate(rotation)

    cell = atoms.get_cell()
    disp = atoms.get_celldisp().flatten()

    if show_unit_cell > 0:
        L, T, D = cell_to_lines(writer, cell)
        cell_vertices = np.empty((2, 2, 2, 3))
        for c1 in range(2):
            for c2 in range(2):
                for c3 in range(2):
                    cell_vertices[c1, c2, c3] = np.dot([c1, c2, c3],
                                                       cell) + disp
        cell_vertices.shape = (8, 3)
        cell_vertices = np.dot(cell_vertices, rotation)
    else:
        L = np.empty((0, 3))
        T = None
        D = None
        cell_vertices = None

    nlines = len(L)

    positions = np.empty((natoms + nlines, 3))
    R = atoms.get_positions()
    positions[:natoms] = R
    positions[natoms:] = L

    r2 = radii**2
    for n in range(nlines):
        d = D[T[n]]
        if ((((R - L[n] - d)**2).sum(1) < r2) &
            (((R - L[n] + d)**2).sum(1) < r2)).any():
            T[n] = -1

    positions = np.dot(positions, rotation)
    R = positions[:natoms]

    if bbox is None:
        X1 = (R - radii[:, None]).min(0)
        X2 = (R + radii[:, None]).max(0)
        if show_unit_cell == 2:
            X1 = np.minimum(X1, cell_vertices.min(0))
            X2 = np.maximum(X2, cell_vertices.max(0))
        M = (X1 + X2) / 2
        S = 1.05 * (X2 - X1)
        w = scale * S[0]
        if w > maxwidth:
            w = maxwidth
            scale = w / S[0]
        h = scale * S[1]
        offset = np.array([scale * M[0] - w / 2, scale * M[1] - h / 2, 0])
    else:
        w = (bbox[2] - bbox[0]) * scale
        h = (bbox[3] - bbox[1]) * scale
        offset = np.array([bbox[0], bbox[1], 0]) * scale

    offset[0] = offset[0] - extra_offset[0]
    offset[1] = offset[1] - extra_offset[1]
    writer.w = w + extra_offset[0]
    writer.h = h + extra_offset[1]

    positions *= scale
    positions -= offset

    if nlines > 0:
        D = np.dot(D, rotation)[:, :2] * scale

    if cell_vertices is not None:
        cell_vertices *= scale
        cell_vertices -= offset

    cell = np.dot(cell, rotation)
    cell *= scale

    writer.cell = cell
    writer.positions = positions
    writer.D = D
    writer.T = T
    writer.cell_vertices = cell_vertices
    writer.natoms = natoms
    writer.d = 2 * scale * radii

    # extension for partial occupancies
    writer.frac_occ = False
    writer.tags = None
    writer.occs = None

    try:
        writer.occs = atoms.info['occupancy']
        writer.tags = atoms.get_tags()
        writer.frac_occ = True
    except KeyError:
        pass


def cell_to_lines(writer, cell):
    # XXX this needs to be updated for cell vectors that are zero.
    # Cannot read the code though!  (What are T and D? nn?)
    nlines = 0
    nsegments = []
    for c in range(3):
        d = sqrt((cell[c]**2).sum())
        n = max(2, int(d / 0.3))
        nsegments.append(n)
        nlines += 4 * n

    positions = np.empty((nlines, 3))
    T = np.empty(nlines, int)
    D = np.zeros((3, 3))

    n1 = 0
    for c in range(3):
        n = nsegments[c]
        dd = cell[c] / (4 * n - 2)
        D[c] = dd
        P = np.arange(1, 4 * n + 1, 4)[:, None] * dd
        T[n1:] = c
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            n2 = n1 + n
            positions[n1:n2] = P + i * cell[c - 2] + j * cell[c - 1]
            n1 = n2

    return positions, T, D


def make_patch_list(writer):
    try:
        from matplotlib.path import Path
    except ImportError:
        Path = None
        from matplotlib.patches import Circle, Polygon, Wedge
    else:
        from matplotlib.patches import Circle, PathPatch, Wedge

    indices = writer.positions[:, 2].argsort()
    patch_list = []
    for a in indices:
        xy = writer.positions[a, :2]
        if a < writer.natoms:
            r = writer.d[a] / 2
            if writer.frac_occ:
                site_occ = writer.occs[writer.tags[a]]
                # first an empty circle if a site is not fully occupied
                if (np.sum([v for v in site_occ.values()])) < 1.0:
                    # fill with white
                    fill = '#ffffff'
                    patch = Circle(xy, r, facecolor=fill,
                                   edgecolor='black')
                    patch_list.append(patch)

                start = 0
                # start with the dominant species
                for sym, occ in sorted(site_occ.items(), key=lambda x: x[1], reverse=True):
                    if np.round(occ, decimals=4) == 1.0:
                        patch = Circle(xy, r, facecolor=writer.colors[a],
                                       edgecolor='black')
                        patch_list.append(patch)
                    else:
                        # jmol colors for the moment
                        extent = 360. * occ
                        patch = Wedge(xy, r, start, start+extent,
                                      facecolor=jmol_colors[atomic_numbers[sym]],
                                      edgecolor='black')
                        patch_list.append(patch)
                        start += extent

            else:
                if ((xy[1] + r > 0) and (xy[1] - r < writer.h) and
                    (xy[0] + r > 0) and (xy[0] - r < writer.w)):
                    patch = Circle(xy, r, facecolor=writer.colors[a],
                                   edgecolor='black')
                    patch_list.append(patch)
        else:
            a -= writer.natoms
            c = writer.T[a]
            if c != -1:
                hxy = writer.D[c]
                if Path is None:
                    patch = Polygon((xy + hxy, xy - hxy))
                else:
                    patch = PathPatch(Path((xy + hxy, xy - hxy)))
                patch_list.append(patch)
    return patch_list
