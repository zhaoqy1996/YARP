from math import pi, sin, cos
import numpy as np


def bz_vertices(icell):
    from scipy.spatial import Voronoi
    I = (np.indices((3, 3, 3)) - 1).reshape((3, 27))
    G = np.dot(icell.T, I).T
    vor = Voronoi(G)
    bz1 = []
    for vertices, points in zip(vor.ridge_vertices, vor.ridge_points):
        if -1 not in vertices and 13 in points:
            normal = G[points].sum(0)
            normal /= (normal**2).sum()**0.5
            bz1.append((vor.vertices[vertices], normal))
    return bz1


def bz3d_plot(cell, vectors=False, paths=None, points=None,
              elev=None, scale=1, interactive=False):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib.patches import FancyArrowPatch
    Axes3D  # silence pyflakes

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    icell = np.linalg.inv(cell).T
    kpoints = points
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')

    azim = pi / 5
    elev = elev or pi / 6
    x = sin(azim)
    y = cos(azim)
    view = [x * cos(elev), y * cos(elev), sin(elev)]

    bz1 = bz_vertices(icell)

    maxp = 0.0
    for points, normal in bz1:
        if np.dot(normal, view) < 0 and not interactive:
            ls = ':'
        else:
            ls = '-'
        x, y, z = np.concatenate([points, points[:1]]).T
        ax.plot(x, y, z, c='k', ls=ls)
        maxp = max(maxp, points.max())

    if vectors:
        ax.add_artist(Arrow3D([0, icell[0, 0]],
                              [0, icell[0, 1]],
                              [0, icell[0, 2]],
                              mutation_scale=20, lw=1,
                              arrowstyle='-|>', color='k'))
        ax.add_artist(Arrow3D([0, icell[1, 0]],
                              [0, icell[1, 1]],
                              [0, icell[1, 2]],
                              mutation_scale=20, lw=1,
                              arrowstyle='-|>', color='k'))
        ax.add_artist(Arrow3D([0, icell[2, 0]],
                              [0, icell[2, 1]],
                              [0, icell[2, 2]],
                              mutation_scale=20, lw=1,
                              arrowstyle='-|>', color='k'))
        maxp = max(maxp, 0.6 * icell.max())

    if paths is not None:
        for names, points in paths:
            x, y, z = np.array(points).T
            ax.plot(x, y, z, c='r', ls='-')

            for name, point in zip(names, points):
                x, y, z = point
                if name == 'G':
                    name = '\\Gamma'
                elif len(name) > 1:
                    name = name[0] + '_' + name[1]
                ax.text(x, y, z, '$' + name + '$',
                        ha='center', va='bottom', color='r')

    if kpoints is not None:
        for p in kpoints:
            ax.scatter(p[0], p[1], p[2], c='b')

    ax.set_axis_off()
    ax.autoscale_view(tight=True)
    s = maxp / 0.5 * 0.45 * scale
    ax.set_xlim(-s, s)
    ax.set_ylim(-s, s)
    ax.set_zlim(-s, s)
    ax.set_aspect('equal')

    ax.view_init(azim=azim / pi * 180, elev=elev / pi * 180)


def bz2d_plot(cell, vectors=False, paths=None, points=None):
    import matplotlib.pyplot as plt
    # 2d in x-y plane
    assert all(abs(cell[2][0:2]) < 1e-6) and all(abs(cell.T[2][0:2]) < 1e-6)

    icell = np.linalg.inv(cell).T
    kpoints = points
    ax = plt.axes()

    bz1 = bz_vertices(icell)

    maxp = 0.0
    for points, normal in bz1:
        x, y, z = np.concatenate([points, points[:1]]).T
        ax.plot(x, y, c='k', ls='-')
        maxp = max(maxp, points.max())

    if vectors:
        ax.arrow(0, 0, icell[0, 0], icell[0, 1],
                 lw=1, color='k',
                 length_includes_head=True,
                 head_width=0.03, head_length=0.05)
        ax.arrow(0, 0, icell[1, 0], icell[1, 1],
                 lw=1, color='k',
                 length_includes_head=True,
                 head_width=0.03, head_length=0.05)
        maxp = max(maxp, icell.max())

    if paths is not None:
        for names, points in paths:
            x, y, z = np.array(points).T
            ax.plot(x, y, c='r', ls='-')

            for name, point in zip(names, points):
                x, y, z = point
                if name == 'G':
                    name = '\\Gamma'
                elif len(name) > 1:
                    name = name[0] + '_' + name[1]
                if abs(z) < 1e-6:
                    ax.text(x, y, '$' + name + '$',
                            ha='center', va='bottom', color='r')

    if kpoints is not None:
        for p in kpoints:
            ax.scatter(p[0], p[1], c='b')

    ax.set_axis_off()
    ax.autoscale_view(tight=True)
    s = maxp * 1.05
    ax.set_xlim(-s, s)
    ax.set_ylim(-s, s)
    ax.set_aspect('equal')


def bz1d_plot(cell, vectors=False, paths=None, points=None):
    import matplotlib.pyplot as plt
    # 1d in x
    assert (all(abs(cell[2][0:2]) < 1e-6) and
            all(abs(cell.T[2][0:2]) < 1e-6) and
            abs(cell[0][1]) < 1e-6 and abs(cell[1][0]) < 1e-6)

    icell = np.linalg.inv(cell).T
    kpoints = points
    ax = plt.axes()

    maxp = 0.0
    x = np.array([-0.5 * icell[0, 0], 0.5 * icell[0, 0], -0.5 * icell[0, 0]])
    y = np.array([0, 0, 0])
    ax.plot(x, y, c='k', ls='-')
    maxp = icell[0, 0]

    if vectors:
        ax.arrow(0, 0, icell[0, 0], 0,
                 lw=1, color='k',
                 length_includes_head=True,
                 head_width=0.03, head_length=0.05)
        maxp = max(maxp, icell.max())

    if paths is not None:
        for names, points in paths:
            x, y, z = np.array(points).T
            ax.plot(x, y, c='r', ls='-')

            for name, point in zip(names, points):
                x, y, z = point
                if name == 'G':
                    name = '\\Gamma'
                elif len(name) > 1:
                    name = name[0] + '_' + name[1]
                if abs(y) < 1e-6 and abs(z) < 1e-6:
                    ax.text(x, y, '$' + name + '$',
                            ha='center', va='bottom', color='r')

    if kpoints is not None:
        for p in kpoints:
            ax.scatter(p[0], 0, c='b')

    ax.set_axis_off()
    ax.autoscale_view(tight=True)
    s = maxp * 1.05
    ax.set_xlim(-s, s)
    ax.set_ylim(-s, s)
    ax.set_aspect('equal')
