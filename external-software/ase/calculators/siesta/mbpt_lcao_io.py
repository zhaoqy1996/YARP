"""
  Module containing the routine to read the mbpt_lcao output
"""


from __future__ import division
import ase.io as aio
import numpy as np


"""
  class read_mbpt_lcao_output: Main class of the library
          use to load the data from the TDDFT code
  class MBPT_LCAO_Parameters: class that fix the input parameter
          concerning loading data
  class MBPT_LCAO_Properties_figure: class that fix the properties
          concerning the figures
"""


class read_mbpt_lcao_output:
    """
    Top class, to be call by a script.
    Read the output data from the mbpt_lcao program

    Parameters
    ----------
        No input paramters, but the args, and prop variable have
        to be modify as function of your wishes

    References
    ----------

    Example
    -------
    """

    def __init__(self):

        self.args = MBPT_LCAO_Parameters()
        self.prop = MBPT_LCAO_Properties_figure()

    def Read(self, YFname=None):
        if self.args.folder != './' and \
                self.prop.fatoms == 'domiprod_atom2coord.xyz':
            self.prop.fatoms = self.args.folder + self.prop.fatoms
        self.args.check_input()
        if self.args.format_input == 'txt':
            output = read_text_data(self.args, self.prop, YFname)
        elif self.args.format_input == 'hdf5':
            output = read_hdf5_data(self.args, self.prop, YFname)
        else:
            raise ValueError('no other format supported')
        return output


class read_hdf5_data:
    """
        read data from mbpt_lcao calculation saved in the hdf5 format
    """

    def __init__(self, args, prop_fig, YFname):

        try:
            import h5py
        except:
            raise ValueError('The module need h5py library in order \
                        to read hdf5 files!')

        self.atoms = aio.read(prop_fig.fatoms)
        self.determine_fname(args, YFname)

        self.File = h5py.File(self.fname, 'r')

        if args.tem_iter == 'iter':
            self.check_file_iter(args)
            self.extract_data_iter(args)
        elif args.tem_iter == 'tem':
            self.check_file_tem(args)
            self.extract_data_tem(args)
        else:
            raise ValueError('only tem or iter!!')

        self.set_carac(args, prop_fig)

    def set_carac(self, args_p, prop):
        """
            set the ylabel for the plot and other parameters
        """
        if args_p.quantity == 'intensity':
            self.ylabel = r'$|\frac{E}{E_{0}}|^{2}$'
        elif args_p.quantity == 'density':
            if args_p.ReIm == 'im':
                self.ylabel = r'$Im(\delta n)$'
            elif args_p.ReIm == 're':
                self.ylabel = r'$Re(\delta n)$'
        else:
            self.ylabel = 'Intensity (a.u.)'

        self.pl_num = str(prop.plan_coord)
        self.pl_file = prop.plan_coord
        for i in range(len(self.pl_num)):
            if self.pl_num[i] == '.':
                self.pl_file = self.pl_num[0:i] + '-' + \
                    self.pl_num[i + 1:len(self.pl_num)]

    def determine_fname(self, args_p, perso=None):
        """
        determine the files name
        """

        if perso is None:
            self.fname = args_p.folder + 'tddft_' + \
                args_p.tem_iter + '_output.hdf5'
        else:
            self.fname = args_p.folder + perso

        print(self.fname)

    def check_file_tem(self, args_p):
        """
        Check if the file tddft_tem_output.hdf5 caontians the wished data
        """

        field_spatial = ['dens', 'potential', 'intensity', 'efield']
        if args_p.quantity == 'spectrum':
            if args_p.tem_input['tem_method'] == 'C':
                quantity = 'tem_spectrum_cc'
            else:
                quantity = 'tem_spectrum'
        elif args_p.quantity == 'E_loss':
            self.dname = args_p.quantity + '_' + args_p.inter
            quantity = None
        else:
            if args_p.time == 0:
                print(args_p.tem_input['tem_method'])
                if args_p.tem_input['tem_method'] == 'C':
                    quantity = 'field_spatial_cc_freq_{0:.2f}'\
                        .format(args_p.plot_freq)
                else:
                    quantity = 'field_spatial_freq_{0:.2f}'\
                        .format(args_p.plot_freq)
            else:
                quantity = 'field_time'

        print('quantity: ', quantity)
        chain = '_v{0:.8f}_bx{1:.8f}_by{2:.8f}_bz{3:.8f}'\
            .format(args_p.tem_input['vnorm'], args_p.tem_input['b'][0],
                    args_p.tem_input['b'][1], args_p.tem_input['b'][2])
        if quantity is None:
            self.group = self.File['/']
        else:
            self.group = self.File[quantity]
            if args_p.quantity == 'spectrum':
                self.dname = 'tem_spectrum_' + args_p.inter + chain
            elif args_p.quantity in field_spatial or \
                    quantity == 'field_time':
                self.group = self.group['tem_' + args_p.inter + chain]

                if args_p.quantity in field_spatial:
                    if args_p.quantity == 'intensity':
                        self.dname = args_p.quantity
                    else:
                        self.dname = args_p.quantity + '_' + args_p.ReIm
                else:
                    self.dname = []
                    if args_p.quantity == 'intensity':
                        for i in range(args_p.time_num):
                            self.dname.append('data_time_{0}/'
                                              .format(i) + args_p.quantity)
                    else:
                        for i in range(args_p.time_num):
                            self.dname.append(
                                'data_time_{0}/' .format(i) +
                                args_p.quantity + '_' + args_p.ReIm)

        if isinstance(self.dname, list):
            for i, name in enumerate(self.dname):
                print(name)
                sub_group = self.group['data_time_{0}/'.format(i)]
                if args_p.quantity not in sub_group.keys():
                    raise ValueError(
                        name +
                        ' not saved in ' +
                        self.fname +
                        '. check with h5ls command.')
        else:
            if self.dname not in self.group.keys():
                print(self.group.keys())
                raise ValueError(self.dname + ' not saved in ' +
                                 self.fname + '. check with h5ls command.')

    def extract_data_tem(self, args_p):
        """
        Extract the data into the foloowing structures,
        Array: contains the data that you wish to get
        freq: frequency range if polarizability

        geometrical data
        dr: the spatial step
        origin: origin of the system
        lbound: lower bound of the array
        ubound: upper bound of the array
        dim: shape of the array
        mesh: the 2D or 3D mesh for the plotting
        """
        if args_p.quantity == 'spectrum':
            self.Array = self.group[self.dname].value[1, :]
            self.freq = self.group[self.dname].value[0, :]
        elif args_p.quantity == 'E_loss':
            self.Array = self.group[self.dname].value
        else:
            keys_f = ['dr', 'origin', 'ibox']
            keys = ['dr', 'origin', 'ibox']
            dico = {}

            for i, k in enumerate(keys):
                dico[k] = self.group[keys_f[i]].value

            self.dr = dico['dr']
            self.origin = dico['origin']
            self.lbound = dico['ibox'][:, 0]
            self.ubound = dico['ibox'][:, 1]

            if args_p.time == 0:
                keys_f.append(self.dname)

                dico['Array'] = self.group[keys_f[3]].value

                dim = dico['Array'].shape[::-1]
                if args_p.quantity == 'Efield':
                    self.Array = dico['Array'].ravel('F').reshape(
                        dim[0], dim[1], dim[2], dim[3])
                else:
                    self.Array = dico['Array'].T

            else:
                sh = self.group[self.dname[0]].value.shape
                self.t = self.group['t'].value
                if args_p.quantity == 'Efield':
                    self.Array = np.array(
                        (self.t.shape[0], sh[0], sh[1], sh[2], sh[3]),
                        dtype=float)
                    for i in range(len(self.dname)):
                        self.Array[i] = self.group[self.dname[i]].value
                else:
                    self.Array = np.zeros(
                        (self.t.shape[0], sh[0], sh[1], sh[2]), dtype=float)
                    for i in range(len(self.dname)):
                        print('array ', self.Array.shape)
                        print('sum(data) ',
                              np.sum(abs(self.group[self.dname[i]].value)))
                        self.Array[i] = self.group[self.dname[i]].value

            self.determine_box()
            self.mesh3D()
            self.mesh2D()

            self.xy_prof = self.xy_mesh[:, int(self.xy_mesh.shape[1] / 2)]
            self.xz_prof = self.xz_mesh[int(self.xz_mesh.shape[0] / 2), :]

            self.yx_prof = self.yx_mesh[:, int(self.yx_mesh.shape[1] / 2)]
            self.yz_prof = self.yz_mesh[int(self.yz_mesh.shape[0] / 2), :]

            self.zx_prof = self.zx_mesh[:, int(self.zx_mesh.shape[1] / 2)]
            self.zy_prof = self.zy_mesh[int(self.zy_mesh.shape[0] / 2), :]

    def check_file_iter(self, args_p):
        if args_p.quantity == 'polarizability':
            quantity = args_p.quantity
        else:
            quantity = 'field_spatial_freq_{0:.2f}_'.format(
                args_p.plot_freq) + args_p.inter

        if quantity not in self.File.keys():
            raise ValueError(
                quantity +
                ' not saved in ' +
                self.fname +
                '. check with h5ls command.')

        self.group = self.File['/' + quantity]

        if args_p.quantity == 'polarizability':
            self.dname = 'dipol_' + args_p.inter + '_iter_krylov'
        else:
            self.dname = args_p.quantity

        if self.dname == 'intensity':
            if self.dname not in self.group.keys():
                raise ValueError(
                    self.dname +
                    ' not saved in ' +
                    self.fname +
                    '. check with h5ls command.')
        elif self.dname == 'polarizability':
            if args_p.species != '':
                if self.dname + '_' + args_p.ReIm + '_' + \
                        args_p.species not in self.group.keys():
                    raise ValueError(
                        self.dname +
                        '_' +
                        args_p.ReIm +
                        '_' +
                        args_p.species +
                        ' not saved in ' +
                        self.fname +
                        '. check with h5ls command.')
            else:
                if self.dname + '_' + args_p.ReIm not in self.group.keys():
                    raise ValueError(
                        self.dname +
                        '_' +
                        args_p.ReIm +
                        ' not saved in ' +
                        self.fname +
                        '. check with h5ls command.')

        else:
            if self.dname + '_' + args_p.ReIm not in self.group.keys():
                raise ValueError(
                    self.dname +
                    '_' +
                    args_p.ReIm +
                    ' not saved in ' +
                    self.fname +
                    '. check with h5ls command.')

    def extract_data_iter(self, args_p):
        if args_p.quantity == 'polarizability':
            self.freq = self.group['frequency'].value
            if args_p.species != '':
                self.Array = self.group[
                    self.dname +
                    '_' +
                    args_p.ReIm +
                    '_' +
                    args_p.species].value.T
            else:
                self.Array = self.group[self.dname + '_' + args_p.ReIm].value.T
                for i in range(self.Array.shape[0]):
                    self.Array[i, :, :] = self.Array[i, :, :].T
        else:
            if self.dname == 'dens':
                keys_f = [
                    'dr',
                    'origin_dens',
                    'ibox_dens',
                    self.dname +
                    '_' +
                    args_p.ReIm]
            elif self.dname == 'intensity':
                keys_f = ['dr', 'origin', 'ibox', self.dname]
            else:
                keys_f = [
                    'dr',
                    'origin',
                    'ibox',
                    self.dname +
                    '_' +
                    args_p.ReIm]

            keys = ['dr', 'origin', 'ibox', 'Array']
            dico = {}

            for i, k in enumerate(keys):
                dico[k] = self.group[keys_f[i]].value

            self.dr = dico['dr']
            self.origin = dico['origin']
            self.lbound = dico['ibox'][:, 0]
            self.ubound = dico['ibox'][:, 1]

            dim = dico['Array'].shape[::-1]
            if args_p.quantity == 'Efield':
                self.Array = dico['Array'].ravel('F').reshape(
                    dim[0], dim[1], dim[2], dim[3])
            else:
                self.Array = dico['Array'].T

            self.determine_box()
            self.mesh3D()
            self.mesh2D()

            self.xy_prof = self.xy_mesh[:, int(self.xy_mesh.shape[1] / 2)]
            self.xz_prof = self.xz_mesh[int(self.xz_mesh.shape[0] / 2), :]

            self.yx_prof = self.yx_mesh[:, int(self.yx_mesh.shape[1] / 2)]
            self.yz_prof = self.yz_mesh[int(self.yz_mesh.shape[0] / 2), :]

            self.zx_prof = self.zx_mesh[:, int(self.zx_mesh.shape[1] / 2)]
            self.zy_prof = self.zy_mesh[int(self.zy_mesh.shape[0] / 2), :]

    def determine_box(self):
        box = list()
        box.append(self.dr * self.lbound + self.origin)
        box.append(self.dr * self.ubound + self.origin)
        self.box = np.array(box)
        self.dim = self.ubound - self.lbound + 1

    def mesh2D(self):

        self.xy_mesh = np.zeros((self.dim[1], self.dim[2]), dtype=float)
        self.xz_mesh = np.zeros((self.dim[1], self.dim[2]), dtype=float)
        for j in range(self.xy_mesh.shape[1]):
            for i in range(self.xy_mesh.shape[0]):
                self.xy_mesh[i, j] = self.box[0][1] + \
                    i * self.dr[1] + self.origin[1]

        for i in range(self.xz_mesh.shape[0]):
            for j in range(self.xz_mesh.shape[1]):
                self.xz_mesh[i, j] = self.box[0][2] + \
                    j * self.dr[2] + self.origin[2]

        self.yx_mesh = np.zeros((self.dim[0], self.dim[2]), dtype=float)
        self.yz_mesh = np.zeros((self.dim[0], self.dim[2]), dtype=float)
        for j in range(self.yx_mesh.shape[1]):
            for i in range(self.yx_mesh.shape[0]):
                self.yx_mesh[i, j] = self.box[0][0] + \
                    i * self.dr[0] + self.origin[0]

        for i in range(self.yz_mesh.shape[0]):
            for j in range(self.yz_mesh.shape[1]):
                self.yz_mesh[i, j] = self.box[0][2] + \
                    j * self.dr[2] + self.origin[2]

        self.zx_mesh = np.zeros((self.dim[0], self.dim[1]), dtype=float)
        self.zy_mesh = np.zeros((self.dim[0], self.dim[1]), dtype=float)
        for j in range(self.zx_mesh.shape[1]):
            for i in range(self.zx_mesh.shape[0]):
                self.zx_mesh[i, j] = self.box[0][0] + \
                    i * self.dr[0] + self.origin[0]

        for i in range(self.zy_mesh.shape[0]):
            for j in range(self.zy_mesh.shape[1]):
                self.zy_mesh[i, j] = self.box[0][1] + \
                    j * self.dr[1] + self.origin[1]

    def mesh3D(self):
        self.xmesh = np.zeros(
            (self.dim[0], self.dim[1], self.dim[2]), dtype=float)
        self.ymesh = np.zeros(
            (self.dim[0], self.dim[1], self.dim[2]), dtype=float)
        self.zmesh = np.zeros(
            (self.dim[0], self.dim[1], self.dim[2]), dtype=float)

        for i in range(self.xmesh.shape[0]):
            nb = self.box[0][0] + i * self.dr[0] + self.origin[0]
            self.xmesh[i, :, :] = nb

        for i in range(self.xmesh.shape[1]):
            nb = self.box[0][1] + i * self.dr[1] + self.origin[1]
            self.ymesh[:, i, :] = nb

        for i in range(self.xmesh.shape[2]):
            nb = self.box[0][2] + i * self.dr[2] + self.origin[2]
            self.zmesh[:, :, i] = nb


class read_text_data:
    """
    Class that read the output data of the tddft program for the
    field enhancement.
    can read .dat files (text file), .npy files (binary files) or
    .hdf5 files (binary).
    it is loading all the different parameter in array to plotting purpose.
    Input parameters:
    -----------------
    args: (class parameter), input parameter
    prop_fig: (class Properties_figure): Properties of the figure
    kwargs (optionnal):
      new_data: (list of args class!), list of the other args if needed
      to get more than one data
    Output parameters:
    ------------------
    all the data about the box save in:
    self.dr
    self.box
    self.Array
    self.dim
    self.mesh.....
    Function of the class:
    ----------------------
    initialise
    determine_fname
    recover_data
    readhdf5
    read_npy
    read_txt
    determine_box
    mesh3D
    """

    def __init__(self, args, prop_fig, YFname):

        if args.tem_iter == 'tem':
            raise ValueError('text format only with iter')

        self.atoms = aio.read(prop_fig.fatoms)
        self.fname = self.determine_fname(args, perso=YFname)
        self.dr, self.origin, self.lbound, self.ubound, self.Array, \
            self.box, self.dim = self.read_txt(args, self.fname)

        if args.quantity != 'polarizability':

            self.mesh3D()
            self.mesh2D()

            self.xy_prof = self.xy_mesh[:, int(self.xy_mesh.shape[1] / 2)]
            self.xz_prof = self.xz_mesh[int(self.xz_mesh.shape[0] / 2), :]

            self.yx_prof = self.yx_mesh[:, int(self.yx_mesh.shape[1] / 2)]
            self.yz_prof = self.yz_mesh[int(self.yz_mesh.shape[0] / 2), :]

            self.zx_prof = self.zx_mesh[:, int(self.zx_mesh.shape[1] / 2)]
            self.zy_prof = self.zy_mesh[int(self.zy_mesh.shape[0] / 2), :]

        self.set_carac(args, prop_fig)

    def set_carac(self, args_p, prop):
        if args_p.quantity == 'intensity':
            self.ylabel = r'$|\frac{E}{E_{0}}|^{2}$'
        elif args_p.quantity == 'density':
            if args_p.ReIm == 'im':
                self.ylabel = r'$Im(\delta n)$'
            elif args_p.ReIm == 're':
                self.ylabel = r'$Re(\delta n)$'
        else:
            self.ylabel = 'Intensity (a.u.)'

        self.pl_num = str(prop.plan_coord)
        self.pl_file = prop.plan_coord
        for i in range(len(self.pl_num)):
            if self.pl_num[i] == '.':
                self.pl_file = self.pl_num[0:i] + '-' + \
                    self.pl_num[i + 1:len(self.pl_num)]

    def determine_fname(self, args_p, perso=None):
        """
        set the files name
        """
        if perso is None:
            if args_p.quantity == 'Efield':
                fname = [
                    args_p.folder +
                    'e_field_' +
                    args_p.ReIm +
                    '.x_' +
                    args_p.inter +
                    args_p.tem_iter +
                    '.dat',
                    args_p.folder +
                    'e_field_' +
                    args_p.ReIm +
                    '.y_' +
                    args_p.inter +
                    args_p.tem_iter +
                    '.dat',
                    args_p.folder +
                    'e_field_' +
                    args_p.ReIm +
                    '.z_' +
                    args_p.inter +
                    args_p.tem_iter +
                    '.dat']
            elif args_p.quantity == 'intensity':
                fname = args_p.folder + args_p.quantity + '_' + \
                    args_p.inter + '_' + args_p.tem_iter + '.dat'
            elif args_p.quantity == 'polarizability':
                fname = args_p.folder + 'dipol_' + args_p.inter + '_' + \
                    args_p.tem_iter + '_krylov_' + args_p.ReIm + '.txt'
            else:
                fname = args_p.folder + args_p.quantity + '_' + args_p.ReIm + \
                    '_' + args_p.inter + '_' + args_p.tem_iter + '.dat'
        else:
            fname = args_p.folder + perso

        print(fname)
        return fname

    def read_txt(self, args_p, fname):
        from mbpt_lcao_utils import read_file, str2float
        data = list()
        dim = list()
        end_box = 10

        if args_p.quantity == 'Efield':
            A = []
            for i in enumerate(fname):
                LINE = read_file(i[1])

                for j in range(end_box):
                    if LINE[j][1] != '#':
                        nb = str2float(LINE[j])
                        data.append(nb)

                dr = np.array(data[0])
                origin = np.array(data[1])
                lbound = np.array(data[2])
                ubound = np.array(data[3])
                box, dim = self.determine_box(dr, ubound, lbound, origin)

                A.append(np.zeros((dim[0], dim[1], dim[2]), dtype=float))

                l = end_box
                for k in range(int(dim[2])):
                    for j in range(int(dim[1])):
                        A[i[0]][:, j, k] = np.array(str2float(LINE[l]))
                        l = l + 1

            Array = np.zeros(
                (A[0].shape[0],
                 A[0].shape[1],
                    A[0].shape[2],
                    3),
                dtype=float)
            Array[:, :, :, 0] = A[0]
            Array[:, :, :, 1] = A[1]
            Array[:, :, :, 2] = A[2]

        elif args_p.quantity == 'polarizability':
            self.freq = np.loadtxt(fname)[:, 0]
            Array = np.loadtxt(fname)[:, 2:11].reshape(
                self.freq.shape[0], 3, 3)
            dr = 0.0
            origin = 0.0
            lbound = 0.0
            ubound = 0.0
            box = 0.0
            dim = 0.0
        else:
            LINE = read_file(fname)

            for i in range(end_box):
                if LINE[i][1] != '#':
                    nb = str2float(LINE[i])
                    data.append(nb)

            dr = np.array(data[0])
            origin = np.array(data[1])
            lbound = np.array(data[2])
            ubound = np.array(data[3])
            box, dim = self.determine_box(dr, ubound, lbound, origin)

            Array = np.zeros((dim[0], dim[1], dim[2]), dtype=float)

            l = end_box
            for k in range(int(dim[2])):
                for j in range(int(dim[1])):
                    Array[:, j, k] = np.array(str2float(LINE[l]))
                    l = l + 1

        return dr, origin, lbound, ubound, Array, box, dim

    def determine_box(self, dr, ubound, lbound, origin):
        box = list()
        box.append(dr * lbound + origin)
        box.append(dr * ubound + origin)

        dim = ubound - lbound + 1

        return box, dim

    def mesh2D(self):

        self.xy_mesh = np.zeros((self.dim[1], self.dim[2]), dtype=float)
        self.xz_mesh = np.zeros((self.dim[1], self.dim[2]), dtype=float)
        for j in range(self.xy_mesh.shape[1]):
            for i in range(self.xy_mesh.shape[0]):
                self.xy_mesh[i, j] = self.box[0][1] + \
                    i * self.dr[1] + self.origin[1]

        for i in range(self.xz_mesh.shape[0]):
            for j in range(self.xz_mesh.shape[1]):
                self.xz_mesh[i, j] = self.box[0][2] + \
                    j * self.dr[2] + self.origin[2]

        self.yx_mesh = np.zeros((self.dim[0], self.dim[2]), dtype=float)
        self.yz_mesh = np.zeros((self.dim[0], self.dim[2]), dtype=float)
        for j in range(self.yx_mesh.shape[1]):
            for i in range(self.yx_mesh.shape[0]):
                self.yx_mesh[i, j] = self.box[0][0] + \
                    i * self.dr[0] + self.origin[0]

        for i in range(self.yz_mesh.shape[0]):
            for j in range(self.yz_mesh.shape[1]):
                self.yz_mesh[i, j] = self.box[0][2] + \
                    j * self.dr[2] + self.origin[2]

        self.zx_mesh = np.zeros((self.dim[0], self.dim[1]), dtype=float)
        self.zy_mesh = np.zeros((self.dim[0], self.dim[1]), dtype=float)
        for j in range(self.zx_mesh.shape[1]):
            for i in range(self.zx_mesh.shape[0]):
                self.zx_mesh[i, j] = self.box[0][0] + \
                    i * self.dr[0] + self.origin[0]

        for i in range(self.zy_mesh.shape[0]):
            for j in range(self.zy_mesh.shape[1]):
                self.zy_mesh[i, j] = self.box[0][1] + \
                    j * self.dr[1] + self.origin[1]

    def mesh3D(self):
        self.xmesh = np.zeros(
            (self.dim[0], self.dim[1], self.dim[2]), dtype=float)
        self.ymesh = np.zeros(
            (self.dim[0], self.dim[1], self.dim[2]), dtype=float)
        self.zmesh = np.zeros(
            (self.dim[0], self.dim[1], self.dim[2]), dtype=float)

        for i in range(self.xmesh.shape[0]):
            nb = self.box[0][0] + i * self.dr[0] + self.origin[0]
            self.xmesh[i, :, :] = nb

        for i in range(self.xmesh.shape[1]):
            nb = self.box[0][1] + i * self.dr[1] + self.origin[1]
            self.ymesh[:, i, :] = nb

        for i in range(self.xmesh.shape[2]):
            nb = self.box[0][2] + i * self.dr[2] + self.origin[2]
            self.zmesh[:, :, i] = nb


class MBPT_LCAO_Parameters:
    """
    Contains the input parameters use by plot_3D in plot_lib.py
    The parameters are:
    self.quantity (string, default:'polarizability'): the type of data
    than one wish to plot, can be
                                      intensity
                                      efield
                                      potential
                                      dens
                                      polarizability
                                      spectrum
                                      E_loss
    self.interacting (int, default: 0) interacting or non-interacting data,
                                      depend of your TDDFT calculation
    self.ReIm (string, default: 'im'): plot imaginary or real part
    self.format_input (string, default:'.hdf5'): format of the input file,
    can be
                                                .dat
                                                .npy
                                                .hdf5
    self.folder (string, default: './'): name of the folder where are
                                         save the input data
    self.time (int, default: 1): time plotting, 0 or 1
    self.time_num (int, default: 0): time index than one which to plot
    self.tem_iter (int, default: 0) : tem or iter plotting, 0 => iter, 1 => tem
    self.movie (int, default: 0) : if tem and plot_dens_time=1
                                   (in tddft_lr.inp) then do a movie
    """

    def __init__(self):
        self.quantity = 'polarizability'
        self.inter = 'inter'
        self.ReIm = 'im'
        self.format_input = 'hdf5'
        self.folder = './'
        self.time = 0
        self.time_num = 0
        self.tem_iter = 'iter'
        self.species = ''
        self.movie = 0
        self.plot_freq = 0.0
        self.tem_input = {'dr': np.array([0.3,
                                          0.3,
                                          0.3]),
                          'vnorm': 1.0,
                          'v': np.array([1.0,
                                         0.0,
                                         0.0]),
                          'dw': 0.1,
                          'b': np.array([0.0,
                                         0.0,
                                         0.0]),
                          'vrange': None,
                          'brange': None,
                          'tem_method': 'N'}
        self.exportData = {
            'export': False,
            'dtype': 'HDF5',
            'fname': 'exportData'}
        self.kwargs = {}  # to add more arguments

    def check_input(self):
        """
        Check the validity of the arguments
        """
        self.param = {'quantity': [self.quantity, str],
                      'inter': [self.inter, str],
                      'ReIm': [self.ReIm, str],
                      'format_inp': [self.format_input, str],
                      'folder_inp': [self.folder, str],
                      'time': [self.time, int],
                      'time_num': [self.time_num, int],
                      'tem_iter': [self.tem_iter, str],
                      'species': [self.species, str],
                      'movie': [self.movie, int],
                      'tem_input': [self.tem_input, dict],
                      'exportData': [self.exportData, dict]}

        fields = {'quantity': ['intensity', 'efield', 'potential', 'dens',
                               'polarizability', 'spectrum', 'E_loss'],
                  'ReIm': ['re', 'im'],
                  'format_input': ['txt', 'hdf5'],
                  'tem_iter': ['tem', 'iter']}

        for keys, values in self.param.items():
            if not isinstance(values[0], values[1]):
                raise ValueError('Error: input ' + keys +
                                 ' not right type, must be ' + str(values[1]))

            if keys in fields.keys():
                if values[0] not in fields[keys]:
                    raise ValueError(
                        keys + ' can be only: ' + str(fields[keys]))

                if (values[0] == 'spectrum' or values[0] ==
                        'E_loss') and self.tem_iter != 'tem':
                    raise ValueError('spectrum only with tem')

                if values[0] == 'polarizability' and self.tem_iter != 'iter':
                    raise ValueError('polarizability only with iter')


class MBPT_LCAO_Properties_figure:
    """
    class that define the caracteristic of your Figures.
    Parameters:
    -----------
    self.fontsize (float, default: 30): fontsize of the labels
    self.axis_size (list of float, default: [20, 20]): fontsize of the tickle
    self.folder (string, default: 'images/'): folders where
                                              are save the pictures
    self.figx = 16
    self.figy = 12
    self.figsize (tuple, default: (self.figx, self.figy)): size of
                                        the figure (width, heigth)
    self.fatoms (string, default:'domiprod_atom2coord.xyz'): name of the
                                        file for the atomic positions
    self.plot_atoms (bolleen, default: True): plotting atoms or not
    self.color_arrow = 'red'
    self.ft_arrow_label = 30
    self.arrow_label = r'$E_{ext}$'
    self.title = 'none'
    self.linewidth = 3
    self.linecolor = 'red'
    self.plan (string, default: 'z'): plan than one wish to plot, can be
                                      x
                                      y
                                      z
    self.dynamic (int, default: 0): available only for Mayavi plot
                                        make the plot rotating, only 0 or 1
    self.show (int, default: 1): show the plot, 0 or 1
    self.plan_coord (float, default: 0.0): coordinate of the plan than one wish
                                          to plot in Ang
    self.output (string, default: 'pdf'): format of the output file, can be
                                        pdf
                                        png
                                        ps
                                        eps
                                        svg
    self.animation (int, default: 0): save a movie of the animation made
                                      by self.dynamic only available with
                                      Mayavi plot and if self.dynamic=1
    self.Edir (int, default: 0): plot direction of the E field, 0 or 1
    self.coord_Ef (2D numpy array default:'default') plotting
                                            coordinate E field??
    self.plot (string, default:'2D'): define the plotting method, can be
                                      1D
                                      2D
                                      3D
                                      Mayavi
    self.coord (1D numpy array, default: np.array([0.0, 0.0, 0.0]))
                                        coordinate for curve in 1D
    self.figname (string, default:'default.'): name of the figure
    """

    def __init__(self):
        import matplotlib.cm as cm

        self.fontsize = 30
        self.axis_size = [20, 20]
        self.folder = 'images/'
        self.figx = 16
        self.figy = 12
        self.figsize = (self.figx, self.figy)
        self.fatoms = 'domiprod_atom2coord.xyz'
        self.plot_atm = True
        self.color_arrow = 'red'
        self.ft_arrow_label = 30
        self.arrow_label = r'$E_{ext}$'
        self.title = None
        self.linewidth = 3
        self.linecolor = 'red'
        self.plan_coord = 0.0
        self.plan = 'z'
        self.dynamic = 0
        self.show = 1
        self.output = 'pdf'
        self.animation = 0
        self.Edir = 0
        self.coord_Ef = 'default'
        self.plot = '2D'
        self.coord = np.array([0.0, 0.0, 0.0])
        self.bohr_rad = 0.52917721
        self.figname = 'default'
        self.cmap = cm.jet
        self.units = 'au'
        self.interpolation = "bicubic"
        # for the moment only the polarizability can be modify in nm**2
        self.vmin = None
        self.vmax = None
        # to save a mayavi scrennshot in order to perform subplot with
        # matplotlib
        self.mayavi_screenshot = 0

        self.maya_prop = {
            'extent_factor': 1.0,
            'figsize': (
                640,
                480),
            'contours': 3,
            'atoms_resolution': 8,
            'atoms_scale': 1,
            'fps': 20,
            'opacity': 0.5,
            'line_width': 2.0,
            'magnification': 1}
        self.maya_cam = {
            'distance': None,
            'azimuth': None,
            'elevation': None,
            'roll': None,
            'reset_roll': True,
            'figure': None,
            'focalpoint': 'auto'}
