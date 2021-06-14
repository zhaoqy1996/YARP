from __future__ import division
import numpy as np
import re


def read_file(fname):
    """
    read the file fname and return a list of the lines.
    """
    f = open(fname, 'r')
    LINE = list()

    for line in f:
        LINE.append(line)

    return LINE


def delete_blanc(L):
    """
    delete the blank space from a string
    """
    compt = 0
    while L[compt] == ' ' or L[compt] == '\t':
        compt = compt + 1
        L = L[compt:len(L)]
    return L


def read_number(L):
    compt = 0

    while L[compt] != ' ' and compt < (len(L) - 1):
        compt = compt + 1

        nb1 = float(L[0:compt + 1])
        L = L[compt:len(L)]

    return nb1, L


def recover_data_string(fname, string):
    L = read_file(fname)

    for i in L:
        print(i[0:len(string)], string)
        if i[0:len(string)] == string:
            v = str2float(i)
    return v


def save_line(Lnb, Cnb, LINE):
    number = LINE[Lnb]
    nombre = list()

    for i in range(Cnb):
        number = delete_blanc(number)
        nb1, number = read_number(number)
        nombre.append(nb1)

    return nombre


def dim_y(Cnb, L):

    if len(L) == Cnb * 10 + Cnb * 6 + 1:
        nb_col = Cnb
    else:
        nb_col = Cnb + 1

    return nb_col


def read_color_file(fname):

    L = read_file(fname)

    atom = list()

    for i in range(len(L)):
        nb = np.array(str2float(L[i]))
        species = recover_species(L[i])
        atom.append([species, nb])

    return atom


def readSiestaFA(fname):
    L = read_file(fname)

    Forces = []
    for i in range(1, len(L)):
        Forces.append([i, np.array(str2float(L[i])[1:4])])

    return Forces


def readBasis_spec(fname, nb_species):
    """
    Example Basis_specs from siesta output
    <basis_specs>
    ===============================================================================
    H                    Z=   1    Mass=  1.0100        Charge= 0.17977+309
    Lmxo=0 Lmxkb= 2    BasisType=split      Semic=F
    L=0  Nsemic=0  Cnfigmx=1
             n=1  nzeta=2  polorb=1
               splnorm:   0.15000
                  vcte:    0.0000
                  rinn:    0.0000
                   rcs:    0.0000      0.0000
               lambdas:    1.0000      1.0000
    -------------------------------------------------------------------------------
    L=0  Nkbl=1  erefs: 0.17977+309
    L=1  Nkbl=1  erefs: 0.17977+309
    L=2  Nkbl=1  erefs: 0.17977+309
    ===============================================================================
    </basis_specs>
    """

    L = read_file(fname)

    species_charac = {}
    line = 0
    len_basis = len('<basis_specs>')
    i = 0
    while i < nb_species:
        if L[line][0:len_basis] == '<basis_specs>':
            i = i + 1
            info = str2float(L[line + 2])
            if L[line + 2][1] == ' ':
                species_charac[
                    L[line + 2][0]] = {'Z': info[0],
                                       'Mass': info[1], 'Charge': info[2]}
            else:
                species_charac[
                    L[line + 2][0:2]] = {'Z': info[0],
                                         'Mass': info[1], 'Charge': info[2]}

            while L[line][0:len('</basis_specs>')] != '</basis_specs>':
                line = line + 1
        else:
            line = line + 1

    return species_charac


def str2float(string):
    numeric_const_pattern = r"""
  [-+]? # optional sign
  (?:
    (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
    |
    (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
  )
  # followed by optional exponent part if desired
  (?: [Ee] [+-]? \d+ ) ?
  """
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    nb = rx.findall(string)
    for i in enumerate(nb):
        nb[i[0]] = float(i[1])

    return np.array(nb)


def str2int(string):
    numeric_const_pattern = r"""
  [-+]? # optional sign
  (?:
    (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
    |
    (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
  )
  # followed by optional exponent part if desired
  (?: [Ee] [+-]? \d+ ) ?
  """
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    nb = rx.findall(string)
    for i in enumerate(nb):
        nb[i[0]] = int(i[1])

    return np.array(nb)


def recover_species(string):
    """
    Select species in a string of caractere from
    a .xyz file
    Input parameters:
      string (str): the string to analyse
    Output parameter:
      string_p (str): the specie
    """

    species = list()
    comp = 0
    letter = string[0]
    if letter == ' ':
        while letter == ' ' or comp >= len(string):
            letter = string[comp]
            comp = comp + 1
        while letter != ' ' or comp >= len(string):
            letter = string[comp]
            species.append(letter)
            comp = comp + 1
    else:
        while letter != ' ' or comp >= len(string):
            letter = string[comp]
            species.append(letter)
            comp = comp + 1

    species.remove(' ')

    string_p = ''
    for i in species:
        string_p = string_p + i

    return string_p


def delete_number_string(string):
    # not working for exponential expression
    nb_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-']
    L = list()

    for i in string:
        L.append(i)

    L_P = list()
    for i in enumerate(L):
        inside = False
        for j in nb_list:
            if i[1] == j:
                inside = True

        if not inside:
            L_P.append(i[1])

    string_p = ''
    for i in L_P:
        if i != ' ':
            string_p = string_p + i

    return string_p


def pol2cross_sec(p, omg):
    """
    Convert the polarizability in au to cross section in nm**2
    INPUT PARAMETERS:
    -----------------
    p (np array): polarizability from mbpt_lcao calc
    omg (np.array): frequency range in eV
    OUTPUT_PARAMETERS:
    ------------------
    sigma (np array): cross section in nm**2
    """
    c = 137  # speed of the light in au
    omg = omg * 0.036749309  # to convert from eV to Hartree

    sigma = 4 * np.pi * omg * p / (c)  # bohr**2
    sigma = sigma * (0.052917725)**2  # nm**2

    return sigma


def interpolate(x, y, nb_pts):
    """
    perform a 1D spline interpolation.
    INPUT PARAMETERS
    ----------------
    x (1D np array) : the original abscisse
    y (1D np array) : the original data
    nb_pts (integer): the number of points for the interpolation
    OUTPUT PARAMETERS
    -----------------
    xnew (1D np array) : the spline abscisse
    ynew (1D np array) : the spline approximations
    """
    from scipy import interpolate
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.linspace(x[0], x[x.shape[0] - 1], nb_pts)
    ynew = interpolate.splev(xnew, tck, der=0)
    return xnew, ynew
