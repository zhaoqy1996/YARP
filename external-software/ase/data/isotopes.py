"""Isotope data extracted from NIST public website.

Source data has been compiled by NIST:

    https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses

The atomic weights data were published in:

    J. Meija et al, Atomic weights of the elements 2013,
    Pure and Applied Chemistry 88, 265-291 (2016).
    https://doi.org/10.1515/pac-2015-0305
    http://www.ciaaw.org/atomic-weights.htm

Isotopic compositions data were published in:

    Michael Berglund and Michael E. Wieser,
    Isotopic compositions of the elements 2009 (IUPAC Technical Report)
    Pure Appl. Chem., 2011, Vol. 83, No. 2, pp. 397-410
    http://dx.doi.org/10.1351/PAC-REP-10-06-02

The relative atomic masses of the isotopes data were published in:

    M. Wang, G. Audi, A.H. Wapstra, F.G. Kondev, M. MacCormick, X. Xu,
    and B. Pfeiffer, The AME2012 Atomic Mass Evaluation,
    Chinese Phys. C 36 1603
    http://dx.doi.org/10.1088/1674-1137/36/12/003
    http://amdc.impcas.ac.cn/evaluation/data2012/ame.html
"""


def download_isotope_data():
    """Download isotope data from NIST public website.

    Relative atomic masses of individual isotopes their abundance
    (mole fraction) are compiled into a dictionary. Individual items can be
    indexed by the atomic number and mass number, e.g. titanium-48:

    >>> from ase.data.isotopes import extract_isotope_data
    >>> isotopes = extract_isotope_data()
    >>> isotopes[22][48]['mass']
    47.94794198
    >>> isotopes[22][48]['composition']
    0.7372
    """
    import requests

    raw_data = requests.get(
        'http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl'
        '?ele=&ascii=ascii&isotype=all').content.decode().splitlines()

    indexes = [idx for (idx, line) in enumerate(raw_data) if "_____" in line]

    isotopes = {}
    for idx1, idx2 in zip(indexes, indexes[1:]):
        atomic_number = int(raw_data[idx1 + 1].split()[0])
        isotopes[atomic_number] = dct = {}
        for isotope_idx in range(idx1+1, idx2):
            mass_number = int(raw_data[isotope_idx][8:12])
            # drop uncertainty
            mass = float(raw_data[isotope_idx][13:31].split('(')[0])
            try:
                composition = float(raw_data[isotope_idx][32:46].split('(')[0])
            except ValueError:
                composition = 0.0
            dct[mass_number] = {'mass': mass, 'composition': composition}

    return isotopes
