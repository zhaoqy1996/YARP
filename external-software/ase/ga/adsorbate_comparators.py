"""Comparator objects relevant to particles with adsorbates."""

from ase import Atoms
        

def count_ads(atoms, adsorbate):
    """Very naive implementation only taking into account
    the symbols. atoms and adsorbate should both be supplied
    as Atoms objects."""
    syms = atoms.get_chemical_symbols()
    try:
        ads_syms = adsorbate.get_chemical_symbols()
    except AttributeError:
        # It is hopefully a string
        ads_syms = Atoms(adsorbate).get_chemical_symbols()

    counts = []
    for c in ads_syms:
        counts.append(syms.count(c))
        if len(set(counts)) == 1:
            return counts[0]
        else:
            raise NotImplementedError


class AdsorbateCountComparator(object):
    """Compares the number of adsorbates on the particles and
    returns True if the numbers are the same, False otherwise.
    
    Parameters:

    adsorbate: list or string
    a supplied list of adsorbates or a string if only one adsorbate
    is possible
    """
    def __init__(self, adsorbate):
        try:
            adsorbate + ''
            # It is a string (or similar) type
            self.adsorbate = [adsorbate]
        except TypeError:
            self.adsorbate = adsorbate

    def looks_like(self, a1, a2):
        """Does the actual comparison."""
        for ads in self.adsorbate:
            ads = Atoms(ads)
            if count_ads(a1, ads) != count_ads(a2, ads):
                return False
        return True


class AdsorptionSitesComparator(object):
    """Compares the metal atoms in the adsorption sites and returns True
    if less than min_diff_adsorption_sites of the sites with adsorbates
    consist of different atoms.

    Ex:
    a1.info['data']['adsorbates_site_atoms'] =
    [('Cu','Ni'),('Cu','Ni'),('Ni'),('Ni')]

    a2.info['data']['adsorbates_site_atoms'] =
    [('Cu','Ni'),('Ni','Ni', 'Ni'),('Ni'),('Ni')]

    will have a difference of 2:
    (2*('Cu','Ni')-1*('Cu','Ni')=1, 1*('Ni','Ni','Ni')=1, 2*('Ni')-2*('Ni')=0)

    """
    def __init__(self, min_diff_adsorption_sites=2):
        self.min_diff_adsorption_sites = min_diff_adsorption_sites

    def looks_like(self, a1, a2):
        s = 'adsorbates_site_atoms'
        if not all([(s in a.info['data'] and
                     a.info['data'][s] != [])
                    for a in [a1, a2]]):
            return False

        counter = {}
        for asa in a1.info['data'][s]:
            t_asa = tuple(sorted(asa))
            if t_asa not in counter.keys():
                counter[t_asa] = 1
            else:
                counter[t_asa] += 1

        for asa in a2.info['data'][s]:
            t_asa = tuple(sorted(asa))
            if t_asa not in counter.keys():
                counter[t_asa] = -1
            else:
                counter[t_asa] -= 1

        # diffs = len([k for k, v in counter.items() if v != 0])
        sumdiffs = sum([abs(v) for k, v in counter.items()])

        if sumdiffs < self.min_diff_adsorption_sites:
            return True

        return False


class AdsorptionMetalsComparator(object):
    """Compares the number of adsorbate-metal bonds and returns True if the
    number for a1 and a2 differs by less than the supplied parameter
    ``same_adsorption_number``

    Ex:
    a1.info['data']['adsorbates_bound_to'] = {'Cu':1, 'Ni':3}
    a2.info['data']['adsorbates_bound_to'] = {'Cu':.5, 'Ni':3.5}
    will have a difference of .5 in both elements:
    """
    def __init__(self, same_adsorption_number):
        self.same_adsorption_number = same_adsorption_number

    def looks_like(self, a1, a2):
        s = 'adsorbates_bound_to'
        if not all([(s in a.info['data'] and
                     any(a.info['data'][s].values()))
                    for a in [a1, a2]]):
            return False

        diffs = [a1.info['data'][s][k] - a2.info['data'][s][k]
                 for k in a1.info['data'][s].keys()]
        for d in diffs:
            if abs(d) < self.same_adsorption_number:
                return True
        return False
