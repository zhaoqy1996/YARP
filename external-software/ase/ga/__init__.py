"""Functions that are important for the genetic algorithm.
Shorthand for setting and getting
- the raw_score
- the neighbor_list
- the parametrization
of an atoms object.
"""


def set_raw_score(atoms, raw_score):
    """Set the raw_score of an atoms object in the
    atoms.info['key_value_pairs'] dictionary.
    
    Parameters
    ----------
    atoms : Atoms object
        The atoms object that corresponds to this raw_score
    raw_score : float or int
        Independent calculation of how fit the candidate is.
    """
    if 'key_value_pairs' not in atoms.info:
        atoms.info['key_value_pairs'] = {}
    atoms.info['key_value_pairs']['raw_score'] = raw_score


def get_raw_score(atoms):
    """Gets the raw_score of the supplied atoms object.

    Parameters
    ----------
    atoms : Atoms object
        The atoms object from which the raw_score will be returned.

    Returns
    -------
    raw_score : float or int
        The raw_score set previously.
    """
    return atoms.info['key_value_pairs']['raw_score']

    
def set_parametrization(atoms, parametrization):
    if 'data' not in atoms.info:
        atoms.info['data'] = {}
    atoms.info['data']['parametrization'] = parametrization

    
def get_parametrization(atoms):
    if 'parametrization' in atoms.info['data']:
        return atoms.info['data']['parametrization']
    else:
        raise ValueError('Trying to get the parametrization before it is set!')

        
def set_neighbor_list(atoms, neighbor_list):
    if 'data' not in atoms.info:
        atoms.info['data'] = {}
    atoms.info['data']['neighborlist'] = neighbor_list

    
def get_neighbor_list(atoms):
    if 'neighborlist' in atoms.info['data']:
        return atoms.info['data']['neighborlist']
    else:
        return None
