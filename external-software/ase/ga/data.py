"""
    Objects which handle all communication with the SQLite database.
"""
import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db


def split_description(desc):
    """ Utility method for string splitting. """
    d = desc.split(':')
    assert len(d) == 2, desc
    return d[0], d[1]


def test_raw_score(atoms):
    """Test that raw_score can be extracted."""
    err_msg = "raw_score not put in atoms.info['key_value_pairs']"
    assert 'raw_score' in atoms.info['key_value_pairs'], err_msg


class DataConnection(object):
    """Class that handles all database communication.

    All data communication is collected in this class in order to
    make a decoupling of the data representation and the GA method.

    A new candidate must be added with one of the functions
    add_unrelaxed_candidate or add_relaxed_candidate this will correctly
    initialize a configuration id used to keep track of candidates in the
    database.
    After one of the add_*_candidate functions have been used, if the candidate
    is further modified or relaxed the functions add_unrelaxed_step or
    add_relaxed_step must be used. This way the configuration id carries
    through correctly.

    Parameters:

    db_file_name: Path to the ase.db data file.
    """

    def __init__(self, db_file_name):
        self.db_file_name = db_file_name
        if not os.path.isfile(self.db_file_name):
            raise IOError('DB file {0} not found'.format(self.db_file_name))
        self.c = ase.db.connect(self.db_file_name)
        self.already_returned = set()

    def get_number_of_unrelaxed_candidates(self):
        """ Returns the number of candidates not yet queued or relaxed. """
        return len(self.__get_ids_of_all_unrelaxed_candidates__())

    def get_an_unrelaxed_candidate(self):
        """ Returns a candidate ready for relaxation. """
        to_get = self.__get_ids_of_all_unrelaxed_candidates__()
        if len(to_get) == 0:
            raise ValueError('No unrelaxed candidate to return')

        a = self.__get_latest_traj_for_confid__(to_get[0])
        a.info['confid'] = to_get[0]
        if 'data' not in a.info:
            a.info['data'] = {}
        return a

    def get_all_unrelaxed_candidates(self):
        """Return all unrelaxed candidates,
        useful if they can all be evaluated quickly."""
        to_get = self.__get_ids_of_all_unrelaxed_candidates__()
        if len(to_get) == 0:
            return []
        res = []
        for confid in to_get:
            a = self.__get_latest_traj_for_confid__(confid)
            a.info['confid'] = confid
            if 'data' not in a.info:
                a.info['data'] = {}
            res.append(a)
        return res

    def __get_ids_of_all_unrelaxed_candidates__(self):
        """ Helper method used by the two above methods. """

        all_unrelaxed_ids = set([t.gaid for t in self.c.select(relaxed=0)])
        all_relaxed_ids = set([t.gaid for t in self.c.select(relaxed=1)])
        all_queued_ids = set([t.gaid for t in self.c.select(queued=1)])

        actually_unrelaxed = [gaid for gaid in all_unrelaxed_ids
                              if (gaid not in all_relaxed_ids and
                                  gaid not in all_queued_ids)]

        return actually_unrelaxed

    def __get_latest_traj_for_confid__(self, confid):
        """ Method for obtaining the latest traj
            file for a given configuration.
            There can be several traj files for
            one configuration if it has undergone
            several changes (mutations, pairings, etc.)."""
        allcands = list(self.c.select(gaid=confid))
        allcands.sort(key=lambda x: x.mtime)
        # return self.get_atoms(all[-1].gaid)
        return self.get_atoms(allcands[-1].id)

    def mark_as_queued(self, a):
        """ Marks a configuration as queued for relaxation. """
        gaid = a.info['confid']
        self.c.write(None, gaid=gaid, queued=1,
                     key_value_pairs=a.info['key_value_pairs'])

#         if not np.array_equal(a.numbers, self.atom_numbers):
#             raise ValueError('Wrong stoichiometry')
#         self.c.write(a, gaid=gaid, queued=1)

    def add_relaxed_step(self, a, find_neighbors=None,
                         perform_parametrization=None):
        """After a candidate is relaxed it must be marked
        as such. Use this function if the candidate has already been in the
        database in an unrelaxed version, i.e. add_unrelaxed_candidate has
        been used.

        Neighbor list and parametrization parameters to screen
        candidates before relaxation can be added. Default is not to use.
        """
        # test that raw_score can be extracted
        err_msg = "raw_score not put in atoms.info['key_value_pairs']"
        assert 'raw_score' in a.info['key_value_pairs'], err_msg

        #  confid has already been set in add_unrelaxed_candidate
        gaid = a.info['confid']

        if 'generation' not in a.info['key_value_pairs']:
            g = self.get_generation_number()
            a.info['key_value_pairs']['generation'] = g

        if find_neighbors is not None:
            set_neighbor_list(a, find_neighbors(a))
        if perform_parametrization is not None:
            set_parametrization(a, perform_parametrization(a))

        relax_id = self.c.write(a, relaxed=1, gaid=gaid,
                                key_value_pairs=a.info['key_value_pairs'],
                                data=a.info['data'])
        a.info['relax_id'] = relax_id

    def add_relaxed_candidate(self, a, find_neighbors=None,
                              perform_parametrization=None):
        """After a candidate is relaxed it must be marked
        as such. Use this function if the candidate has *not* been in the
        database in an unrelaxed version, i.e. add_unrelaxed_candidate has
        *not* been used.

        Neighbor list and parametrization parameters to screen
        candidates before relaxation can be added. Default is not to use.
        """
        test_raw_score(a)

        if 'generation' not in a.info['key_value_pairs']:
            g = self.get_generation_number()
            a.info['key_value_pairs']['generation'] = g

        if find_neighbors is not None:
            set_neighbor_list(a, find_neighbors(a))
        if perform_parametrization is not None:
            set_parametrization(a, perform_parametrization(a))

        relax_id = self.c.write(a, relaxed=1,
                                key_value_pairs=a.info['key_value_pairs'],
                                data=a.info['data'])
        self.c.update(relax_id, gaid=relax_id)
        a.info['confid'] = relax_id
        a.info['relax_id'] = relax_id

    def add_more_relaxed_steps(self, a_list):
        # This function will be removed soon as the function name indicates
        # that unrelaxed candidates are added beforehand
        print('Please use add_more_relaxed_candidates instead')
        self.add_more_relaxed_candidates(a_list)

    def add_more_relaxed_candidates(self, a_list):
        """Add more relaxed candidates quickly"""
        for a in a_list:
            try:
                a.info['key_value_pairs']['raw_score']
            except KeyError:
                print("raw_score not put in atoms.info['key_value_pairs']")

        g = self.get_generation_number()

        # Insert gaid by getting the next available id and assuming that the
        # entire a_list will be written without interuption
        next_id = self.get_next_id()
        with self.c as con:
            for j, a in enumerate(a_list):
                if 'generation' not in a.info['key_value_pairs']:
                    a.info['key_value_pairs']['generation'] = g

                gaid = next_id + j
                relax_id = con.write(a, relaxed=1, gaid=gaid,
                                     key_value_pairs=a.info['key_value_pairs'],
                                     data=a.info['data'])
                assert gaid == relax_id
                a.info['confid'] = relax_id
                a.info['relax_id'] = relax_id

    def get_next_id(self):
        """Get the id of the next candidate to be added to the database.
        This is a hacky way of obtaining the id and it only works on a
        sqlite database.
        """
        con = self.c._connect()
        last_id = self.c.get_last_id(con.cursor())
        con.close()
        return last_id + 1

    def get_largest_in_db(self, var):
        return next(self.c.select(sort='-{0}'.format(var))).get(var)

    def add_unrelaxed_candidate(self, candidate, description):
        """ Adds a new candidate which needs to be relaxed. """
        t, desc = split_description(description)
        kwargs = {'relaxed': 0,
                  'extinct': 0,
                  t: 1,
                  'description': desc}

        if 'generation' not in candidate.info['key_value_pairs']:
            kwargs.update({'generation': self.get_generation_number()})

        gaid = self.c.write(candidate,
                            key_value_pairs=candidate.info['key_value_pairs'],
                            data=candidate.info['data'],
                            **kwargs)
        self.c.update(gaid, gaid=gaid)
        candidate.info['confid'] = gaid

    def add_unrelaxed_step(self, candidate, description):
        """ Add a change to a candidate without it having been relaxed.
            This method is typically used when a
            candidate has been mutated. """

        # confid has already been set by add_unrelaxed_candidate
        gaid = candidate.info['confid']

        t, desc = split_description(description)
        kwargs = {'relaxed': 0,
                  'extinct': 0,
                  t: 1,
                  'description': desc, 'gaid': gaid}

        self.c.write(candidate,
                     key_value_pairs=candidate.info['key_value_pairs'],
                     data=candidate.info['data'],
                     **kwargs)

    def get_number_of_atoms_to_optimize(self):
        """ Get the number of atoms being optimized. """
        v = self.c.get(simulation_cell=True)
        return len(v.data.stoichiometry)

    def get_atom_numbers_to_optimize(self):
        """ Get the list of atom numbers being optimized. """
        v = self.c.get(simulation_cell=True)
        return v.data.stoichiometry

    def get_slab(self):
        """ Get the super cell, including stationary atoms, in which
            the structure is being optimized. """
        return self.c.get_atoms(simulation_cell=True)

    def get_participation_in_pairing(self):
        """ Get information about how many direct
            offsprings each candidate has, and which specific
            pairings have been made. This information is used
            for the extended fitness calculation described in
            L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816
        """
        entries = self.c.select(pairing=1)

        frequency = dict()
        pairs = []
        for e in entries:
            c1, c2 = e.data['parents']
            pairs.append(tuple(sorted([c1, c2])))
            if c1 not in frequency.keys():
                frequency[c1] = 0
            frequency[c1] += 1
            if c2 not in frequency.keys():
                frequency[c2] = 0
            frequency[c2] += 1
        return (frequency, pairs)

    def get_all_relaxed_candidates(self, only_new=False, use_extinct=False):
        """ Returns all candidates that have been relaxed.

        Parameters:

        only_new: boolean (optional)
            Used to specify only to get candidates relaxed since last
            time this function was invoked. Default: False.

        use_extinct: boolean (optional)
            Set to True if the extinct key (and mass extinction) is going
            to be used. Default: False."""

        if use_extinct:
            entries = self.c.select('relaxed=1,extinct=0',
                                    sort='-raw_score')
        else:
            entries = self.c.select('relaxed=1', sort='-raw_score')

        trajs = []
        for v in entries:
            if only_new and v.gaid in self.already_returned:
                continue
            t = self.get_atoms(id=v.id)
            t.info['confid'] = v.gaid
            t.info['relax_id'] = v.id
            trajs.append(t)
            self.already_returned.add(v.gaid)
        return trajs

    def get_all_relaxed_candidates_after_generation(self, gen):
        """ Returns all candidates that have been relaxed up to
            and including the specified generation
        """
        q = 'relaxed=1,extinct=0,generation<={0}'
        entries = self.c.select(q.format(gen))

        trajs = []
        for v in entries:
            t = self.get_atoms(id=v.id)
            t.info['confid'] = v.gaid
            t.info['relax_id'] = v.id
            trajs.append(t)
        trajs.sort(key=lambda x: get_raw_score(x),
                   reverse=True)
        return trajs

    def get_all_candidates_in_queue(self):
        """ Returns all structures that are queued, but have not yet
            been relaxed. """
        all_queued_ids = [t.gaid for t in self.c.select(queued=1)]
        all_relaxed_ids = [t.gaid for t in self.c.select(relaxed=1)]

        in_queue = [qid for qid in all_queued_ids
                    if qid not in all_relaxed_ids]
        return in_queue

    def remove_from_queue(self, confid):
        """ Removes the candidate confid from the queue. """

        queued_ids = self.c.select(queued=1, gaid=confid)
        ids = [q.id for q in queued_ids]
        self.c.delete(ids)

    def get_generation_number(self, size=None):
        """ Returns the current generation number, by looking
            at the number of relaxed individuals and comparing
            this number to the supplied size or population size.

            If all individuals in generation 3 has been relaxed
            it will return 4 if not all in generation 4 has been
            relaxed.
        """
        if size is None:
            size = self.get_param('population_size')
        if size is None:
            # size = len(list(self.c.select(relaxed=0,generation=0)))
            return 0
        lg = size
        g = 0
        all_candidates = list(self.c.select(relaxed=1))
        while lg > 0:
            lg = len([c for c in all_candidates if c.generation == g])
            if lg >= size:
                g += 1
            else:
                return g

    def get_atoms(self, id, add_info=True):
        """Return the atoms object with the specified id"""
        a = self.c.get_atoms(id, add_additional_information=add_info)
        return a

    def get_param(self, parameter):
        """ Get a parameter saved when creating the database. """
        if self.c.get(1).get('data'):
            return self.c.get(1).data.get(parameter, None)
        return None

    def remove_old_queued(self):
        pass
        # gen = self.get_generation_number()
        # self.c.select()

    def is_duplicate(self, **kwargs):
        """Check if the key-value pair is already present in the database"""
        return len(list(self.c.select(**kwargs))) > 0

    def kill_candidate(self, confid):
        """Sets extinct=1 in the key_value_pairs of the candidate
        with gaid=confid. This could be used in the
        mass extinction operator."""
        for dct in self.c.select(gaid=confid):
            self.c.update(dct.id, extinct=1)


class PrepareDB(object):
    """ Class used to initialize a database.

        This class is used once to setup the database and create
        working directories.

        Parameters:

        db_file_name: Database file to use

    """

    def __init__(self, db_file_name, simulation_cell=None, **kwargs):
        if os.path.exists(db_file_name):
            raise IOError('DB file {0} already exists'
                          .format(os.path.abspath(db_file_name)))
        self.db_file_name = db_file_name
        if simulation_cell is None:
            simulation_cell = Atoms()

        self.c = ase.db.connect(self.db_file_name)

        # Just put everything in data,
        # because we don't want to search the db for it.
        data = dict(kwargs)

        self.c.write(simulation_cell, data=data,
                     simulation_cell=True)

    def add_unrelaxed_candidate(self, candidate, **kwargs):
        """ Add an unrelaxed starting candidate. """
        gaid = self.c.write(candidate, origin='StartingCandidateUnrelaxed',
                            relaxed=0, generation=0, extinct=0, **kwargs)
        self.c.update(gaid, gaid=gaid)
        candidate.info['confid'] = gaid

    def add_relaxed_candidate(self, candidate, **kwargs):
        """ Add a relaxed starting candidate. """
        test_raw_score(candidate)

        if 'data' in candidate.info:
            data = candidate.info['data']
        else:
            data = {}

        gaid = self.c.write(candidate, origin='StartingCandidateRelaxed',
                            relaxed=1, generation=0, extinct=0,
                            key_value_pairs=candidate.info['key_value_pairs'],
                            data=data, **kwargs)
        self.c.update(gaid, gaid=gaid)
        candidate.info['confid'] = gaid
