import os
import sys
import threading

import numpy as np

from ase.calculators.socketio import SocketClient, SocketIOCalculator
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.cluster.icosahedron import Icosahedron

# If multiple test suites are running, we don't want port clashes.
# Thus we generate a port from the pid.
# maxpid is commonly 32768, and max port number is 65536.
# But in case maxpid is much larger for some reason:
pid = os.getpid()
port = (3141 + pid) % 65536
# We could also use a Unix port perhaps, but not yet implemented

#unixsocket = 'grumble'
timeout = 20.0

def getatoms():
    return Icosahedron('Au', 3)


def run_server(launchclient=True):
    atoms = getatoms()

    with SocketIOCalculator(log=sys.stdout, port=port,
                            timeout=timeout) as calc:
        if launchclient:
            thread = launch_client_thread()
        atoms.calc = calc
        opt = BFGS(atoms)
        opt.run()

    if launchclient:
        thread.join()

    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()

    atoms.calc = EMT()
    ref_forces = atoms.get_forces()
    ref_energy = atoms.get_potential_energy()

    refatoms = run_normal()
    ref_energy = refatoms.get_potential_energy()
    eerr = abs(energy - ref_energy)
    ferr = np.abs(forces - ref_forces).max()

    perr = np.abs(refatoms.positions - atoms.positions).max()
    print('errs e={} f={} pos={}'.format(eerr, ferr, perr))
    assert eerr < 1e-12, eerr
    assert ferr < 1e-12, ferr
    assert perr < 1e-12, perr

def run_normal():
    atoms = getatoms()
    atoms.calc = EMT()
    opt = BFGS(atoms)
    opt.run()
    return atoms

def run_client():
    atoms = getatoms()
    atoms.calc = EMT()

    import socket
    BrokenPipe = socket.error if sys.version_info[0] == 2 else BrokenPipeError

    try:
        with open('client.log', 'w') as fd:
            client = SocketClient(log=fd, port=port,
                                  timeout=timeout)
            client.run(atoms, use_stress=False)
    except BrokenPipe:
        # I think we can find a way to close sockets so as not to get an
        # error, but presently things are not like that.
        pass


def launch_client_thread():
    thread = threading.Thread(target=run_client)
    thread.start()
    return thread


#try:
run_server()
#finally:
#    if os.path.exists(unixsocket):
#        os.unlink(unixsocket)
