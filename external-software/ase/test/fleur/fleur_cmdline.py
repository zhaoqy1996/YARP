from ase.test import cli, require
require('fleur')
cli('ase build -x fcc -a 4.04 Al | ase run fleur -p kpts=3.0,xc=PBE')
