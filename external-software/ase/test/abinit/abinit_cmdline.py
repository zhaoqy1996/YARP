from ase.test import cli

cli("""
ase build -x fcc -a 4.04 Al |
ase run abinit -p xc=PBE,kpts=3.0,ecut=340,toldfe=1e-5,chksymbreak=0""",
    'abinit')
