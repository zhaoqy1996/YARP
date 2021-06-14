from ase.test import cli, require

require('aims')
# warning! parameters are not converged - only an illustration!
cli("""ase build -x bcc -a 3.6 Li | \
ase run aims -s 0.3 -p \
kpts=1.5,xc=LDA,sc_accuracy_rho=5.e-2,relativistic=none,compute_analytical_stress=True,sc_accuracy_forces=5.e-1""")
