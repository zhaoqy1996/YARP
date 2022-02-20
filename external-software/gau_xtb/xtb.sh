#!/bin/bash

read atoms derivs charge spin < $2

#Create temporary .xyz file
#the element index should be replaced with element name, and the coordinate should be convert to Angstrom
echo "Generating mol.tmp"
cat >> mol.tmp <<EOF
$atoms

$(sed -n 2,$(($atoms+1))p < $2 | cut -c 1-72)
EOF

echo "Generating mol.xyz via genxyz"
./../genxyz
rm -f mol.tmp

rm -f charges energy xtbrestart gradient hessian xtbout
rm -f hessian xtb_normalmodes g98_canmode.out g98.out wbo xtbhess.coord

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
uhf=`echo "$spin-1" | bc` #nalpha-nbeta
if [ $derivs == "2" ] ; then
	echo "Running: xtb mol.xyz --chrg $charge --uhf $uhf --hess --grad > xtbout"
	xtb mol.xyz --chrg $charge --uhf $uhf --hess --grad > xtbout
elif [ $derivs == "1" ] ; then
	echo "Running: xtb mol.xyz --chrg $charge --uhf $uhf --grad > xtbout"
	xtb mol.xyz --chrg $charge --uhf $uhf --grad > xtbout
fi
echo "xtb running finished!"

echo "Extracting data from xtb outputs via extderi"
./../extderi $3 $atoms $derivs

rm -f charges energy xtbrestart gradient hessian xtbout mol.xyz tmpxx vibspectrum
rm -f hessian xtb_normalmodes g98_canmode.out g98.out wbo xtbhess.coord .tmpxtbmodef
rm -f .engrad xtbtopo.mol xtbhess.xyz

