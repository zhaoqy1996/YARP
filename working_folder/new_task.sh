#!/usr/bin/env bash
source setup.sh

python ../yarp/ERS_enumeration/reaction_enumeration.py ../paper-example/Reactant/ketohydroperoxide.xyz -N 1 -t [1,2,3,4] -o KHP

mkdir KHP-network
python ../yarp/Construct_pathway/locate_TS.py
echo "The result stored in 'KHP-network/pp_0/IRC-result/record.txt ' "
