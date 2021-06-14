#!/usr/bin/env bash
source setup.sh

python ../yarp/Construct_pathway/locate_TS.py -c analysis_config.txt 
echo "The result stored in '../paper-example/scratch/KHP-network/IRC-result/record.txt ' "
