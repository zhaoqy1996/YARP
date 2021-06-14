#!/usr/bin/env bash
export PATH="/scratch/brown/bsavoie/YARP/yarp/external-software/openbabel/bin/:$PATH"         # ABSOLUTE LOCATION OF OPENBABEL/BIN 
export PATH="/scratch/brown/bsavoie/YARP/yarp/external-software/xtb/bin/:$PATH"               # ABSOLUTE LOCATION OF XTB/BIN
here=$( pwd | awk 'BEGIN{FS=OFS="/"}{NF--; print}' )                                          # AUTOMATICALLY DETERMINE LOCATION OF DEMO FOLDER
sed -i 's|REPLACE_ME|'$here'|g' analysis_config.txt                                           # UPDATE PATHS IN analysis_config.txt FOR USER
echo "A reminder: Please make sure absolute pathway is assigned"                              # REMINDER FOR USER
