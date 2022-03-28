# Reaction conformational sampling
This document provides instructions for using the reaction conformational sampling algorithm.
 
Follow the steps as listed below to perform the conformational sampling on your interested systems.

## Set up package/software dependencies

   There are two softwares/packages YARP calls to prepare quantum chemistry tasks. 
   
1. Openbabel - used to convert xyz files to inchikey & smiles strings. 
2. xTB       - semi-empirical quantum chemistry for preoptimizing reaction pathways. 
3. CREST     - conformational sampling package developed by Grimme group

* Make sure you have added openbabel and xTB to your environment PATH if they are not already installed in your current environment. Also make sure the crest is put in the same folder as xtb. Check it by entering

           xtb -h
           obabel -H
           crest -H
the terminal should print out the helping message.
        
* If you encounter the error "ModuleNotFoundError: No module named [missing package]" when running YARP, you are missing some common Python packages that are available through conda. To add these dependencies run:
       
           conda install [missing package]
or 
       
           pip install [missing package]
  
## Using the package
 
To run the reaction conformational sampling, you should first prepare an input folder containing xyz files of input reactions (see the example xyz files in paper-example/inputs). After that, you can simply entering 
        
           python conf_gen.py INPUT_FOLDER -w OUTPUT_FOLDER

to perform the reaction conformational sampling. However, this command will use default settings to generate reaction conformations. To changing the default setting, like the properties of the system (charge, multiplicity), the number of maximum output conformations, the sampling schemes, you can obtain the help information by 
        
           python conf_gen.py -h

Follow the description of each parameter you will be able to set up the calculation according to your desire. 

* --remove_constraints: this flag is kind of important. We suggest you include this flag if you are generating conformations for reactions other than b2f2 reactions.
* --product_opt: be careful with this flag, it might change the product geometry if xTB has trouble dealing with such system.
* --rank_by_energy: if you are dealing with an extremely complex system, you can add this flag to speed up the conformational sampling. However, the quality of finally generated reaction conformations might be affected.
## Contents in paper-example

* inputs: xyz files of reactions studied in the RCS paper, including peptide (SAAA), Ireland-Claisen (IC), competing intramolecular Diels-Alder reaction (ketothioester) and several failure cases in the YARP study.

* IC: the output folder of the Ireland-Claisen case study. "conformer" folder stores the CREST output files. "xTB-folder" is a scratch folder for running xTB level calculations. The "input_files_conf" is the target output folder which contains the RF selected reaction conformations. 
