# Basic YARP usage
This document provides instructions for running a YARP prediction task.
 
Follow the steps as listed below to perform YARP on your interested systems.

## Set up package/software dependencies for YARP

   There are four softwares/packages YARP calls to prepare quantum chemistry tasks. 
   
1. Openbabel - used to convert xyz files to inchikey & smiles strings. 
2. pyGSM     - used to perform growing string calculations. 
3. xTB       - semi-empirical quantum chemistry for preoptimizing reaction pathways. 
4. ASE       - used to align reactant and product. 
       
All of these packages are stored in the 'yarp/external-software' folder.    
    
* The provided version of YARP is written in python 2.7 to be compatible with pyGSM. The rest of the instructions assume that an anaconda python 2.7 distribution is installed on the user's machine.

* Since the newest version of ASE (Atomic Simulation Environment) requires python version >3.6, an older compatible ASE version is provided in the 'yarp/external-software' folder. Simply copy this folder to the site-packages folder of your anaconda 2.7 distribution.

* Add openbabel and xTB to your PATH if they are not already installed in your current environment. Replace the pathways in 'working_folder/setup.sh' by the absolute paths for these packages and run:

           source working_folder/setup.sh

    Note: we strongly recommend you install or load openbabel in your cluster. Some unexpected errors might appear when importing openbabel from the external-software folder without installation on the local system. 
        
* If you encounter the error "ModuleNotFoundError: No module named [missing package]" when running YARP, you are missing some common Python packages that are available through conda. To add these dependencies run:
       
           conda install [missing package]
  
## Using YARP 
 
We have provided three bash scripts to reproduce the results from the paper "More and Faster: Simultaneously Improving Reaction Coverage and Computational Cost in Automated Reaction Prediction Tasks" in 'working_folder/':

1. read_output.sh  - this script will go through all steps in YARP but will read in the Gaussian output files rather run the quantum chemistry calculations from scratch. 	  
2. partial_test.sh - this script will directly use the jointly-optimized reactant and product pair *.xyz files and go into the 'pygsm'->'TS opt'->'IRC' workflow. 
3. new_task.sh     - this script will begin a new YARP tak. It will first perform elementary reaction enumeration and then perform joint-optimization to generate reactant product pairs. After this the script will perform the 'pygsm'->'TS  opt'->'IRC' workflow.

For the purpose of the demo, we only recommend running (i). (ii) and (iii) will run quantum chemistry jobs and expect to be run on a cluster. If experimenting with (ii) and (iii), the scripts require the user to modify the YARP/working_folder/config.txt and YARP/working_folder/analysis_config.txt files with your cluster-specific details (e.g., walltime, number of processors, queue, etc.) and gaussian is expected to be available via a module load call. Advanced users are welcome to experiment with (iii) on their own systems and reach out to the authors for guidance on configuring their run.  

## Expectations for the Demo run

* Running read_output.sh will perform the YARP steps for calculating and analyzing the first step of decomposition products for KHP from the main text. We have included quantum chemistry output files for all of the products (even failure cases, as reported in the main text) which are utilized by the script instead of directly performing the quantum chemistry calculations. For this reason, read_output.txt can be run on a desktop computer within a few minutes. 

* During execution of the read_output.sh script, a small set of summary details are printed to STDOUT. The user should see that some of the transition state searches and IRC calculations fail, which is expected behavior as reported in the results. 

* Input geometries, GSM localized reaction pathways, and transition states, are stored in the YARP/paper_example for additional investigation. 

## Contents in paper-example

* Reactant: xyz files of ketohydroperoxide and molecules in Zimmerman's benchmarking dataset. 

* potential product: outputs of reaction_enumeration.py performed on ketohydroperoxide. reaction channel folder contains corresponding products from reactant (pp0) and other first step intermediates. 

* scratch: pyGSM and Gaussian outputs are stored here. We first provide depth=1 reaction pathways of ketohydroperoxide decomposition for YARP testing. Users are welcome to experiment with other reactions (i.e depth=2 KHP decomposition and depth=1 reactions on Zimmerman's dataset) by reaching out to the authors. 
