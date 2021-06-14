# standard library imports
import sys
import os
from os import path
import subprocess
from subprocess import Popen

# third party 
import numpy as np

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from base_lot import Lot
from utilities import *

class xTB(Lot):

    def run(self,geom,multiplicity):
        # set default parameters
        config = {}
        config["charge"]   = 0
        config["spin"]     = 0
        config["accuracy"] = 1.0
        config["max_iter"] = 100
        config["calc_type"]= 'grad'
        config["namespace"]= "xTB"

        # import given parameters
        if self.lot_inp_file != False:
            with open(self.lot_inp_file) as lot_inp:
                lot_inp_lines = lot_inp.readlines()
            for line in lot_inp_lines:
                if line[0] == '#': 
                    continue
                
                field = line.split()
                if field[0] in config.keys():
                    config[field[0]] = field[1]
                else:
                    print("Warning! '{}' is ignored...".format(line))
                    
        inpstring = '{} \n\n'.format(len(geom))
        for coord in geom:
            for i in coord:
                inpstring += str(i)+' '
            inpstring += '\n'
        
        tempfilename = 'temp_{}'.format(config["namespace"])
        tempfile = open(tempfilename+'.xyz','w')
        tempfile.write(inpstring)
        tempfile.close()
        
        path2xTB = os.popen('which xtb').read().rstrip()
        #path2xTB = '/home/zhao922/bin/xTB/xtb_6.2.3/bin/xtb'
        user = os.environ['USER']
        cwd = os.environ['PWD']
        try:
            slurmID = os.environ['SLURM_ARRAY_JOB_ID']
            try:
                slurmTASK = os.environ['SLURM_ARRAY_TASK_ID']
                runscr = '/home/'+user+'/'+slurmID+'/'+slurmTASK
            except:
                runscr = '/home/'+user+'/'+slurmID
        except:
            xTBscr = 'tempxTBrun'
            runscr = '/home/'+user+'/'+xTBscr

        # command line
        substring="{} -c {} -u {} -a {} --iterations {} --{} --namespace '{}' --norestart {}.xyz"
        sub_cmd  =substring.format(path2xTB,config["charge"],config["spin"],config["accuracy"],config["max_iter"],\
                                   config["calc_type"],config["namespace"],tempfilename)
        
        os.system('mkdir -p {}'.format(runscr))
        os.system('mv {}.xyz {}/'.format(tempfilename,runscr))
        cmd = 'cd {}; {} > {}/{}.log; cd {}'.format(runscr,sub_cmd,runscr,tempfilename,cwd)
        Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0]
        #os.system(cmd)

        engradpath = runscr+'/{}.gradient'.format(config["namespace"]) 
        with open(engradpath) as engradfile:
            engradlines = engradfile.readlines()

        # total energies
        for ln,line in enumerate(engradlines):
            fields = line.split()
            if len(fields)==0: continue
            if fields[0]=="cycle":
                N_line = ln
                energy = float(fields[6])
        
        natoms = len(geom)
        forces = np.zeros((natoms,3))
        fl = 0
        for ln in range(N_line+natoms*2+2)[-1-natoms:-1]:
            F = []
            fields=engradlines[ln].split()
            for li in range(3):
                F += [float(fields[li].split('E')[0])*10**float(fields[li].split('E')[1])]
            forces[fl] = F
            fl += 1
        
        self.E.append((multiplicity,energy))
        self.grada.append((multiplicity,forces))
        return


    def get_energy(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.runall(geom)
        tmp = self.search_tuple(self.E,multiplicity)
        return np.asarray(tmp[state][1])*units.KCAL_MOL_PER_AU


    def get_gradient(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.runall(geom)
        tmp = self.search_tuple(self.grada,multiplicity)
        return np.asarray(tmp[state][1]) #*ANGSTROM_TO_AU #xTB grad is given in AU
        
