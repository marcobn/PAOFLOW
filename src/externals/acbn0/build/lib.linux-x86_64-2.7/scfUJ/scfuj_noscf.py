import os,sys
import re
import numpy
import subprocess



def get_exeCmd(prefix, engine, calcType,inputFile):
    execPrefix = "mpirun -np 32"
    execPostfix = "-nk 2" 
    QE_path = "/home/laalitha/local/QE.intel.static/bin/"
    WanT_path = "/home/laalitha/local/want.intel.static/bin/"

    if engine=='espresso':
        execDict={'scf':'pw.x','nscf':'pw.x','pdos':'projwfc.x'}
	exeDir = QE_path
	                                                                                                        
    if engine=='want':
        execDict={'want':'bands.x'}
        exeDir = WanT_path
	                                                                                                        
    executable = execDict[calcType]
    outputFile = inputFile.split('.')[0] + '.out'
	
	                                                                                                        
    command  = '%s %s < %s %s >  %s' % ( execPrefix, os.path.join(exeDir, executable),inputFile, execPostfix, outputFile )
	
    return command
	

def chk_species(elm):

    species_Nms = {#d elements
                        'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                        'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                        'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg','Sc',
                        'Ga', 'In','Y',
                       #p elements
                        'C', 'N', 'O', 'Se', 'S', 'Te','Sn','B','F','Al','Si','P','Cl',
                        'Ge','As','Br','Sb','I','Tl','Pb','Bi','Po','At'
                       #s elements
                        'H', 'Sr','Mg', 'Ba','Li','Be','Na','K','Ca','Rb','Cs'}

    if elm in species_Nms: return True
    else: return False

def chkSpinCalc(outfile):

        '''
                Check  output of pw.x calculation is spin polarized or not.
		
        
                Arguments:
                
                --inputFile : Input file for a calculation

        '''

        fin = file(outfile,'r')
	lines = fin.read()
        regex = re.compile(r"(spin.*)\n",re.MULTILINE)
        if len(regex.findall(lines)) != 0: return 2
        else: return 1

def acbn0(prefix):
	
	projOut = "%s_pdos.out" % prefix
        nspin = chkSpinCalc(os.path.join(subdir, "%s_scf.in"%prefix))

        def get_orbital(elm):

                #d elements
                trM_Nms ={ 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                        'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                        'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg','Sc',
                        'Ga', 'In','Y'}

                #p elements
                pElm_Nms = {'C', 'N', 'O', 'Se', 'S', 'Te','Sn','B','F','Al','Si','P','Cl',
                        'Ge','As','Br','Sb','I','Tl','Pb','Bi','Po','At'}

                #s elements
                sElm_Nms = {'H', 'Sr','Mg', 'Ba','Li','Be','Na','K','Ca','Rb','Cs'}

                if elm in trM_Nms: return 2
                elif elm in pElm_Nms: return 1
                elif elm in sElm_Nms: return 0

	def get_CellParams(scfOutput):
                '''
                reads the output from the SCF or relax calculation to get the cell parameters
                produced by the calculation.
        
                Arguments:
                 - scfOutput -- Output from VC-RELAX/RELAX/SCF calculation
                '''
                
                with open(scfOutput,'r') as outFile:
                        lines = outFile.read()
                        re0 = re.compile(r"Begin final coordinates\n.+\n+.+\n(.+)\n(.+)\n(.+)\n",re.MULTILINE)
                        alat = re.findall(r'\s*lattice parameter\s*\(alat\)\s*=\s*([\d\.]*)\s*a.u.',lines)
                        cellParams = re0.findall(lines)
                        initialParamsRegex = re.compile(r"crystal axes: \(cart. coord. in units of alat\)\n(.+)\n(.+)\n(.+)\n",re.MULTILINE)
                        initialParams = initialParamsRegex.findall(lines)
        
                        if len(cellParams):
                                paramMatrixList = []
                                for params in cellParams[0]:
                                        paramArray = params.split()
        
                                        paramMatrixList.append(paramArray)
        
                                paramMatrixList = [paramMatrixList[0],paramMatrixList[1],paramMatrixList[2]]
                                paramMatrix =  numpy.array(paramMatrixList,dtype='|S10')
                                paramMatrix = paramMatrix.astype(numpy.float)
        
                                return float(alat[0]),paramMatrix
        
                        elif len(initialParams):
                                paramMatrixList = []
                                for params in initialParams[0]:
                                        paramArray = params.split()
        
                                        paramMatrixList.append([paramArray[3],paramArray[4],paramArray[5]])
                                paramMatrix =  numpy.array(paramMatrixList,dtype='|S10')
                                paramMatrix = paramMatrix.astype(numpy.float)
        
                                return float(alat[0]),paramMatrix
        
                        else:
                                print 'No card!'
                                return float(alat[0]),[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]


        def gen_input(prefix,nspin):

                #Get cell parameters, arranged as a single string in pattern a1i, a1j, a1k, a2i, a2j, a2k, a3i, a3j, a3k
		scfOutput = os.path.join(subdir,"%s_scf.out"%prefix)
                a,cell= get_CellParams(scfOutput)
                cellParaMatrix=a*cell
                l=cellParaMatrix.tolist()
                cellParaStr = ""
                for i in range(3):
                        cellParaStr += str(l[i]).strip('[]')
                        if i != 2:cellParaStr += ' ,'

                #Get atomic positions in cartesian coordinates, in a single string in pattern x1,y1,z1, x2, y2, z2, ..., xn, yn, zn
                scfOutput = '%s_scf.out' % prefix
                fin = file(os.path.join(subdir,scfOutput),'r')
                lines = fin.read()
                fin.close()
                atmPosRegex = re.compile(r"positions \(alat units\)\n((?:.*\w*\s*tau\(.*\)\s=.*\(.*\)\n)+)",re.MULTILINE)
                lines1 = atmPosRegex.findall(lines)[0]
                atmPosRegex1 = re.compile(r".*=.*\((.*)\)\n+",re.MULTILINE)
                atmPos = atmPosRegex1.findall(lines1)
                atmPosList = []
                for i in atmPos:atmPosList.append(map(float,i.split()))
		        
				#Convert fractional atomic coordinates to cartesian coordinates using the lattice vectors
                atmPosStr = ""
                for i in range(len(atmPosList)):
                        atmPosStr += str(list(numpy.array(atmPosList[i])*a)).strip('[]')
                        if i != len(atmPosList)-1:
                                atmPosStr += ' ,'

                #Get the list of atom labels arranged as a single string
                atmLblRegex = re.compile(r"\d+\s+(\w+).*=.*\n+",re.MULTILINE)
                atmLbls = atmLblRegex.findall(lines1)
                for i in range(len(atmLbls)):
					atmLbls[i]=atmLbls[i].strip('0123456789')
                atmLblsStr= str(atmLbls).strip('[]').replace("'","")

				#Get number of atoms
                natoms = len(atmLblsStr.split())

                #Get the list of atom species
                atmSpRegex = re.compile(r"atomic species   valence    mass     pseudopotential\n(?:.*\(.*\)\n)+",re.MULTILINE)
                atmSpStrList = atmSpRegex.findall(lines)[0].split('\n')[1:-1]
                atmSpList = []
                for i in atmSpStrList:
					atmSpList.append(i.split()[0])


                #Get list of orbitals for each atom species from output of projwfc - projOut
                fin = file(os.path.join(subdir,projOut), 'r')
                proj_lines = fin.read()
		fin.close()

                inFileList = []; #List of acbn0.py input file names
                #For each atomic species
                for atmSp in atmSpList:

                        print "Creating acbn0 inpufile for %s"%atmSp

                        #Get orbital type to apply Hubbard correction
                        ql = get_orbital(atmSp.strip('0123456789'))

                        #Get list of all orbitals of type ql of the same species
                        eqOrbRegex = re.compile(r"state #\s*(\d*): atom.*\(%s.*\).*\(l=%d.*\)\n"%(atmSp.strip('0123456789'),ql),re.MULTILINE)
                        eqOrbList = map(int, map(float,eqOrbRegex.findall(proj_lines)))
                        red_basis = [x - 1 for x in eqOrbList]

                        #Get ones relevant for hubbard center
                        eqOrbRegex = re.compile(r"state #\s*(\d*): atom.*\(%s\s*\).*\(l=%d.*\)\n"%(atmSp,ql),re.MULTILINE)
                        eqOrbList = map(int, map(float,eqOrbRegex.findall(proj_lines)));print eqOrbList
                        red_basis_for2e = [x - 1 for x in eqOrbList]
                        #Get list of orbitals of type l for one atom of the species
                        red_basis_2e = []
                        red_basis_2e.append(red_basis_for2e[0])
                        for i in range(1,len(red_basis_for2e)):
                                if float(red_basis_for2e[i]) == float(red_basis_for2e[i-1])+1:red_basis_2e.append(red_basis_for2e[i])
                                else:break

                        #Create input file for respective species
                        infnm = prefix + "_acbn0_infile_%s.txt"%atmSp
                        fout = file(os.path.join(subdir,infnm), 'w')
                        S = "latvects = " + cellParaStr + "\n"
                        fout.write(S)
                        S = "coords = " + atmPosStr + "\n"
                        fout.write(S)
                        S = "atlabels = " + atmLblsStr + "\n"
                        fout.write(S)
                        fout.write("nspin = %d\n" % nspin)
                        fout.write("fpath = %s\n" % subdir)
                        outfnm = prefix + "_acbn0_outfile_%s.txt"%atmSp
                        fout.write("outfile = %s\n"%outfnm)
                        S = "reduced_basis_dm = " + str(red_basis).strip('[]') + "\n"
                        fout.write(S)
                        S = "reduced_basis_2e = " + str(red_basis_2e).strip('[]') + "\n"
                        fout.write(S)
                        fout.close()

                        #Add filename to acbn0 run list
                        inFileList.append(infnm)

                return inFileList

        def run_acbn0(inputFiles):

                for infnm in inputFiles:

                        cmd="python %s/acbn0.py %s > /dev/null"%(subdir,os.path.join(subdir,infnm))

			try:
                        	print "Starting python acbn0.py %s\n"%(os.path.join(subdir,infnm))
				subprocess.check_output([cmd],shell=True)
	                        print "Finished python acbn0.py %s\n"%(os.path.join(subdir,infnm))
			except subprocess.CalledProcessError as e:
				print "######### ABORTING ACBN0 LOOP ######### \n FAILED %s \n %s\n"%(cmd,e)
				raise SystemExit

        acbn0_inFileList = gen_input(prefix,nspin)
        run_acbn0(acbn0_inFileList)

def get_Ueff(prefix):


        #Get species
        scfInput = os.path.join(subdir, "%s_scf.in" % prefix)
	fin = file(scfInput, 'r')
	inputfile = fin.read()
	fin.close()		
        species = re.findall("(\w+).*UPF",inputfile)
        Uvals = {}

        for isp in species:

                #Check for acbn0 output in the work directory
                acbn0_outFile = os.path.join(subdir, "%s_acbn0_outfile_%s.txt"%(prefix,isp))
                if os.path.isfile(acbn0_outFile):
                        #Get U value from acbn0 output
                        lines = file(acbn0_outFile, 'r').read()
			try:
	                        acbn0_Uval = re.findall("U_eff\s*=\s*(\d+.\d+)",lines)[0]
	                        Uvals[isp] = float(acbn0_Uval)
			except Exception as e:
				print "######### ABORTING ACBN0 LOOP ######### \n Could not find U values from acbn0 output"
				raise SystemExit
                else:
                        Uvals[isp] = 0.001

	print Uvals

	#Record U values in a Log file
        if os.path.isfile(os.path.join(subdir,'%s_uValLog.log' % prefix)):
                with open(os.path.join(subdir,'%s_uValLog.log' % prefix),'a') as uValLog:
                        uValLog.write('%s\n' % Uvals)
        else:
                with open(os.path.join(subdir,'%s_uValLog.log' % prefix),'w') as uValLog:
                        uValLog.write('%s\n' % Uvals)

        return Uvals

def updateUvals(infile, Uvals):
        """
        Modify scf input file to do a lda+u calculation.
        
        Arguments:
         - infile    -- SCF/NSCF input file as a string 
         - Uvals        -- Dictionary of Uvals

        """
	fin = file(infile,'r')
	inputfile = fin.read()
	fin.close()

	print "Updating U values of %s with "%infile, Uvals
		
        #Get species
        species = re.findall("(\w+).*UPF",inputfile)

        #Remove tags
        junkre = re.compile("lda_plus_u.*=.*\n")
        inputfile = junkre.sub('',inputfile)
        junkre = re.compile(" *Hubbard_U.*=.*\n")
        inputfile = junkre.sub('',inputfile)

        #Check for lda_plus_u and Hubbard_U declarations and initialize correctly.      
        A,B = re.split(' *&system', inputfile)

        if len(re.findall('lda_plus_u',inputfile)) == 0:
                insert = ' &system\n    lda_plus_u = .true.,\n'
                for isp in range(len(species)):
                        insert = insert + '    Hubbard_U('+str(isp+1)+') = %f,\n'%(Uvals[species[isp]])
                        new_inputfile = A+insert+B

	fout = file(infile,'w')
	fout.write(new_inputfile)
	fout.close()

        return new_inputfile 

	
def oneRun(prefix,scfOne=False,isInit=False):

		initDir = os.path.join(subdir, "_%s.save"%prefix)
		bakDir = os.path.join(subdir, "_%s.save.bak"%prefix)
		scfInput=file(os.path.join(subdir, "%s_scf.in"%prefix),'r').read()
		regEx = re.compile(r'.*nspin.*=.*(\d).*\n')
		try:
			nspin = int(regEx.findall(scfInput)[0])
		except Exception as e:
			print "Detected Non-spin polarized calculation"
			nspin = 1
			pass

                #Make single scf step input file
                if scfOne == True and isInit ==False:
			os.system("rm -rf %s"%initDir)	
			os.system("cp -r %s %s"%(bakDir, initDir))

			if len(re.findall("restart_mode\s*=\s*'restart'",scfInput)) == 0:
				replaceCalcRE = re.compile(r"restart_mode\s*=\s*'\s*(.+?)\s*'")
        	        	replaceCalc = replaceCalcRE.findall(scfInput)[0]
				scfInput = replaceCalcRE.sub("restart_mode='restart'",scfInput)		                
                        if len(re.findall('electron_maxstep = 1',scfInput)) == 0:
                                A,B = re.split(' *&electrons', scfInput)
                                insert = ' &electrons\n   electron_maxstep = 1,\n'
                                scfInput = A + insert + B
                        fout = file(os.path.join(subdir, "%s_scf.in"%prefix),'w')
                        fout.write(scfInput)
                        fout.close()

		#Dictionary of files for calculations
		fileList = {'scf':"%s_scf.in"%prefix, 
			    'nscf':"%s_nscf.in"%prefix,
			    'pdos':"%s_pdos.in"%prefix,
			    'want_bands':"%s_want_bands.in"%prefix,
			    'want_bands_up':"%s_want_bands_up.in"%prefix, 
			    'want_bands_down':"%s_want_bands_down.in"%prefix, }
		
		if nspin != 2 :
			calcList = ['scf','nscf','pdos','want_bands']
			
		elif nspin == 2:
			calcList = ['scf','nscf','pdos','want_bands_up','want_bands_down']

		print "List of calculations for each ACBN0 iteration ", str(calcList).strip('[]')

		engine = {'scf':'espresso',
			  'nscf':'espresso',
			  'pdos':'espresso',
			  'want':'want',}

		#Run scf,nscf,pdos and want_bands
		for calc in calcList:
			command = get_exeCmd(prefix, engine[calc.split("_")[0]],calc.split("_")[0],fileList[calc])
			try:
				if calc == 'nscf':
					os.system("cp -r %s %s"%(initDir, bakDir))
						
				print "Starting %s in %s"%(command, subdir)
				subprocess.check_output([command],shell=True)
				print "Finished %s in %s"%(command, subdir)

			except subprocess.CalledProcessError as e:
				print "######### ERROR IN ACBN0 LOOP CALCULATION ######### \n FAILED %s in %s\n %s\n"%(command, subdir,e)
				pass
			#	raise SystemExit
			if "want" in calc:
				try:
					wantOutput = file("%s_%s.out"%(prefix,calc),'r').read()
					errorList = re.findall(r'.*error #\s*\d+.*\n.*\n',wantOutput)
					if len(errorList) > 0:
						print "######### ABORTING ACBN0 LOOP ######### \n FAILED %s in %s\n WanT ERROR: \n%s"%(command, subdir,errorList[0])
						raise SystemExit
				except Exception as e:
					print e 
					pass

		#Run acbn0 
		acbn0(prefix)

		#Get new U values
		uVals = get_Ueff(prefix)

		return uVals
			


def main():

	#Set path to current working dir
	global subdir
	subdir = os.getcwd()
	uThresh = 0.001

	prefix = sys.argv[1]
	if len(sys.argv)>1:

		#Set initial U values in scf and nscf inputfiles
		uVals = {}
		fin = file(os.path.join(subdir,"%s_scf.in"%prefix),'r')
		inputfile=fin.read()
		fin.close()
		species = re.findall("(\w+).*UPF",inputfile)
		for sp in species:
			uVals[sp] = uThresh
		updateUvals(os.path.join(subdir,"%s_scf.in"%prefix),uVals)
		updateUvals(os.path.join(subdir,"%s_nscf.in"%prefix),uVals)

		newUvals = oneRun(prefix, isInit=True)
		convergence = False
		while convergence == False:

			#Update U values		
			uVals = newUvals
			updateUvals(os.path.join(subdir,"%s_scf.in"%prefix),uVals)
			updateUvals(os.path.join(subdir,"%s_nscf.in"%prefix),uVals)
			#Get new U values
			newUvals = oneRun(prefix,scfOne=True)
		
			#Check for convergence
			for key in uVals.keys():
				if abs(uVals[key]-newUvals[key]) > uThresh:
					convergence = False
					break;
				else:
					convergence = True

	else:
		print "Usage: scfuj_noMTF.py prefix"

		

if __name__ == "__main__":
	main()
