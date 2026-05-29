
Instructions for running the Quantum ESPRESSO examples:

For backward compatability with PAOFLOW v1.0 inputfiles, use the 'main.py' in this directory to run PAOFLOW with an xml input file ('inputfile.xml'). Place 'main.py' and the inputfile in the same directory and execute PAOFLOW in one of the standard ways:
python main.py
python main.py <work_directory>
python main.py <work_directory> <inputfile_name>

mpirun -np <num_cores> python main.py

The options have default values:
<work_directory> - './'
<inputfile_name> - 'inputfile.xml'
