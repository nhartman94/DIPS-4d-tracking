''' subPreprocess.py

Goal: For a number of preprocess files, loop over the files 
      and submit the batch jobs and submit each of these to 
      a separate file 

Nicole Hartman
Summer 2023
'''

def writeScript(cmd,fname):
    '''
    Write cmd to a .sh script with name `fname` 
    '''

    f = open(fname,"w")

    # write the output
    f.write("#!/bin/bash -l\n#\n")
    f.write(cmd+'\n')

    # Close the file
    f.close()

    # Make it executable
    ex = "chmod a+x {}".format(fname)
    os.system(ex)


def writeSubFile(subFile, exeFile, logFile):
    '''
    Write a submission script (subFile) that runs an executable (exeFile)
    and prints the output to logFile.
    '''

    f = open(subFile,"w")

    # write the output
    f.write("#!/bin/bash -l\n#\n")
    f.write(f"executable = {exeFile}\n")
    f.write(f"output = {logFile}\n")
    f.write(f"error = {logFile}\n")
    f.write(f"log = {logFile}\n")
    f.write(f"+JobFlavour = \"espresso\"\n")
    f.write('queue \n')

    # Close the file
    f.close()

    # Make it executable
    ex = "chmod a+x {}".format(subFile)
    os.system(ex)



from glob import glob
import os

# glob the files
fileList = glob('TODO:_replace_whatever_you_want')

for i, fin in glob(fileList):

    # Optionally, could do some string manipulation so 
    # that the number of the output file matches the number
    # of the input file


    for mode in ["train","test"]:
        cmd = f"python root_to_ml.py --filename {fin} --output df_{mode}_{i}.h5 --mode {mode} --onlyCuts"

        exeFile = f'exe_{mode}_{i}.sh'
        subFile = f'sub_{mode}_{i}.sh'
        logFile = f'log_{mode}_{i}.txt'

        writeSubFile(cmd, exeFile)
        os.system(subFile, exeFile, logFile)



