'''
steerConcat.py

Just the two lines that concatenates the inputs from submitting
the jobs in parallel to the lxplus batch system. 

Nicole Hartman
Summer 2023

'''

import os

'''
TO DO: Add the input files that is the output of the steering script
subPreprocess.py (if you're changing this)
'''

# Important: Need the globbed file input to be passed in quotation marks!
cmd0 = 'python root_to_ml.py --filename "df_train_*.h5" --output output_train.h5 --mode train'
cmd1 = 'python root_to_ml.py --filename "df_test_*.h5" --output output_test.h5 --mode test'
os.system(cmd0)
os.system(cmd1)


