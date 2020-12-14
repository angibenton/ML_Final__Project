# ML Final Project - summary of repository contents 
## Code 
Preprocessing scripts: 
* **preprocess.py** - cleaning, class balancing, 3-class to 2-class. raw_data.csv -> p1_train.csv, etc.
* **correctSpelling.py** - correcting abnormal word spellings, unused in final workflow because it lowered accuracy 
* **parse_conll.py** -  uses the tweebo library to construct dependency graphs. p1_train.csv -> p1_train_parsed.csv
  Note: this file can't be run unless docker is running locally. 

SVM:
* **dualsvm_unused.py** - our implementation of a dual SVM, unused in the final workflow 
* **models.py** -  KernelPegasos model from HW4, modified slightly 
* **kernel.py** -  the novelty of this project, 4 kernels that use dependency graphs

**driver_functions.py** - functions for training and testing, reveals how the SVM should be used on the data.

**generate_hate_speech.py** - hate speech generator 

__misc_scripts/*__ - various python files that were for testing, etc., and are not clean/current. 

__baseline/*__ - code from hw4 used for computing baseline accuracy



## Other 
* __required.txt__  - all required python packages 
* __data/*__ - csv files containing training and testing examples
* __kernel_matrices/*__ - csv files containing precomputed kernel matrices
* __saved_models/*__ - models saved with pickle 




