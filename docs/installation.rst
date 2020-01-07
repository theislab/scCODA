Installation
============

A functioning python environment (>=3.7) is necessary to run this package.

This package uses the tensorflow (>=2.0) and tensorflow-probability (>=0.8.0) packages. 
The GPU versions of these packages have not been tested with compositionalDiff and are thus not recommended.
To install these packages via pip, call::

    pip install tensorflow
    pip install tensorflow-probability
    
To install compositionalDiff from source:

- Navigate to the directory you want compositionalDiff in
- Clone the repository from (Github)[https://github.com/theislab/compositionalDiff]
- Navigate to the root directory of compositionalDiff
- Install compositionalDiff::

    pip install -e
    
Import compositionalDiff in Python via::

    import compositionalDiff as cd

    



