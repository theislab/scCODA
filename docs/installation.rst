Installation
============

A functioning python environment (>=3.7) is necessary to run this package.

This package uses the tensorflow (>=2.1.0) and tensorflow-probability (>=0.9.0) packages.
The GPU versions of these packages have not been tested with scCODA and are thus not recommended.
To install these packages via pip, call::

    pip install tensorflow
    pip install tensorflow-probability
    
To install scCODA from source:

- Navigate to the directory you want scCODA in
- Clone the repository from (Github)[https://github.com/johannesostner/scCODA_public]
- Navigate to the root directory of scCODA
- Install scCODA::

    pip install -e
    
Import scCODA in a Python session via::

    import scCODA as sccoda

