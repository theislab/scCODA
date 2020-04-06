Installation
============

A functioning python environment (>=3.7) is necessary to run this package.

This package uses the tensorflow (>=2.1.0) and tensorflow-probability (>=0.9.0) packages.
The GPU versions of these packages have not been tested with SCDCdm and are thus not recommended.
To install these packages via pip, call::

    pip install tensorflow
    pip install tensorflow-probability
    
To install SCDCdm from source:

- Navigate to the directory you want SCDCdm in
- Clone the repository from (Github)[https://github.com/johannesostner/SCDCdm_public]
- Navigate to the root directory of SCDCdm
- Install SCDCdm::

    pip install -e
    
Import SCDCdm in a Python session via::

    import SCDCdm as scdcdm

