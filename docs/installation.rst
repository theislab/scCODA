Installation
============

A functioning python environment (>=3.7) is necessary to run this package.

This package uses the tensorflow (>=2.1.0) and tensorflow-probability (>=0.9.0) packages.
The GPU versions of these packages have not been tested with scCODA and are thus not recommended.

To install scCODA from source:

- Navigate to the directory you want scCODA in
- Clone the repository from Github (https://github.com/theislab/scCODA)::

    git clone https://github.com/theislab/scCODA

- Navigate to the root directory of scCODA::

    cd scCODA

- Install dependencies::

    pip install -r requirements.txt

Import scCODA in a Python session via::

    import sccoda

