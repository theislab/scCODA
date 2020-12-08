# scCODA - Single-cell differential composition analysis 
scCODA allows for identification of compositional changes in high-throughput sequencing count data, especially cell compositions from scRNA-seq.
It also provides a framework for integration of results directly from *scanpy* and other sources.

The statistical methodology and benchmarking performance are described in:
 
*scCODA: A Bayesian model for compositional single-cell data analysis (Büttner and Ostner et al., 2020)*
(Code for the article available at https://github.com/theislab/scCODA_reproducibility)

For further information, please refer to the 
[documentation](https://scdcdm-public.readthedocs.io/en/latest/) and the 
[tutorials](https://github.com/theislab/SCDCdm/blob/master/tutorials/Tutorial.ipynb).

## Installation

A functioning python environment (>=3.7) is necessary to run this package.

This package uses the tensorflow (>=2.1.0) and tensorflow-probability (>=0.9.0) packages.
The GPU versions of these packages have not been tested with scCODA and are thus not recommended.
To install these packages and all other dependencies via pip, call::

    pip install tensorflow
    pip install tensorflow-probability
    
To install scCODA from source:

- Navigate to the directory you want scCODA in
- Clone the repository from [Github](https://github.com/johannesostner/scCODA_public)
- Navigate to the root directory of scCODA
- Install scCODA::

    pip install -e
    
Import scCODA in a Python session via::

    import scCODA as sccoda




