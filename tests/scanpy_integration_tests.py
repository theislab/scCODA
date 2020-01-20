import scanpy as sc
import numpy as np
import pandas as pd

#%%
data = sc.datasets.ebi_expression_atlas("E-MTAB-4888i8")
print(data)