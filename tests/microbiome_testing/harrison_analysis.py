import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import arviz as az
import pickle as pkl
from matplotlib.pyplot import cm
import matplotlib

from scdcdm.util import data_generation as gen
from scdcdm.util import comp_ana as mod
from scdcdm.util import result_classes as res
from scdcdm.util import multi_parameter_sampling as mult
from scdcdm.util import cell_composition_data as dat
from scdcdm.util import data_visualization as viz

pd.options.display.float_format = '{:10,.3f}'.format
pd.set_option('display.max_columns', 10)

#%%

data_path = "C:/Users/Johannes/Documents/PhD/data/rosen_mincount10_maxee2_trim200_results_forpaper/"

otu_file = "rosen_mincount10_maxee2_trim200.otu_table.99.denovo.rdp_assigned.paper_samples.txt"

with open(data_path + otu_file, "rb") as file:
    otus = pd.read_csv(file, sep="\t", index_col=0)

print(otus)

#%%
meta_file = "patient_clinical_metadata.csv"

with open(data_path + meta_file, "rb") as file:
    metadata = pd.read_csv(file, sep=",", index_col=0)

metadata = metadata[~metadata.index.str.endswith('F')]
metadata = metadata[~metadata.index.str.endswith('sick')]
metadata = metadata[~metadata.index.str.endswith('F2')]
metadata = metadata[~metadata.index.str.endswith('F2T')]
metadata = metadata[~metadata.index.str.endswith('2')]
metadata = metadata[~metadata.index.str.startswith('05')]

print(metadata)

#%%

meta_rel = metadata[(~pd.isna(metadata["mbs_consolidated"])) & (~pd.isna(metadata["bal"]))]

print(meta_rel)

#%%

print(meta_rel.groupby("mbs_consolidated").count())

#%%

otus_rel = otus.loc[otus.index.isin(meta_rel.index)]

print(otus_rel)


