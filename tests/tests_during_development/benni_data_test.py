import numpy as np
import arviz as az
import seaborn as sns
import pandas as pd
import pickle as pkl
import importlib
import anndata as ad
import matplotlib.pyplot as plt

from sccoda.util import result_classes as res
from sccoda.util import comp_ana as mod
from sccoda.util import data_generation as gen
from sccoda.util import cell_composition_data as dat

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
sns.set_style("ticks")

#%%

# Import data

celltypes_path = "C:/Users/Johannes/Documents/Uni/Master's_Thesis/data/cell_fractions.csv"

celltypes = pd.read_csv(celltypes_path, sep=",", header=[0], index_col=[0])

#%%


cell_counts = celltypes.loc[:, ["sample_id", "cell_type", "counts"]].\
    set_index(["sample_id", "cell_type"]).unstack(fill_value=0).fillna(0).reset_index()
print(cell_counts)

#%%

cell_counts_2 = cell_counts
cell_counts_2.columns = cell_counts_2.columns.droplevel(0)
cell_counts_2 = cell_counts_2.rename(columns={"": "sample_id"})
cell_counts_2["Condition"] = cell_counts_2["sample_id"].str.replace(r"[0-9]", "")
print(cell_counts_2)

#%%
importlib.reload(dat)
data = dat.from_pandas(cell_counts_2, ["sample_id", "Condition"])
print(data.X)
print(data.obs)
print(data.var)

#%%

cells = cell_counts.iloc[:, 1:].to_numpy().astype("int")
print(cells)

obs = pd.DataFrame(cell_counts["sample_id"])
obs["Condition"] = obs["sample_id"].str.replace(r"[0-9]", "")
print(obs)

var = pd.DataFrame(index=cell_counts.iloc[:, 1:].columns.droplevel(0))
print(var)

data = ad.AnnData(X=cells.astype("int32"), obs=obs, var=var)

#%%
importlib.reload(mod)

model = mod.CompositionalAnalysis(data=data, formula="Condition", baseline_index=3)
result = model.sample_hmc()

#%%

result.summary()


#%%
