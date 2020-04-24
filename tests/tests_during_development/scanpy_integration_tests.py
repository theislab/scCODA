import scanpy as sc
import numpy as np
import pandas as pd
from scdcdm.util import cell_composition_data as ccd

#%%
adata_ref = sc.datasets.pbmc3k_processed()  # this is an earlier version of the dataset from the pbmc3k tutorial
print(adata_ref.X.shape)

#%%
cell_counts = adata_ref.obs["louvain"].value_counts()

print(cell_counts)

#%%
df = pd.DataFrame()
df = df.append(cell_counts, ignore_index=True)
print(df)


#%%
cell_counts_2 = cell_counts
new_dat = np.random.choice(1500, cell_counts_2.size)
cell_counts_2 = cell_counts_2.replace(cell_counts_2.data, new_dat)
cell_counts_2.index = cell_counts_2.index.tolist()
cell_counts_2["test_type"] = 256
print(cell_counts_2)

#%%

df = df.append(cell_counts_2, ignore_index=True)
print(df)

#%%
cell_counts_3 = cell_counts_2.iloc[[0, 3, 7, 8]]
print(cell_counts_3)

#%%
df = df.append(cell_counts_3, ignore_index=True)
print(df)


#%%

covs = dict(zip(np.arange(3), np.random.uniform(0, 1, 3)))
print(covs)

print(covs[0])

#%%
ddf = pd.DataFrame()
ddf = ddf.append(pd.Series(covs), ignore_index=True)
print(ddf)

#%%
print(adata_ref.uns_keys())
print(adata_ref.uns["neighbors"])

#%%
adata_ref.uns["cov"] = {"x1": 0, "x2": 1}
print(adata_ref.uns["cov"])

#%%
print(df.sum(axis=0).rename("n_cells").to_frame())


#%%

data = ccd.from_scanpy_list([adata_ref, adata_ref, adata_ref],
                            cell_type_identifier="louvain",
                            covariate_key="cov")

print(data.X)
print(data.var)
print(data.obs)
