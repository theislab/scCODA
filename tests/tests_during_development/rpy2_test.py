import os
os.environ["R_HOME"] = "C:\\Program Files\\R\\R-4.0.3"
os.environ["PATH"] = r"C:\\Program Files\\R\\R-4.0.3\\bin\\x64" + ";" + os.environ["PATH"]

import numpy as np
import pandas as pd
import importlib
import pickle as pkl

from scdcdm.util import data_generation as gen
from scdcdm.model import other_models as om
from paper_simulation_scripts import benchmark_utils as add

import rpy2.robjects as rp
from rpy2.robjects import numpy2ri, pandas2ri
numpy2ri.activate()
pandas2ri.activate()
r = rp.r

import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

#%%
aldex2 = rpackages.importr("ALDEx2")
tv = rpackages.importr("tidyverse")
dirreg = rpackages.importr("DirichletReg")
r_base = rpackages.importr("base")
scdney = rpackages.importr("scdney")

#%%

# r install procedure

utils = rpackages.importr('utils')
utils.install_packages("Rtools")

#%%

devtools = rpackages.importr("devtools")
rtools = rpackages.importr("Rtools")
devtools.install_github("SydneyBioX/scdney", build_opts = ["--no-resave-data", "--no-manual"])

#%%

# generate some data

np.random.seed(1234)

n = 3

cases = 1
K = 5
n_samples = [n, n]
n_total = np.full(shape=[2*n], fill_value=1000)

data = gen.generate_case_control(cases, K, n_total[0], n_samples,
                                 w_true=np.array([[1, 0, 0, 0, 0]]),
                                 b_true=np.log(np.repeat(0.2, K)).tolist())

print(data.uns["w_true"])
print(data.uns["b_true"])

print(data.X)
print(data.obs)

#%%
importlib.reload(om)

mod = om.scdney_model(data)

ct = rp.vectors.StrVector(mod.scdc_celltypes)
sub = rp.vectors.StrVector(mod.scdc_subject)
cond = rp.vectors.StrVector(mod.scdc_cond)
sc = rp.vectors.StrVector(mod.scdc_sample_cond)


#%%

sum = rp.r(f"""
    library(scdney)
    library(tidyverse)
    library(broom.mixed)
    clust = scDC_noClustering({ct.r_repr()}, {sub.r_repr()},
                                     calCI=TRUE,
                                     calCI_method=c("BCa"),
                                     nboot=10)
    
    glm = fitGLM(clust, {sc.r_repr()}, pairwise=FALSE)
    sum = summary(glm$pool_res_random)
    print(sum)
    sum
    """)

print(sum)

#%%
out = mod.analyze()
print(out)

#%%
importlib.reload(om)

aldex_model = om.ALDEx2Model(data)
aldex_model.fit_model("we.eBH", mc_samples=128)



#%%

importlib.reload(om)
dataset_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\model_comparison\\generated_datasets\\"

file_names = os.listdir(dataset_path)

results = []

simulation_parameters = ["cases", "K", "n_total", "n_samples", "b_true", "w_true", "num_results"]
params = pd.DataFrame(columns=simulation_parameters)

for name in file_names[:3]:
    with open(dataset_path + name, "rb") as f:
        data = pkl.load(f)

        params = params.append(data["parameters"])

        for d in range(len(data["datasets"])):
            mod = om.ALDEx2Model(data["datasets"][d])
            mod.fit_model("we.eBH", mc_samples=128, denom=rp.vectors.FloatVector([5]))

            results.append(mod.result)

#%%
pd.set_option('display.max_columns', 500)


for r in range(len(results)):
    print(params.iloc[r,:])
    print(results[r])



#%%

importlib.reload(om)
importlib.reload(add)
dataset_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\model_comparison\\generated_datasets_new_005\\"

file_names = os.listdir(dataset_path)

results = []

model_name = "scdc"

simulation_parameters = ["cases", "K", "n_total", "n_samples", "b_true", "w_true", "num_results"]
params = pd.DataFrame(columns=simulation_parameters)

col_names = simulation_parameters + ["tp", "tn", "fp", "fn", "model"]
results = pd.DataFrame(columns=col_names)

for name in file_names[7:8]:

    with open(dataset_path+name, "rb") as f:
        data = pkl.load(f)

    res = add.model_on_one_datafile(dataset_path+name, model_name, server=False)

    results = results.append(res)








#%%

# try DirichletReg

# generate some data

np.random.seed(1234)

n = 3

cases = 1
K = 5
n_samples = [n, n]
n_total = np.full(shape=[2*n], fill_value=1000)

data = gen.generate_case_control(cases, K, n_total[0], n_samples,
                                 w_true=np.array([[1, 0, 0, 0, 0]]),
                                 b_true=np.log(np.repeat(0.2, K)).tolist())

print(data.uns["w_true"])
print(data.uns["b_true"])

print(data.X)
print(data.obs)

#%%

counts = dirreg.DR_data(data.X)

r_df = pd.DataFrame(counts, columns=["counts." + i for i in data.var.index])

r_data = pandas2ri.py2rpy_pandasdataframe(r_df)

r_data = r_base.cbind(r_data, pandas2ri.py2rpy_pandasdataframe(data.obs))

print(rp.r("r_data[1]"))

formula = rp.Formula("counts ~ x_0")

fit = dirreg.DirichReg(formula, data=r_data)

print(fit)

#%%

fit = rp.r(f"""
library(DirichletReg)

counts = {pandas2ri.py2rpy_pandasdataframe(pd.DataFrame(data.X, columns=data.var.index)).r_repr()}
counts$counts = DR_data(counts)
print(counts)
data = cbind(counts, {pandas2ri.py2rpy_pandasdataframe(data.obs).r_repr()})
print(data)

fit = DirichReg(counts ~ x_0, data)
u = summary(fit)
pvals = u$coef.mat[grep('Intercept', rownames(u$coef.mat), invert=T), 4]
v = names(pvals)
pvals = matrix(pvals, ncol=length(u$varnames))
rownames(pvals) = gsub('condition', '', v[1:nrow(pvals)])
colnames(pvals) = u$varnames
pvals[,1]
""")

#%%
print(fit)
