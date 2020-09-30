import os
os.environ["R_HOME"] = "C:\\Program Files\\R\\R-4.0.2"
os.environ["PATH"] = r"C:\\Program Files\\R\\R-4.0.2\\bin\\x64" + ";" + os.environ["PATH"]

import numpy as np
import pandas as pd
import importlib
import pickle as pkl

from scdcdm.util import data_generation as gen
from scdcdm.model import other_models as om
from paper_simulation_scripts import model_comparison_addition as add

import rpy2.robjects as rp
from rpy2.robjects import numpy2ri, pandas2ri
numpy2ri.activate()
pandas2ri.activate()
r = rp.r

import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

aldex2 = rpackages.importr("ALDEx2")
tv = rpackages.importr("tidyverse")

#%%

# r install procedure

utils = rpackages.importr('utils')
utils.install_packages("tidyverse")

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

cond = rp.vectors.FloatVector(data.obs["x_0"].astype("string").tolist())

X_t = data.X.T
nr, nc = X_t.shape
X_r = rp.r.matrix(X_t, nrow=nr, ncol=nc)

aldex_out = aldex2.aldex(X_r, cond, mc_samples=128)
aldex_out = pd.DataFrame(aldex_out)

#%%

sig_effects = (aldex_out.loc[:, "we.eBH"] < 0.05)

K = data.X.shape[1]
ks = list(range(K))[1:]

tp = sum([sig_effects.loc[0] == True])
fn = sum([sig_effects.loc[0] == False])
tn = sum([sig_effects.loc[k] == False for k in ks])
fp = sum([sig_effects.loc[k] == True for k in ks])

print((tp, tn, fp, fn))

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

for name in file_names[:5]:
    with open(dataset_path + name, "rb") as f:
        data = pkl.load(f)

        params = params.append(data["parameters"])

        for d in range(len(data["datasets"])):
            mod = om.ALDEx2Model(data["datasets"][d])
            mod.fit_model("we.eBH", mc_samples=128)

            results.append(mod.result)

#%%
pd.set_option('display.max_columns', 500)


for r in range(len(results)):
    print(params.iloc[r,:])
    print(results[r])