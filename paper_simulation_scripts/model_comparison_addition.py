import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys
import patsy as pt

sys.path.insert(0, '/home/icb/johannes.ostner/compositional_diff/SCDCdm/')

from scdcdm.util import data_generation as gen
from scdcdm.model import other_models as om


def model_on_one_datafile(file_path, model_name, fit_args={}, *args, **kwargs):
    with open(file_path, "rb") as f:
        data = pkl.load(f)

    fin_df = data["parameters"]
    fin_df["tp"] = 0
    fin_df["tn"] = 0
    fin_df["fp"] = 0
    fin_df["fn"] = 0
    fin_df["model"] = model_name

    for d in range(len(data["datasets"])):

        if model_name == "Haber":
            mod = om.HaberModel(data["datasets"][d])
            mod.fit_model()
            tp, tn, fp, fn = mod.eval_model(*args, **kwargs)

        elif model_name == "ttest":
            mod = om.TTest(data["datasets"][d])
            mod.fit_model()
            tp, tn, fp, fn = mod.eval_model(*args, **kwargs)

        elif model_name == "clr_ttest":
            mod = om.CLR_ttest(data["datasets"][d])
            mod.fit_model()
            tp, tn, fp, fn = mod.eval_model(*args, **kwargs)

        elif model_name in ["ALDEx2", "ALDEx2_alr"]:
            mod = om.ALDEx2Model(data["datasets"][d])
            mod.fit_model(**fit_args)
            tp, tn, fp, fn = mod.eval_model(*args, **kwargs)

        elif model_name == "alr_ttest":
            mod = om.ALRModel_ttest(data["datasets"][d])
            mod.fit_model(reference_index=4)
            tp, tn, fp, fn = mod.eval_model(*args, **kwargs)

        elif model_name == "alr_wilcoxon":
            mod = om.ALRModel_wilcoxon(data["datasets"][d])
            mod.fit_model(reference_index=4)
            tp, tn, fp, fn = mod.eval_model(*args, **kwargs)

        elif model_name == "ancom":
            mod = om.AncomModel(data["datasets"][d])
            mod.fit_model()
            tp, tn, fp, fn = mod.eval_model()

        elif model_name == "dirichreg":
            mod = om.DirichRegModel(data["datasets"][d])
            mod.fit_model()
            tp, tn, fp, fn = mod.eval_model()

        elif model_name == "simple_dm":
            dat = data["datasets"][d]
            K = dat.X.shape[1]
            formula = "x_0"

            cell_types = dat.var.index.to_list()

            # Get count data
            data_matrix = dat.X.astype("float32")

            # Build covariate matrix from R-like formula
            covariate_matrix = pt.dmatrix(formula, dat.obs)
            covariate_names = covariate_matrix.design_info.column_names[1:]
            covariate_matrix = covariate_matrix[:, 1:]

            mod = om.SimpleModel(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                           cell_types=cell_types, covariate_names=covariate_names, formula=formula,
                                 baseline_index=K-1)

            result_temp = mod.sample_hmc(*args, **kwargs)
            alphas_df, betas_df = result_temp.summary_prepare(credible_interval=0.95)


            final_betas = np.where((betas_df.loc[:, "HDI 3%"] < 0) &
                                   (betas_df.loc[:, "HDI 97%"] > 0),
                                   0,
                                   betas_df.loc[:, "Final Parameter"])

            ks = list(range(K))[1:]

            tp = sum([final_betas[0] != 0])
            fn = sum([final_betas[0] == 0])
            tn = sum([final_betas[k] == 0 for k in ks])
            fp = sum([final_betas[k] != 0 for k in ks])

        else:
            raise ValueError("Invalid model name specified!")

        fin_df.loc[d, "tp"] = tp
        fin_df.loc[d, "fp"] = fp
        fin_df.loc[d, "tn"] = tn
        fin_df.loc[d, "fn"] = fn

    return fin_df


def model_all_datasets(directory, model_name, fit_args={}, *args, **kwargs):

    file_names = os.listdir(directory)

    l = len(file_names)
    k = 1

    simulation_parameters = ["cases", "K", "n_total", "n_samples", "b_true", "w_true", "num_results"]
    col_names = simulation_parameters + ["tp", "tn", "fp", "fn", "model"]
    results = pd.DataFrame(columns=col_names)

    for name in file_names:
        print(f"{k}/{l}")

        res = model_on_one_datafile(directory+name, model_name, fit_args, *args, **kwargs)

        results = results.append(res)
        k += 1

    return results


def get_scores(df):
    """
    Calculates extended summary statistics, such as TPR, TNR, youden index, f1-score, MCC

    """
    tp = df["tp"].astype("float64")
    tn = df["tn"].astype("float64")
    fp = df["fp"].astype("float64")
    fn = df["fn"].astype("float64")

    tpr = (tp / (tp + fn)).fillna(0)
    df["tpr"] = tpr
    tnr = (tn / (tn + fp)).fillna(0)
    df["tnr"] = tnr
    precision = (tp / (tp + fp)).fillna(0)
    df["precision"] = precision
    acc = (tp + tn) / (tp + tn + fp + fn).fillna(0)
    df["accuracy"] = acc

    df["youden"] = tpr + tnr - 1
    df["f1_score"] = 2 * (tpr * precision / (tpr + precision)).fillna(0)

    df["mcc"] = (((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))).fillna(0)

    return df


def complete_results(results_df):

    # Find relations between numerical values and compsition/inrease vectors
    b = []
    for y1_0 in [200, 400, 600, 800, 1000]:
        b_i = np.round(gen.counts_from_first(y1_0, 5000, 5), 3)
        b.append(np.round(np.log(b_i / 5000), 2))

    b_counts = dict(zip([b_i[0] for b_i in b], [200, 400, 600, 800, 1000]))

    b2 = []
    for y1_0 in [200, 400, 600, 800, 1000]:
        b_i = np.round(gen.counts_from_first(y1_0, 5000, 5), 3)
        b2.append(b_i)

    b_w_dict = {}
    w_rel_dict = {}
    i = 0
    for b_i in b2:
        b_t = np.round(np.log(b_i / 5000), 3)
        w_d = {}
        for change in [b_i[0] / 3, b_i[0] / 2, b_i[0], b_i[0] * 2, b_i[0] * 3]:
            _, w = gen.b_w_from_abs_change(b_i, change, 5000)
            w_0 = np.round(w[0], 3)
            w_d[w_0] = change
            rel_change = np.round(change / b_i[0], 2)
            w_rel_dict[w_0] = rel_change
        b_w_dict[b_t[0]] = w_d
        i += 1

    results_df["n_controls"] = [x[0] for x in results_df["n_samples"].tolist()]
    results_df["n_cases"] = [x[1] for x in results_df["n_samples"].tolist()]
    results_df["n_total"] = results_df["n_total"].astype("float")
    results_df["w"] = [x[0][0] for x in results_df["w_true"]]
    results_df["b_0"] = [x[0] for x in results_df["b_true"]]
    results_df["b_count"] = [b_counts[np.round(x, 2)] for x in results_df["b_0"]]

    bs = results_df["b_0"].tolist()
    ws = results_df["w"].tolist()
    increases = []
    rel_changes = []
    for i in range(len(bs)):
        increases.append(b_w_dict[bs[i]][ws[i]])
        rel_changes.append(w_rel_dict[ws[i]])
    results_df["num_increase"] = increases
    results_df["rel_increase"] = rel_changes
    results_df["log_fold_increase"] = np.round(np.log2((results_df["num_increase"] +
                                                        results_df["b_count"]) /
                                                       results_df["b_count"]), 2)

    param_cols = ["b_count", "num_increase", "n_controls", "n_cases", "log_fold_increase"]
    metrics = ["tpr", "tnr", "precision", "accuracy", "youden", "f1_score", "mcc"]
    results_df = results_df.loc[:, param_cols + metrics + ["model"]]

    results_df = results_df.rename(columns={"b_count": "Base", "num_increase": "Increase", "model": "Model",
                                   "log_fold_increase": "log-fold increase"})

    return results_df
