"""
Initialization of scCODA models.

:authors: Johannes Ostner
"""
import numpy as np
import patsy as pt

from anndata import AnnData
from sccoda.model import dirichlet_models as dm
from typing import Union, Optional
from statsmodels.robust.scale import Huber


class CompositionalAnalysis:
    """
    Initializer class for scCODA models. This class is called when performing compositional analysis with scCODA.

    Usage: model = CompositionalAnalysis(data, formula="covariate1 + covariate2", reference_cell_type="CellTypeA")

    Calling an scCODA model requires these parameters:

    data
        anndata object with cell counts as data.X and covariates saved in data.obs
    formula
        patsy-style formula for building the covariate matrix.
        Categorical covariates are handled automatically, with the covariate value of the first sample being used as the reference category.
        To set a different level as the base category for a categorical covariate, use "C(<CovariateName>, Treatment('<ReferenceLevelName>'))"
    reference_cell_type
        Column index that sets the reference cell type. Can either reference the name of a column or a column number (starting at 0).
        If "automatic", the cell type with the lowest variance in relative abundance that also has a relative abundance
        of at least 0.03 in all samples will be chosen.

    """

    def __new__(
            cls,
            data: AnnData,
            formula: str,
            reference_cell_type: Union[str, int] = "automatic"
    ) -> dm.ReferenceModel:
        """
        Builds count and covariate matrix, returns a CompositionalModel object

        Usage: model = CompositionalAnalysis(data, formula="covariate1 + covariate2", reference_cell_type="CellTypeA")

        Parameters
        ----------
        data
            anndata object with cell counts as data.X and covariates saved in data.obs
        formula
            R-style formula for building the covariate matrix.
            Categorical covariates are handled automatically, with the covariate value of the first sample being used as the reference category.
            To set a different level as the base category for a categorical covariate, use "C(<CovariateName>, Treatment('<ReferenceLevelName>'))"
        reference_cell_type
            Column index that sets the reference cell type. Can either reference the name of a column or the n-th column (indexed at 0).
            If "automatic", the cell type with the lowest variance in relative abundance that also has a relative abundance
            of at least 0.03 in all samples will be chosen.

        Returns
        -------
        A compositional model

        model
            A scCODA.models.dirichlet_models.CompositionalModel object
        """

        cell_types = data.var.index.to_list()

        # Get count data
        data_matrix = data.X.astype("float64")

        # Build covariate matrix from R-like formula
        covariate_matrix = pt.dmatrix(formula, data.obs)
        covariate_names = covariate_matrix.design_info.column_names[1:]
        covariate_matrix = covariate_matrix[:, 1:]

        # Invoke instance of the correct model depending on reference cell type
        # Automatic reference selection
        if reference_cell_type == "Var":
            rel_abun = data_matrix / np.sum(data_matrix, axis=1, keepdims=True)

            # find cell types that always have rel. abundance > 0.03
            not_always_abundant = np.unique(np.where(rel_abun < 0.03)[1])
            is_always_abundant = list(set(np.arange(0, rel_abun.shape[1]).tolist()) - set(not_always_abundant))

            # Exit if non found
            if len(is_always_abundant) == 0:
                raise ValueError(
                    "No abundant cell type found for automatic reference selection! Please choose a reference cell type manually.")

            # select reference
            cell_type_variance = np.var(rel_abun, axis=0)
            min_var = np.min(cell_type_variance[is_always_abundant])
            ref_index = np.where(cell_type_variance == min_var)[0][0]

            ref_cell_type = cell_types[ref_index]
            print(f"Automatic reference selection! Reference cell type set to {ref_cell_type}")

            return dm.ReferenceModel(
                covariate_matrix=np.array(covariate_matrix),
                data_matrix=data_matrix,
                cell_types=cell_types,
                covariate_names=covariate_names,
                reference_cell_type=ref_index,
                formula=formula
            )

        # Dispersion-based selection
        if reference_cell_type == "Disp":
            percent_zero = np.sum(data_matrix == 0, axis=0)/data_matrix.shape[1]
            nonrare_ct = np.where(percent_zero < 0.1)[0]

            rel_abun = data_matrix / np.sum(data_matrix, axis=1, keepdims=True)

            # select reference
            cell_type_disp = np.var(rel_abun, axis=0)/np.mean(rel_abun, axis=0)
            min_var = np.min(cell_type_disp[nonrare_ct])
            ref_index = np.where(cell_type_disp == min_var)[0][0]

            ref_cell_type = cell_types[ref_index]
            print(f"Automatic reference selection (Dispersion)! Reference cell type set to {ref_cell_type}")

            return dm.ReferenceModel(
                covariate_matrix=np.array(covariate_matrix),
                data_matrix=data_matrix,
                cell_types=cell_types,
                covariate_names=covariate_names,
                reference_cell_type=ref_index,
                formula=formula
            )

        # Compositional variance
        if reference_cell_type == "CoVa":
            percent_zero = np.sum(data_matrix == 0, axis=0) / data_matrix.shape[1]
            nonrare_ct = np.where(percent_zero < 0.1)[0]

            # pseudocount if data contains zeros
            if not np.all(data_matrix):
                print("Zeroes encountered! Adding a pseudocount of 1")
                data_matrix_ = data_matrix + 1
            else:
                data_matrix_ = data_matrix

            rel_abun = data_matrix_ / np.sum(data_matrix_, axis=1, keepdims=True)

            def comp_var(c: np.ndarray):
                logratio = np.log(np.outer(c, (1 / c)))
                sample_vars = np.var(logratio)
                return sample_vars

            cv = np.apply_along_axis(comp_var, 0, rel_abun)

            min_var = np.min(cv[nonrare_ct])
            ref_index = np.where(cv == min_var)[0][0]

            ref_cell_type = cell_types[ref_index]
            print(f"Automatic reference selection (Compositional variance)! Reference cell type set to {ref_cell_type}")

            return dm.ReferenceModel(
                covariate_matrix=np.array(covariate_matrix),
                data_matrix=data_matrix,
                cell_types=cell_types,
                covariate_names=covariate_names,
                reference_cell_type=ref_index,
                formula=formula
            )

        # Proportional variability (Heath, 2006)
        if reference_cell_type == "PV":
            percent_zero = np.sum(data_matrix == 0, axis=0) / data_matrix.shape[1]
            nonrare_ct = np.where(percent_zero < 0.1)[0]

            rel_abun = data_matrix / np.sum(data_matrix, axis=1, keepdims=True)

            def prop_var(c: np.ndarray):

                ind = np.transpose(np.triu_indices(c.shape[0], 1))
                C = ind.shape[0]

                D = 0.
                for i in ind:
                    z_1 = c[i[0]]
                    z_2 = c[i[1]]
                    D += np.abs(z_1 - z_2) / np.max([z_1, z_2])
                D = D / C
                return D

            pv = np.apply_along_axis(prop_var, 0, rel_abun)

            min_var = np.min(pv[nonrare_ct])
            ref_index = np.where(pv == min_var)[0][0]

            ref_cell_type = cell_types[ref_index]
            print(f"Automatic reference selection (Proportional variance)! Reference cell type set to {ref_cell_type}")

            return dm.ReferenceModel(
                covariate_matrix=np.array(covariate_matrix),
                data_matrix=data_matrix,
                cell_types=cell_types,
                covariate_names=covariate_names,
                reference_cell_type=ref_index,
                formula=formula
            )

        # Robust variance via Huber M-estimation
        if reference_cell_type == "Var_r":
            percent_zero = np.sum(data_matrix == 0, axis=0) / data_matrix.shape[1]
            nonrare_ct = np.where(percent_zero < 0.1)[0]

            rel_abun = data_matrix / np.sum(data_matrix, axis=1, keepdims=True)

            def huber(x: np.ndarray, k):

                h = Huber(maxiter=100)
                try:
                    loc, scale = h(x)
                except ValueError:
                    print(f"not converged! - {cell_types[k]}")
                    loc, scale = (np.nan, np.nan)

                return loc, scale

            K = rel_abun.shape[1]

            huber_res = [huber(rel_abun[:, a], a) for a in range(K)]
            var_huber = np.array([x[1] for x in huber_res])

            min_var = np.nanmin(var_huber[nonrare_ct])
            ref_index = np.where(var_huber == min_var)[0][0]
            ref_cell_type = cell_types[ref_index]

            print(f"Automatic reference selection (Robust variance)! Reference cell type set to {ref_cell_type}")

            return dm.ReferenceModel(
                covariate_matrix=np.array(covariate_matrix),
                data_matrix=data_matrix,
                cell_types=cell_types,
                covariate_names=covariate_names,
                reference_cell_type=ref_index,
                formula=formula
            )

        # Robust dispersion via Huber M-estimation
        if reference_cell_type == "Disp_r":
            percent_zero = np.sum(data_matrix == 0, axis=0) / data_matrix.shape[1]
            nonrare_ct = np.where(percent_zero < 0.1)[0]

            rel_abun = data_matrix / np.sum(data_matrix, axis=1, keepdims=True)

            def huber(x: np.ndarray, k):

                h = Huber(maxiter=100)
                try:
                    loc, scale = h(x)
                except ValueError:
                    print(f"not converged! - {cell_types[k]}")
                    loc, scale = (np.nan, np.nan)

                return loc, scale

            K = rel_abun.shape[1]

            huber_res = [huber(rel_abun[:, a], a) for a in range(K)]
            mean_huber = np.array([x[0] for x in huber_res])
            var_huber = np.array([x[1] for x in huber_res])
            disp_huber = var_huber / mean_huber

            min_var = np.nanmin(disp_huber[nonrare_ct])
            ref_index = np.where(disp_huber == min_var)[0][0]
            ref_cell_type = cell_types[ref_index]

            print(f"Automatic reference selection (Robust dispersion)! Reference cell type set to {ref_cell_type}")

            return dm.ReferenceModel(
                covariate_matrix=np.array(covariate_matrix),
                data_matrix=data_matrix,
                cell_types=cell_types,
                covariate_names=covariate_names,
                reference_cell_type=ref_index,
                formula=formula
            )

        # Robust compositional variance via Huber M-estimation
        if reference_cell_type == "CoVa_r":
            percent_zero = np.sum(data_matrix == 0, axis=0) / data_matrix.shape[1]
            nonrare_ct = np.where(percent_zero < 0.1)[0]

            # pseudocount if data contains zeros
            if not np.all(data_matrix):
                print("Zeroes encountered! Adding a pseudocount of 1")
                data_matrix_ = data_matrix + 1
            else:
                data_matrix_ = data_matrix

            rel_abun = data_matrix_ / np.sum(data_matrix_, axis=1, keepdims=True)

            def comp_var_huber(x: np.ndarray, k):

                logratio = np.log(np.outer(x, (1 / x)))

                h = Huber(maxiter=100)
                try:
                    loc, scale = h(logratio.flatten())
                except ValueError:
                    print(f"not converged (comp)! - {cell_types[k]}")
                    loc, scale = (np.nan, np.nan)

                return loc, scale

            K = rel_abun.shape[1]

            huber_res_cv = [comp_var_huber(rel_abun[:, a], a) for a in range(K)]
            cv_huber = np.array([x[1] for x in huber_res_cv])

            min_var = np.nanmin(cv_huber[nonrare_ct])
            ref_index = np.where(cv_huber == min_var)[0][0]
            ref_cell_type = cell_types[ref_index]

            print(f"Automatic reference selection (Robust compositional variance)! Reference cell type set to {ref_cell_type}")

            return dm.ReferenceModel(
                covariate_matrix=np.array(covariate_matrix),
                data_matrix=data_matrix,
                cell_types=cell_types,
                covariate_names=covariate_names,
                reference_cell_type=ref_index,
                formula=formula
            )

        # Column name as reference cell type
        if reference_cell_type in cell_types:
            num_index = cell_types.index(reference_cell_type)
            return dm.ReferenceModel(
                covariate_matrix=np.array(covariate_matrix),
                data_matrix=data_matrix,
                cell_types=cell_types,
                covariate_names=covariate_names,
                reference_cell_type=num_index,
                formula=formula
            )

        # Numeric reference cell type
        elif isinstance(reference_cell_type, int) & (reference_cell_type < len(cell_types)) & (reference_cell_type >= 0):
            return dm.ReferenceModel(
                covariate_matrix=np.array(covariate_matrix),
                data_matrix=data_matrix,
                cell_types=cell_types,
                covariate_names=covariate_names,
                reference_cell_type=reference_cell_type,
                formula=formula
            )

        # None of the above: Throw error
        else:
            raise NameError("Reference index is not a valid cell type name or numerical index!")
