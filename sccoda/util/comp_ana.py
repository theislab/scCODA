"""
Initialization of scCODA models.

:authors: Johannes Ostner
"""
import numpy as np
import patsy as pt

from anndata import AnnData
from sccoda.model import scCODA_model as dm
from typing import Union, Optional


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
        If "automatic", the cell type with the lowest dispersion in relative abundance that is present in at least 90% of samlpes will be chosen.
    """

    def __new__(
            cls,
            data: AnnData,
            formula: str,
            reference_cell_type: Union[str, int] = "automatic",
            automatic_reference_absence_threshold: float = 0.05,
    ) -> dm.scCODAModel:
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
            If "automatic", the cell type with the lowest dispersion in relative abundance that is present in at least 90% of samlpes will be chosen.
        automatic_reference_absence_threshold
            If using reference_cell_type = "automatic", determine what the maximum fraction of zero entries for a cell type is to be considered as a possible reference cell type

        Returns
        -------
        A compositional model

        model
            A scCODA.models.scCODA_model.CompositionalModel object
        """

        cell_types = data.var.index.to_list()

        # Get count data
        data_matrix = data.X.astype("float64")

        # Build covariate matrix from R-like formula
        covariate_matrix = pt.dmatrix(formula, data.obs)
        covariate_names = covariate_matrix.design_info.column_names[1:]
        covariate_matrix = covariate_matrix[:, 1:]

        # Invoke instance of the correct model depending on reference cell type
        # Automatic reference selection (dispersion-based)
        if reference_cell_type == "automatic":
            percent_zero = np.sum(data_matrix == 0, axis=0)/data_matrix.shape[0]
            nonrare_ct = np.where(percent_zero < automatic_reference_absence_threshold)[0]

            if len(nonrare_ct) == 0:
                raise ValueError("No cell types that have large enough presence! Please increase automatic_reference_absence_threshold")

            rel_abun = data_matrix / np.sum(data_matrix, axis=1, keepdims=True)

            # select reference
            cell_type_disp = np.var(rel_abun, axis=0)/np.mean(rel_abun, axis=0)
            min_var = np.min(cell_type_disp[nonrare_ct])
            ref_index = np.where(cell_type_disp == min_var)[0][0]

            ref_cell_type = cell_types[ref_index]
            print(f"Automatic reference selection! Reference cell type set to {ref_cell_type}")

            return dm.scCODAModel(
                covariate_matrix=np.array(covariate_matrix),
                data_matrix=data_matrix,
                cell_types=cell_types,
                covariate_names=covariate_names,
                reference_cell_type=ref_index,
                formula=formula,
            )

        # Column name as reference cell type
        elif reference_cell_type in cell_types:
            num_index = cell_types.index(reference_cell_type)
            return dm.scCODAModel(
                covariate_matrix=np.array(covariate_matrix),
                data_matrix=data_matrix,
                cell_types=cell_types,
                covariate_names=covariate_names,
                reference_cell_type=num_index,
                formula=formula,
            )

        # Numeric reference cell type
        elif isinstance(reference_cell_type, int) & (reference_cell_type < len(cell_types)) & (reference_cell_type >= 0):
            return dm.scCODAModel(
                covariate_matrix=np.array(covariate_matrix),
                data_matrix=data_matrix,
                cell_types=cell_types,
                covariate_names=covariate_names,
                reference_cell_type=reference_cell_type,
                formula=formula,
            )

        # None of the above: Throw error
        else:
            raise NameError("Reference index is not a valid cell type name or numerical index!")
