Models
======

CompositionalDiff uses two different models to detect changes in compositional data.
These are implemented in ``sccoda.model.dirichlet_models`` and differ whether a baseline (reference) cell type is used or not.
Both models can be called via creating an instance of ``sccoda.util.comp_ana.CompositionalAnalysis``.
scCODA automatically selects the correct model based on whether a baseline cell type was specified.
f
Model structure
~~~~~~~~~~~~~~~

Both models are based on a Dirichlet-multinomial model, in which each cell type is described by the covariates through a linear model.
The intercepts alpha are modeled via a normal prior. For the effects (beta) of a covariate on a cell type, we perform model selection via a spike-and-slab prior (Continuous approximation via a Logit-normal prior).
The underlying prior for significant effects is a noncentered parametrization of a Normal distribution.

For further information regarding the model structure, please refer to: (paper)


Inference
~~~~~~~~~

Once the model is set up, optimization via HMC sampling can be performed via ``sample_hmc()``.
This produces a ``sccoda.util.result_classes.CAResult`` object.


Result analysis
~~~~~~~~~~~~~~~

To see which effects were found to be significant, call ``summary()`` on your ``sccoda.util.result_classes.CAResult`` object.
The final_parameter column of the effects data frame shows the significances. If the value is 0, the effect is not significant, otherwise it is.

Furthermore, ``sccoda.util.result_classes.CAResult`` supports all functionalities af arviz's ``InferenceData``.
