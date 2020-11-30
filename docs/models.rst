Models
======

scCODA uses Bayesian modeling to detect changes in compositional data.
The model is implemented in ``sccoda.model.dirichlet_models``.
Te easiest way to call a compositional model is via calling an instance of ``sccoda.util.comp_ana.CompositionalAnalysis``, which returns a compositional model
scCODA automatically selects the correct model based on whether a baseline cell type was specified.

Model structure
~~~~~~~~~~~~~~~

The model is based on a Dirichlet-multinomial model, in which each cell type is described by the covariates through a log-linear model.
The intercepts alpha are modeled via a normal prior. For the effects (beta) of a covariate on a cell type, we perform model selection via a spike-and-slab prior (Continuous approximation via a Logit-normal prior).
The underlying prior for significant effects is a noncentered parametrization of a Normal distribution.

For further information regarding the model structure, please refer to:
**scCODA: A Bayesian model for compositional single-cell data analysis (Ostner et al., 2020)**


Inference
~~~~~~~~~

Once the model is set up, optimization via HMC sampling can be performed via ``sample_hmc()``.
This produces a ``sccoda.util.result_classes.CAResult`` object.


Result analysis
~~~~~~~~~~~~~~~

To see which effects were found to be significant, call ``summary()`` on your ``sccoda.util.result_classes.CAResult`` object.
The final_parameter column of the effects data frame shows the significances. If the value is 0, the effect is not significant, otherwise it is.

Furthermore, ``sccoda.util.result_classes.CAResult`` supports all functionalities af arviz's ``InferenceData``.
