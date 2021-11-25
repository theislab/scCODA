The scCODA model
================

scCODA uses Bayesian modeling to detect statistically credible changes in compositional data.
The model is implemented in ``sccoda.model.scCODA_model``.
The easiest way to call a compositional model is via calling an instance of ``sccoda.util.comp_ana.CompositionalAnalysis``.
It requires an ``anndata`` object that contains the compositional data and covariates, a formula string that defines the covariate matrix
(see the `patsy <https://patsy.readthedocs.io/en/latest/>`_ syntax for details), and a reference cell type.


Model structure
^^^^^^^^^^^^^^^

The model is based on a Dirichlet-multinomial model, in which each cell type is described by the covariates through a log-linear linkage.
The intercepts :math:`\alpha` are modeled via a normal prior.
For the effect (:math:`\beta`) of a covariate on a cell type, scCODA performs model selection via a spike-and-slab prior (Continuous approximation via a Logit-normal prior).
The underlying prior for significant effects is normal with a covariate-specific scaling factor.
The only exception are the effects of the reference cell type :math:`\hat{k}`, which are always set to 0.

.. math::
         y|x &\sim DirMult(\phi, \bar{y}) \\
         \log(\phi) &= \alpha + x \beta \\
         \alpha_k &\sim N(0, 5) \quad &\forall k \in [K] \\
         \beta_{m, \hat{k}} &= 0 &\forall m \in [M]\\
         \beta_{m, k} &= \tau_{m, k} \tilde{\beta}_{m, k} \quad &\forall m \in [M], k \in \{[K] \smallsetminus \hat{k}\} \\
         \tau_{m, k} &= \frac{\exp(t_{m, k})}{1+ \exp(t_{m, k})} \quad &\forall m \in [M], k \in \{[K] \smallsetminus \hat{k}\} \\
         \frac{t_{m, k}}{50} &\sim N(0, 1) \quad &\forall m \in [M], k \in \{[K] \smallsetminus \hat{k}\} \\
         \tilde{\beta}_{m, k} &= \sigma_m^2 \cdot \gamma_{m, k} \quad &\forall m \in [M], k \in \{[K] \smallsetminus \hat{k}\} \\
         \sigma_m^2 &\sim HC(0, 1) \quad &\forall m \in [M] \\
         \gamma_{m, k} &\sim N(0,1) \quad &\forall m \in [M], k \in \{[K] \smallsetminus \hat{k}\} \\


For further information regarding the model structure, please refer to:

Büttner, Ostner *et al.* (2021), scCODA is a Bayesian model for compositional single-cell data analysis
`NatComms <https://www.nature.com/articles/s41467-021-27150-6>`_.

Inference
^^^^^^^^^

Once the model is set up, inference via HMC sampling can be performed via ``sample_hmc()``.
Alternatively, No-U-Turn sampling is available via ``sample_nuts()``.
Depending on the size of the dataset and the system hardware, inference usually takes up to 5 minutes.
The resulting ``sccoda.util.result_classes.CAResult`` object extends the ``InferenceData`` class of
`arviz <https://arviz-devs.github.io/arviz/>`_ and supports all its diagnostic and plotting functionality.


Result analysis
^^^^^^^^^^^^^^^

To see which effects were found to be significant, call ``summary()`` on the result object.
The ``Final Parameter`` column of the effects data frame shows the significances.
If the value is 0, the effect is not found to be statistically credible, otherwise it is.
The sign of the effect indicates a decrease or increase in abundance (relative to the reference cell type).
However, the numerical value of these effects should not be used for analysis, as it depends on multiple parameters.
Please refer to the tutorials for more information on how to evaluate scCODA's results.
