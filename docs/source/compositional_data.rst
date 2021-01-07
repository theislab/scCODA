About scCODA
============

Various biological factors, such as diseases, aging, and immunity, are known to have significant effects on the
cellular structure on a wide range of tissues. Thus, studying these changes more carefully is of particular interest
for many research questions. Recent advances in single-cell RNA sequencing technologies open up the possibility of
accurately annotating large numbers of individual cells from a tissue sample, paving the way for differential analysis
of cell populations.

Compositional data analysis in scRNA-seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When doing differential population analysis, one property of cell population data is often overlooked. Since all
single-cell analysis platforms are limited in their throughput, the number of individual cells in a sample is
predetermined. Thus, cell populations are compositional. They can only be determined up to a multiplicative factor, inducing a negative
correlative bias between the cell types. Following
`Aitchison (Journal of the Royal Statistical Society, 1982) <https://www.jstor.org/stable/2345821?seq=1>`_,
compositional data also has to be interpreted in terms of ratios, e.g. with respect to a reference factor.

Features of scCODA
^^^^^^^^^^^^^^^^^^

The scCODA model (`Büttner, Ostner et al. (2020) <https://www.biorxiv.org/content/10.1101/2020.12.14.422688v2>`_)
is the first model that was specifically designed to perform compositional data analysis in scRNA-seq.
Apart from the compositionality of cell population data, there are some other challenges in comparing scRNA-seq
populations, which scCODA addresses. It allows the user to select any reference cell type in order to see the effects
of biological factors from different perspectives.

Because each sample contains up to thousands of cells, performing scRNA-seq on a large number of samples is expensive
and time-consuming. Thus, there are often very few biological replicates available, and frequentist tests will
therefore result in highly uncertain estimates with large confidence intervals. scCODA uses Bayesian
modeling with its possibility to include prior beliefs to obtain accurate results even in a low-sample setting.

Also, most biological factors only effect a fraction of the total cell population. It is therefore important to
determine the most important changes during the analysis. Since Bayesian analysis does not support the concept
of p-values, scCODA instead uses spike-and-slab priors to automatically determine statistically credible effects.

For more detailed information on the scCODA model, see
`Büttner, Ostner et al. (2020) <https://www.biorxiv.org/content/10.1101/2020.12.14.422688v2>`_.

