# Missingness

Strictly speaking, there is not much missingness support at the moment, rows containing missing values in the variables necessary for estimation are dropped.

However, when no cross-validation scheme is used we make sure the propensity score estimation uses as much data as possible. This is because the propensity score ``g(T, W) = p(T|W)`` only depends on ``(T, W)``. That is, missing values in the outcome ``Y`` or extra covariates ``C`` should not impact its estimation. 

If you estimate the effect of some treatment on multiple outcomes, this can be handy because you can make maximal use of the propensity score's related data and reuse it across all outcomes (with the cache).