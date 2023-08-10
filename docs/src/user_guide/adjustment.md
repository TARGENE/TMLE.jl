```@meta
CurrentModule = TMLE
```
# Adjustment Methods

In a `SCM`, each variable is determined by a set of parents and a statistical model describing the functional relationship between them. However, for the estimation of Causal Estimands the fitted models may not exactly correspond to the variable's equation in the `SCM`. Adjustment methods tell the estimation procedure which input variables should be incorporated in the statistical model fits.

## Backdoor Adjustment

At the moment we provide a single adjustment method, namely the Backdoor adjustment method. The adjustment set consists of all the treatment variable's parents. Additional covariates used to fit the outcome model can be provided via `outcome_extra`.

```julia
BackdoorAdjustment(;outcome_extra=[:C])
```
