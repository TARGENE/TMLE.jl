# Miscellaneous

## Adjustment Methods

An adjustment method is a function that transforms a causal estimand into a statistical estimand using an associated `SCM`. At the moment, the only available adjustment method is the backdoor adjustment.

### Backdoor Adjustment

The adjustment set consists of all the treatment variable's parents. Additional covariates used to fit the outcome model can be provided via `outcome_extra`.

```julia
BackdoorAdjustment(;outcome_extra_covariates=[:C])
```

## Serialization

Many objects from TMLE.jl can be serialized to various file formats. This is achieved by transforming these structures to dictionaries that can then be serialized to classic JSON or YAML format. For that purpose you can use the `TMLE.read_json`, `TMLE.write_json`, `TMLE.read_yaml` and `TMLE.write_yaml` functions.
