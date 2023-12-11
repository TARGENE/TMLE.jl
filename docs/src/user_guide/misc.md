# Miscellaneous

## Adjustment Methods

An adjustment method is a function that transforms a causal estimand into a statistical estimand using an associated `SCM`. At the moment, the only available adjustment method is the backdoor adjustment.

### Backdoor Adjustment

The adjustment set consists of all the treatment variable's parents. Additional covariates used to fit the outcome model can be provided via `outcome_extra`.

```julia
BackdoorAdjustment(;outcome_extra_covariates=[:C])
```

## Treatment Transformer

To account for the fact that treatment variables are categorical variables we provide a MLJ compliant transformer that will either:

- Retrieve the floating point representation of a treatment if it has a natural ordering
- One hot encode it otherwise

Such transformer can be created with:

```julia
TreatmentTransformer(;encoder=encoder())
```

where `encoder` is a [OneHotEncoder](https://alan-turing-institute.github.io/MLJ.jl/dev/models/OneHotEncoder_MLJModels/#OneHotEncoder_MLJModels).

The `with_encoder(model; encoder=TreatmentTransformer())` provides a shorthand to combine a `TreatmentTransformer` with another MLJ model in a pipeline.

Of course you are also free to define your own strategy!

## Serialization

Many objects from TMLE.jl can be serialized to various file formats. This is achieved by transforming these structures to dictionaries that can then be serialized to classic JSON or YAML format. For that purpose you can use the `TMLE.read_json`, `TMLE.write_json`, `TMLE.read_yaml` and `TMLE.write_yaml` functions.
