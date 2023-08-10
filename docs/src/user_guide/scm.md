```@meta
CurrentModule = TMLE
```

# Structural Causal Models

In TMLE.jl, everything starts from the definition of a Structural Causal Model (`SCM`). A `SCM` in a series of Structural Equations (`SE`) that describe the causal relationships between the random variables under study. The purpose of this package is not to infer the `SCM`, instead we assume it is provided by the user. There are multiple ways one can define a `SCM` that we now describe.

## Incremental Construction

All models are wrong? Well maybe not the following:

```@example scm
using TMLE # hide
scm = SCM()
```

This model does not say anything about the random variables and is thus not really useful. Let's assume that we are interested in an outcome ``Y`` and that this outcome is determined by 8 other random variables. We can add this assumption to the model

```@example scm
push!(scm, SE(:Y, [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁, :W₂₂, :W, :C]))
```

At this point, we haven't made any assumption regarding the functional form of the relationship between ``Y`` and its parents. We can add a further assumption by setting a statistical model for ``Y``, suppose we know it is generated from a logistic model, we can make that explicit:

```@example scm
using MLJLinearModels
setmodel!(scm.Y, LogisticClassifier())
```

---
ℹ️ **Note on Models**

- TMLE.jl is based on the main machine-learning framework in Julia: [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/). As such, any model respecting the MLJ interface is a valid model in TMLE.jl.
- In real world scenarios, we usually don't know what is the true statistical model for each variable and want to keep it as large as possible. For this reason it is recommended to use Super-Learning which is implemented in MLJ by the [Stack](https://alan-turing-institute.github.io/MLJ.jl/dev/model_stacking/#Model-Stacking) and comes with theoretical properties.
- In the dataset, treatment variables are represented with [categorical data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/). This means the models that depend on such variables will need to properly deal with them. For this purpose we provide a `TreatmentTransformer` which can easily be combined with any `model` in a [Pipelining](https://alan-turing-institute.github.io/MLJ.jl/dev/linear_pipelines/) flavour with `with_encoder(model)`.
- The `SCM` has no knowledge of the data and thus cannot verify that the assumed statistical model is compatible with the data. This is done at a later stage.

---

Let's now assume that we have a more complete knowledge of the problem and we also know how `T₁` and `T₂` depend on the rest of the variables in the system.

```@example scm
push!(scm, SE(:T₁, [:W₁₁, :W₁₂, :W], model=LogisticClassifier()))
push!(scm, SE(:T₂, [:W₂₁, :W₂₂, :W]))
```

## One Step Construction

Instead of constructing the `SCM` incrementally, one can provide all the specified equations at once:

```@example scm
scm = SCM(
    SE(:Y, [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁, :W₂₂, :W, :C], with_encoder(LinearRegressor())),
    SE(:T₁, [:W₁₁, :W₁₂, :W], model=LogisticClassifier()),
    SE(:T₂, [:W₂₁, :W₂₂, :W]),
)
```

Noting that we have used the `with_encoder` function to reflect the fact that we know that `T₁` and `T₂` are categorical variables.

## Classic Structural Causal Models

There are many cases where we are interested in estimating the causal effect of a single treatment variable on a single outcome. Because it is typically only necessary to adjust for backdoor variables in order to identify this causal effect, we provide the `StaticConfoundedModel` interface to build such `SCM`:

```@example scm
scm = StaticConfoundedModel(
    :Y, :T, [:W₁, :W₂];
    covariates=[:C],
    outcome_model = with_encoder(LinearRegressor()),
    treatment_model = LogisticClassifier()
)
```

The optional `covariates` are variables that influence the outcome but are not confounding the treatment.

This model can be extended to a plate-model with multiple treatments and multiple outcomes. In this case the set of confounders is assumed to confound all treatments which are in turn assumed to impact all outcomes. This can be defined as:

```@example scm
scm = StaticConfoundedModel(
    [:Y₁, :Y₂], [:T₁, :T₂], [:W₁, :W₂];
    covariates=[:C],
    outcome_model = with_encoder(LinearRegressor()),
    treatment_model = LogisticClassifier()
)
```

More classic `SCM` may be added in the future based on needs.

## Fitting the SCM

It is usually not necessary to fit an entire `SCM` in order to estimate causal estimands of interest. Instead only some components are required and will be automatically determined (see [Estimation](@ref)). However, if you like, you can fit all the equations for which statistical models have been provided against a dataset:

```julia
fit!(scm, dataset)
```
