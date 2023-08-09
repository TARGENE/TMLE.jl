# Walk Through

The goal of this section is to provide a comprehensive (but non-exhaustive) illustration of the estimation process provided in TMLE.jl. For an in-depth explanation, please refer to the User Guide.

## The Dataset

TMLE.jl is compatible with any dataset respecting the [Tables.jl](https://tables.juliadata.org/stable/) interface, that is for instance, a `NamedTuple`, a `DataFrame`, an `Arrow.Table` etc... In this section, we will be working with the same dataset all along.

⚠️ One thing to note is that treatment variables as well as binary outcomes **must** be encoded as `categorical` variables in the dataset (see [MLJ Working with categorical data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/)).

The dataset is generated as follows:

```@example user-guide
using Random
using Distributions
using DataFrames
using StableRNGs
using CategoricalArrays
using TMLE
using LogExpFunctions

function make_dataset(;n=1000)
    rng = StableRNG(123)
    # Confounders
    W₁₁= rand(rng, Uniform(), n)
    W₁₂ = rand(rng, Uniform(), n)
    W₂₁= rand(rng, Uniform(), n)
    W₂₂ = rand(rng, Uniform(), n)
    # Covariates
    C = rand(rng, Uniform(), n)
    # Treatment | Confounders
    T₁ = rand(rng, Uniform(), n) .< logistic.(0.5sin.(W₁₁) .- 1.5W₁₂)
    T₂ = rand(rng, Uniform(), n) .< logistic.(-3W₂₁ - 1.5W₂₂)
    # Target | Confounders, Covariates, Treatments
    Y = 1 .+ 2W₂₁ .+ 3W₂₂ .+ W₁₁ .- 4C.*T₁ .- 2T₂.*T₁.*W₁₂ .+ rand(rng, Normal(0, 0.1), n)
    return DataFrame(
        W₁₁ = W₁₁, 
        W₁₂ = W₁₂,
        W₂₁ = W₂₁,
        W₂₂ = W₂₂,
        C   = C,
        T₁  = categorical(T₁),
        T₂  = categorical(T₂),
        Y   = Y
        )
end
dataset = make_dataset()
nothing # hide
```

Even though the role of a variable (treatment, outcome, confounder, ...) is relative to the problem setting, this dataset can intuitively be decomposed into:

- 1 Outcome variable (``Y``).
- 2 Treatment variables ``(T₁, T₂)`` with confounders ``(W₁₁, W₁₂)`` and ``(W₂₁, W₂₂)`` respectively.
- 1 Extra Covariate variable (``C``).

## The Structural Causal Model

The modeling stage starts from the definition of a Structural Causal Model (`SCM`). This is simply a list of Structural Equations (`SE`) describing the relationships between the random variables associated with our problem. See [Structural Causal Models](@ref) for an in-depth explanation. For our purposes, we will simply define it as follows:

```@example user-guide
scm = SCM(
    SE(:Y, [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁, :W₂₂, :C], with_encoder(LinearRegressor())),
    SE(:T₁, [:W₁₁, :W₁₂], LogisticClassifier()),
    SE(:T₂, [:W₂₁, :W₂₂], LogisticClassifier()),
)
```

---
**NOTE**

- Each Structural Equation specifies a child node, its parents and the assumed relationship between them. Here we know the model class from which each variable has been generated but in practice this is usually not the case. Instead we recommend the use of Super Learning / Stacking (see TODO).
- Because the treatment variables are categorical, they need to be encoded to be digested by the downstream models, `with_encoder(model)` simply creates a [Pipeline](https://alan-turing-institute.github.io/MLJ.jl/dev/linear_pipelines/#Linear-Pipelines) equivalent to `TreatmentEncoder() |> model`.
- At this point, the `SCM` has no knowledge about the dataset and cannot verify that the models are compatible with the actual data.

---

## The Estimands

From the previous causal model we can ask multiple causal questions which are each represented by a distinct estimand. The set of available estimands types can be listed as follow:

```@example user-guide
AVAILABLE_ESTIMANDS
```

At the moment there are 3 main estimand types we can estimate in TMLE.jl, we provide below a few examples.

- The Interventional Conditional Mean (see: TODO):

```@example user-guide
cm = CM(
    scm,
    outcome=:Y,
    treatment=(T₁=1,) 
    )
```

- The Average Treatment Effect (see: TODO):

```@example user-guide
ate = ATE(
    scm,
    outcome=:Y,
    treatment=(T₁=(case=1, control=0), T₂=(case=1, control=0)) 
)
marginal_ate_t1 = ATE(
    scm,
    outcome=:Y,
    treatment=(T₁=(case=1, control=0),) 
)
```

- The Interaction Average Treatment Effect (see: TODO):

```@example user-guide
iate = IATE(
    scm,
    outcome=:Y,
    treatment=(T₁=(case=1, control=0), T₂=(case=1, control=0)) 
)
```

## Targeted Estimation

Then each parameter can be estimated by calling the `tmle` function. For example:

```@example user-guide
result, _ = tmle(cm, dataset)
result
```

The `result` contains 3 main elements:

- The `TMLEEstimate` than can be accessed via: `tmle(result)`.
- The `OSEstimate` than can be accessed via: `ose(result)`.
- The naive initial estimate.

The adjustment set is determined by the provided `adjustment_method` keyword. At the moment, only `BackdoorAdjustment` is available. However one can specify that extra covariates could be used to fit the outcome model.

```@example user-guide
result, _ = tmle(iate, dataset;adjustment_method=BackdoorAdjustment([:C]))
result
```

## Hypothesis Testing

Because the TMLE and OSE are asymptotically linear estimators, they asymptotically follow a Normal distribution. This means one can perform standard T tests of null hypothesis. TMLE.jl extends the method provided by the [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/) package that can be used as follows.

```@example user-guide
OneSampleTTest(tmle(result))
```
