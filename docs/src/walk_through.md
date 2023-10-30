```@meta
CurrentModule = TMLE
```

# Walk Through

The goal of this section is to provide a comprehensive (but non-exhaustive) illustration of the estimation process provided in TMLE.jl. For an in-depth explanation, please refer to the User Guide.

## The Dataset

TMLE.jl is compatible with any dataset respecting the [Tables.jl](https://tables.juliadata.org/stable/) interface, that is for instance, a `NamedTuple`, a `DataFrame`, an `Arrow.Table` etc... In this section, we will be working with the same dataset all along.

⚠️ One thing to note is that treatment variables as well as binary outcomes **must** be encoded as `categorical` variables in the dataset (see [MLJ Working with categorical data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/)).

The dataset is generated as follows:

```@example walk-through
using TMLE
using Random
using Distributions
using DataFrames
using StableRNGs
using CategoricalArrays
using TMLE
using LogExpFunctions
using MLJLinearModels

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

```@example walk-through
scm = SCM(
    :Y  => [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁, :W₂₂, :C],
    :T₁ => [:W₁₁, :W₁₂],
    :T₂ => [:W₂₁, :W₂₂]
)
```

## The Causal Estimands

From the previous causal model we can ask multiple causal questions which are each represented by distinct causal estimands. The set of available estimands types can be listed as follow:

```@example walk-through
AVAILABLE_ESTIMANDS
```

At the moment there are 3 main causal estimands in TMLE.jl, we provide below a few examples.

- The Counterfactual Mean:

```@example walk-through
cm = CM(
    outcome = :Y,
    treatment_values = (T₁=true,) 
)
```

- The Average Treatment Effect:

```@example walk-through
total_ate = ATE(
    outcome = :Y,
    treatment_values = (
        T₁=(case=1, control=0), 
        T₂=(case=1, control=0)
    ) 
)
marginal_ate_t1 = ATE(
    outcome = :Y,
    treatment_values = (T₁=(case=1, control=0),) 
)
```

- The Interaction Average Treatment Effect:

```@example walk-through
iate = IATE(
    outcome = :Y,
    treatment_values = (
        T₁=(case=1, control=0), 
        T₂=(case=1, control=0)
    ) 
)
```

## Identification

Identification is the process by which a Causal Estimand is turned into a Statistical Estimand, that is, a quantity we may estimate from data. This is done via the `identify` function which also takes in the ``SCM``:

```@example walk-through
statistical_iate = identify(iate, scm)
```

Alternatively, you can also directly define the statistical parameters (see [Estimands](@ref)).

## Estimation

Then each parameter can be estimated by building an estimator (which is simply a function) and calling it on data. For illustration, we will keep the models used simple linear models. We define a TMLE:

```@example walk-through
models = (
    Y  = with_encoder(LinearRegressor()),
    T₁ = LogisticClassifier(),
    T₂ = LogisticClassifier()
)
tmle = TMLEE(models)
```

And estimate the `cm` causal parameter. Because we haven't identified the parameter, we need to provide the `scm` as well:

```@example walk-through
result, cache = tmle(cm, scm, dataset);
result
```

We can estimate statistical parameters directly, let's use the One-Step estimator:

```@example walk-through
ose = OSE(models)
result, cache = ose(statistical_iate, dataset)
result
```

## Hypothesis Testing

Because the TMLE and OSE are asymptotically linear estimators, they asymptotically follow a Normal distribution. This means one can perform standard T/Z tests of null hypothesis. TMLE.jl extends the method provided by the [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/) package that can be used as follows.

```@example walk-through
OneSampleTTest(result)
```
