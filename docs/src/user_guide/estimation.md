# Estimation

## Estimating a single Estimand

```@setup estimation
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
scm = SCM(
    SE(:Y, [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁, :W₂₂, :C], with_encoder(LinearRegressor())),
    SE(:T₁, [:W₁₁, :W₁₂], LogisticClassifier()),
    SE(:T₂, [:W₂₁, :W₂₂], LogisticClassifier()),
)
```

Once a `SCM` and an estimand have been defined, we can proceed with Targeted Estimation. This is done via the `tmle` function. Drawing from the example dataset and `SCM` from the Walk Through section, we can estimate the ATE for `T₁`.

```@example estimation
Ψ₁ = ATE(scm, outcome=:Y, treatment=(T₁=(case=true, control=false),))
result, fluctuation_mach = tmle(Ψ₁, dataset;
    adjustment_method=BackdoorAdjustment([:C]), 
    verbosity=1, 
    force=false, 
    threshold=1e-8, 
    weighted_fluctuation=false
)
```

We see that both models corresponding to variables `Y` and `T₁` were fitted in the process but that the model for `T₂` was not because it was not necessary to estimate this estimand.

The `fluctuation_mach` corresponds to the fitted machine that was used to fluctuate the initial fit. For instance, we can see what is the value of ``\epsilon`` corresponding to the clever covariate.

```@example estimation
ϵ = fitted_params(fluctuation_mach).coef[1]
```

The `result` corresponds to the estimation result and contains 3 main elements:

- The `TMLEEstimate` than can be accessed via: `tmle(result)`.
- The `OSEstimate` than can be accessed via: `ose(result)`.
- The naive initial estimate.

Since both the TMLE and OSE are asymptotically linear estimators, standard T tests from [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/) can be performed for each of them.

```@example estimation
tmle_test_result = OneSampleTTest(tmle(result))
```

We could now get an interest in the Average Treatment Effect of `T₂`:

```@example estimation
Ψ₂ = ATE(scm, outcome=:Y, treatment=(T₂=(case=true, control=false),))
result, fluctuation_mach = tmle(Ψ₂, dataset;
    adjustment_method=BackdoorAdjustment([:C]), 
    verbosity=1, 
    force=false, 
    threshold=1e-8, 
    weighted_fluctuation=false
)
```

The model for `T₂` was fitted in the process but so was the model for `Y` 🤔. This is because the `BackdoorAdjustment` method determined that the set of inputs for `Y` were different in both cases.

## Reusing the SCM

Let's now see how the models can be reused with a new estimand, say the Total Average Treatment Effecto of both `T₁` and `T₂`.

```@example estimation
Ψ₃ = ATE(scm, outcome=:Y, treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false)))
result, fluctuation_mach = tmle(Ψ₃, dataset;
    adjustment_method=BackdoorAdjustment([:C]), 
    verbosity=1, 
    force=false, 
    threshold=1e-8, 
    weighted_fluctuation=false
)
```

This time only the statistical model for `Y` is fitted again while reusing the models for `T₁` and `T₂`. Finally, let's see what happens if we estimate the `IATE` between `T₁` and `T₂`.

```@example estimation
Ψ₄ = IATE(scm, outcome=:Y, treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false)))
result, fluctuation_mach = tmle(Ψ₄, dataset;
    adjustment_method=BackdoorAdjustment([:C]), 
    verbosity=1, 
    force=false, 
    threshold=1e-8, 
    weighted_fluctuation=false
)
```

All statistical models have been reused 😊!

## Ordering the estimands

Given a vector of estimands, a clever ordering can be obtained via the `optimize_ordering/optimize_ordering!` functions.

```@example estimation
optimize_ordering([Ψ₃, Ψ₁, Ψ₂, Ψ₄]) == [Ψ₁, Ψ₃, Ψ₄, Ψ₂]
```

## Composing Estimands

By leveraging the multivariate Central Limit Theorem and Julia's automatic differentiation facilities, we can estimate any estimand which is a function of already estimated estimands. By default, TMLE.jl will use [Zygote](https://fluxml.ai/Zygote.jl/latest/) but since we are using [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) you can change the backend to your favorite AD system.

For instance, by definition of the ATE, we should be able to retrieve ``ATE_{T_1=0 \rightarrow 1, T_2=0 \rightarrow 1}`` by composing ``CM_{T_1=1, T_2=1} - CM_{T_1=0, T_2=0}``. We already have almost all of the pieces, we just need an estimate for ``CM_{T_1=0, T_2=0}``, let's get it.

```@example estimation
Ψ = CM(
    outcome      = :Y,
    treatment   = (T₁=false, T₂=false),
    confounders = [:W₁, :W₂]
)
cm_result₀₀, _ = tmle(Ψ, η_spec, dataset, verbosity=0)
nothing # hide
```

```@example estimation
composed_ate_result = compose(-, cm_result₁₁.tmle, cm_result₀₀.tmle)
nothing # hide
```

## Weighted Fluctuation

It has been reported that, in settings close to positivity violation (some treatments' values are very rare) TMLE may be unstable. This has been shown to be stabilized by fitting a weighted fluctuation model instead and by slightly modifying the clever covariate to keep things mathematically sound.

This is implemented in TMLE.jl and can be turned on by selecting `weighted_fluctuation=true` in the `tmle` function.
