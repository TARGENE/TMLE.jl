```@meta
CurrentModule = TMLE
```

# Estimation

## Constructing and Using Estimators

```@setup estimation
using Random
using Distributions
using DataFrames
using StableRNGs
using CategoricalArrays
using TMLE
using LogExpFunctions
using MLJLinearModels
using MLJ

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
dataset = make_dataset(n=10000)
scm = SCM([
    :Y  => [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁, :W₂₂, :C],
    :T₁ => [:W₁₁, :W₁₂],
    :T₂ => [:W₂₁, :W₂₂]
]
)
```

Once a statistical estimand has been defined, we can proceed with estimation. There are two semi-parametric efficient estimators in TMLE.jl:

- The Targeted Maximum-Likelihood Estimator (`TMLEE`)
- The One-Step Estimator (`OSE`)

While they have similar asymptotic properties, their finite sample performance may be different. They also have a very distinguishing feature, the TMLE is a plugin estimator, which means it respects the natural bounds of the estimand of interest. In contrast, the OSE may in theory report values outside these bounds. In practice, this is not often the case and the estimand of interest may not impose any restriction on its domain.

Drawing from the example dataset and `SCM` from the Walk Through section, we can estimate the ATE for `T₁`. Let's use TMLE:

```@example estimation
Ψ₁ = ATE(
    outcome=:Y, 
    treatment_values=(T₁=(case=true, control=false),),
    treatment_confounders=(T₁=[:W₁₁, :W₁₂],),
    outcome_extra_covariates=[:C]
)
tmle = TMLEE()
result₁, cache = tmle(Ψ₁, dataset);
result₁
nothing # hide
```

The `cache` (see below) contains estimates for the nuisance functions that were necessary to estimate the ATE. For instance, we can see what is the value of ``\epsilon`` corresponding to the clever covariate.

```@example estimation
ϵ = last_fluctuation_epsilon(cache)
```

The `result₁` structure corresponds to the estimation result and will display the result of a T-Test including:

- A point estimate.
- A 95% confidence interval.
- A p-value (Corresponding to the test that the estimand is different than 0).

Both the TMLE and OSE are asymptotically linear estimators, standard Z/T tests from [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/) can be performed and `confint` and `pvalue` methods used.

```@example estimation
tmle_test_result₁ = pvalue(OneSampleTTest(result₁))
```

Let us now turn to the Average Treatment Effect of `T₂`, we will estimate it with a `OSE`:

```@example estimation
Ψ₂ = ATE(
    outcome=:Y, 
    treatment_values=(T₂=(case=true, control=false),),
    treatment_confounders=(T₂=[:W₂₁, :W₂₂],),
    outcome_extra_covariates=[:C]
)
ose = OSE()
result₂, cache = ose(Ψ₂, dataset;cache=cache);
result₂
nothing # hide
```

Again, required nuisance functions are fitted and stored in the cache.

## Specifying Models

By default, TMLE.jl uses generalized linear models for the estimation of relevant and nuisance factors such as the outcome mean and the propensity score. However, this is not the recommended usage since the estimators' performance is closely related to how well we can estimate these factors. More sophisticated models can be provided using the `models` keyword argument of each estimator which is essentially a `NamedTuple` mapping variables' names to their respective model.

Rather than specifying a specific model for each variable it may be easier to override the default models using the `default_models` function:

For example one can override all default models with XGBoost models from `MLJXGBoostInterface`:

```@example estimation
using MLJXGBoostInterface
xgboost_regressor = XGBoostRegressor()
xgboost_classifier = XGBoostClassifier()
models = default_models(
    Q_binary=xgboost_classifier,
    Q_continuous=xgboost_regressor,
    G=xgboost_classifier
)
tmle_gboost = TMLEE(models=models)
```

The advantage of using `default_models` is that it will automatically prepend each model with a [ContinuousEncoder](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/#MLJModels.ContinuousEncoder) to make sure the correct types are passed to the downstream models.

Super Learning ([Stack](https://alan-turing-institute.github.io/MLJ.jl/dev/model_stacking/#Model-Stacking)) as well as variable specific models can be defined as well. Here is a more customized version:

```@example estimation
lr = LogisticClassifier(lambda=0.)
stack_binary = Stack(
    metalearner=lr,
    xgboost=xgboost_classifier,
    lr=lr
)

models = (
    T₁ = with_encoder(xgboost_classifier), # T₁ with XGBoost prepended with a Continuous Encoder
    default_models( # For all other variables use the following defaults
        Q_binary=stack_binary, # A Super Learner
        Q_continuous=xgboost_regressor, # An XGBoost
        # Unspecified G defaults to Logistic Regression
    )...
)

tmle_custom = TMLEE(models=models)
```

Notice that `with_encoder` is simply a shorthand to construct a pipeline with a `ContinuousEncoder` and that the resulting `models` is simply a `NamedTuple`.

## CV-Estimation

Canonical TMLE/OSE are essentially using the dataset twice, once for the estimation of the nuisance functions and once for the estimation of the parameter of interest. This means that there is a risk of over-fitting and residual bias ([see here](https://arxiv.org/abs/2203.06469) for some discussion). One way to address this limitation is to use a technique called sample-splitting / cross-validating. In order to activate the sample-splitting mode, simply provide a `MLJ.ResamplingStrategy` using the `resampling` keyword argument:

```@example estimation
TMLEE(resampling=StratifiedCV());
```

or

```julia
OSE(resampling=StratifiedCV(nfolds=3));
```

There are some practical considerations

- Choice of `resampling` Strategy: The theory behind sample-splitting requires the nuisance functions to be sufficiently well estimated on **each and every** fold. A practical aspect of it is that each fold should contain a sample representative of the dataset. In particular, when the treatment and outcome variables are categorical it is important to make sure the proportions are preserved. This is typically done using `StratifiedCV`.
- Computational Complexity: Sample-splitting results in ``K`` fits of the nuisance functions, drastically increasing computational complexity. In particular, if the nuisance functions are estimated using (P-fold) Super-Learning, this will result in two nested cross-validation loops and ``K \times P`` fits.
- Caching of Nuisance Functions: Because the `resampling` strategy typically needs to preserve the outcome and treatment proportions, very little reuse of cached models is possible (see [Caching Models](@ref)).

## Caching Models

Let's now see how the `cache` can be reused with a new estimand, say the Total Average Treatment Effect of both `T₁` and `T₂`.

```@example estimation
Ψ₃ = ATE(
    outcome=:Y, 
    treatment_values=(
        T₁=(case=true, control=false), 
        T₂=(case=true, control=false)
    ),
    treatment_confounders=(
        T₁=[:W₁₁, :W₁₂], 
        T₂=[:W₂₁, :W₂₂],
    ),
    outcome_extra_covariates=[:C]
)
result₃, cache = tmle(Ψ₃, dataset; cache=cache);
result₃
nothing # hide
```

This time only the model for `Y` is fitted again while reusing the models for `T₁` and `T₂`. Finally, let's see what happens if we estimate the `IATE` between `T₁` and `T₂`.

```@example estimation
Ψ₄ = IATE(
    outcome=:Y, 
    treatment_values=(
        T₁=(case=true, control=false), 
        T₂=(case=true, control=false)
    ),
    treatment_confounders=(
        T₁=[:W₁₁, :W₁₂], 
        T₂=[:W₂₁, :W₂₂],
    ),
    outcome_extra_covariates=[:C]
)
result₄, cache = tmle(Ψ₄, dataset; cache=cache);
result₄
nothing # hide
```

All nuisance functions have been reused, only the fluctuation is fitted!

## Composing Estimands

By leveraging the multivariate Central Limit Theorem and Julia's automatic differentiation facilities, we can estimate any estimand which is a function of already estimated estimands. By default, TMLE.jl will use [Zygote](https://fluxml.ai/Zygote.jl/latest/) but since we are using [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) you can change the backend to your favorite AD system.

For instance, by definition of the ``IATE``, we should be able to retrieve:

```math
IATE_{T_1=0 \rightarrow 1, T_2=0 \rightarrow 1} = ATE_{T_1=0 \rightarrow 1, T_2=0 \rightarrow 1} - ATE_{T_1=0, T_2=0 \rightarrow 1} - ATE_{T_1=0 \rightarrow 1, T_2=0}
```

```@example estimation
first_ate = ATE(
    outcome=:Y, 
    treatment_values=(
        T₁=(case=true, control=false), 
        T₂=(case=false, control=false)),
    treatment_confounders=(
        T₁=[:W₁₁, :W₁₂], 
        T₂=[:W₂₁, :W₂₂],
    ),
)
first_ate_result, cache = tmle(first_ate, dataset, cache=cache, verbosity=0);

second_ate = ATE(
    outcome=:Y, 
    treatment_values=(
        T₁=(case=false, control=false), 
        T₂=(case=true, control=false)),
    treatment_confounders=(
        T₁=[:W₁₁, :W₁₂], 
        T₂=[:W₂₁, :W₂₂],
    ),
    )
second_ate_result, cache = tmle(second_ate, dataset, cache=cache, verbosity=0);

composed_iate_result = compose(
    (x, y, z) -> x - y - z, 
    result₃, first_ate_result, second_ate_result
)
isapprox(
    estimate(result₄),
    estimate(composed_iate_result),
    atol=0.1
)
```
