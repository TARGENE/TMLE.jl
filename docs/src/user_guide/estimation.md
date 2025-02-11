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

The `cache` (see below) contains estimates for the nuisance functions that were necessary to estimate the ATE.

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

By default, TMLE.jl uses generalized linear models for the estimation of relevant and nuisance factors such as the outcome mean and the propensity score. However, this is not the recommended usage since the estimators' performance is closely related to how well we can estimate these factors. More sophisticated models can be provided using the `models` keyword argument of each estimator which is a `Dict{Symbol, Model}` mapping variables' names to their respective model.

Rather than specifying a specific model for each variable it may be easier to override the default models using the `default_models` function:

For example one can override all default models with XGBoost models from `MLJXGBoostInterface`:

```@example estimation
using MLJXGBoostInterface
xgboost_regressor = XGBoostRegressor()
xgboost_classifier = XGBoostClassifier()
models = default_models(
    Q_binary     = xgboost_classifier,
    Q_continuous = xgboost_regressor,
    G            = xgboost_classifier
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

models = default_models( # For all non-specified variables use the following defaults
        Q_binary     = stack_binary, # A Super Learner
        Q_continuous = xgboost_regressor, # An XGBoost
        # T₁ with XGBoost prepended with a Continuous Encoder
        T₁           = xgboost_classifier
        # Unspecified G defaults to Logistic Regression
)

tmle_custom = TMLEE(models=models)
```

Notice that `with_encoder` is simply a shorthand to construct a pipeline with a `ContinuousEncoder` and that the resulting `models` is simply a `Dict`.

## CV-Estimation

Canonical TMLE/OSE are essentially using the dataset twice, once for the estimation of the nuisance functions and once for the estimation of the parameter of interest. This means that there is a risk of over-fitting and residual bias ([see here](https://arxiv.org/abs/2203.06469) for some discussion). One way to address this limitation is to use a technique called sample-splitting / cross-validation. In order to activate the sample-splitting mode, simply provide a `MLJ.ResamplingStrategy` using the `resampling` keyword argument:

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
- Caching of Nuisance Functions: Because the `resampling` strategy typically needs to preserve the outcome and treatment proportions, very little reuse of cached models is possible (see [Using the Cache](@ref)).

## Using the Cache

TMLE and OSE are expensive procedures, it may therefore be useful to store some information for further reuse. This is the purpose of the `cache` object, which is produced as a byproduct of the estimation process. 

### Reusing Models

The cache contains in particular the machine-learning models that were fitted in the process and which can sometimes be reused to estimate other quantities of interest. For example, say we are now interested in the Joint Average Treatment Effect of both `T₁` and `T₂`. We can provide the cache to the next round of estimation as follows.

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

Only the conditional distribution of `Y` given `T₁` and `T₂` is fitted as it is absent from the cache. However, the propensity scores corresponding to `T₁` and `T₂` have been reused. Finally, let's see what happens if we estimate the interaction effect between `T₁` and `T₂` on `Y`.

```@example estimation
Ψ₄ = AIE(
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

## Accessing Fluctuations' Reports (Advanced)

The cache also holds the last targeted factor that was estimated if TMLE was used. Some key information related to the targeting steps can be accessed, for example:

```@example estimation
gradients(cache);
estimates(cache);
epsilons(cache)
```

correspond to the gradients, point estimates and epsilons obtained after each targeting step which was performed (usually only one).

One can for instance check that the mean of the gradient is close to zero.

```@example estimation
mean(last(gradients(cache)))
```

## Joint Estimands and Composition

As explained in [Joint And Composed Estimands](@ref), a joint estimand is simply a collection of estimands. Here, we will illustrate that an Average Interaction Effect is also defined as a difference in partial Average Treatment Effects.

More precisely, we would like to see if the left-hand side of this equation is equal to the right-hand side:

```math
AIE_{T_1=0 \rightarrow 1, T_2=0 \rightarrow 1} = ATE_{T_1=0 \rightarrow 1, T_2=0 \rightarrow 1} - ATE_{T_1=0, T_2=0 \rightarrow 1} - ATE_{T_1=0 \rightarrow 1, T_2=0}
```

For that, we need to define a joint estimand of three components:

```@example estimation
ATE₁ = ATE(
    outcome=:Y, 
    treatment_values=(
        T₁=(case=true, control=false), 
        T₂=(case=false, control=false)),
    treatment_confounders=(
        T₁=[:W₁₁, :W₁₂], 
        T₂=[:W₂₁, :W₂₂],
    ),
)
ATE₂ = ATE(
    outcome=:Y, 
    treatment_values=(
        T₁=(case=false, control=false), 
        T₂=(case=true, control=false)),
    treatment_confounders=(
        T₁=[:W₁₁, :W₁₂], 
        T₂=[:W₂₁, :W₂₂],
    ),
    )
joint_estimand = JointEstimand(Ψ₃, ATE₁, ATE₂)
```

where the interaction `Ψ₃` was defined earlier. This joint estimand can be estimated like any other estimand using our estimator of choice:

```@example estimation
joint_estimate, cache = tmle(joint_estimand, dataset, cache=cache, verbosity=0);
joint_estimate
```

The printed output is the result of a Hotelling's T2 Test which is the multivariate counterpart of the Student's T Test. It tells us whether any of the component of this joint estimand is different from 0.

Then we can formally test our hypothesis by leveraging the multivariate Central Limit Theorem and Julia's automatic differentiation.

```@example estimation
composed_result = compose((x, y, z) -> x - y - z, joint_estimate)
isapprox(
    estimate(result₄),
    first(estimate(composed_result)),
    atol=0.1
)
```

By default, TMLE.jl will use [Zygote](https://fluxml.ai/Zygote.jl/latest/) but since we are using [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) you can change the backend to your favorite AD system.
