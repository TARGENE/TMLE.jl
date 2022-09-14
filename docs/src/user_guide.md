# User Guide

```@meta
CurrentModule = TMLE
```

## The Dataset

TMLE.jl should be compatible with any dataset respecting the [Tables.jl](https://tables.juliadata.org/stable/) interface, that is, a structure like a `NamedTuple` or a `DataFrame` from [DataFrames.jl](https://dataframes.juliadata.org/stable/) should work. In the remainder of this section, we will be working with the same dataset and see that we can ask very many questions (Parameters) from it.

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
    W₁ = rand(rng, Uniform(), n)
    W₂ = rand(rng, Uniform(), n)
    # Covariates
    C₁ = rand(rng, Uniform(), n)
    # Treatment | Confounders
    T₁ = rand(rng, Uniform(), n) .< logistic.(0.5sin.(W₁) .- 1.5W₂)
    T₂ = rand(rng, Uniform(), n) .< logistic.(-3W₁ - 1.5W₂)
    # Target | Confounders, Covariates, Treatments
    Y = 1 .+ 2W₁ .+ 3W₂ .- 4C₁.*T₁ .- 2T₂.*T₁.*W₂ .+ rand(rng, Normal(0, 0.1), n)
    return DataFrame(
        W₁ = W₁, 
        W₂ = W₂,
        C₁ = C₁,
        T₁ = categorical(T₁),
        T₂ = categorical(T₂),
        Y  = Y
        )
end
dataset = make_dataset()
nothing # hide
```

!!! note "Note on Treatment variables"
    It should be noted that the treatment variables **must** be [categorical](https://categoricalarrays.juliadata.org/stable/). Since the treatment is also used as an input to the ``Q_0`` learner, a `OneHotEncoder` is used by default (see [The Nuisance Parameters](@ref) section).

## The Nuisance Parameters

As described in the [Mathematical setting](@ref) section, we need to provide an estimation strategy for both $Q_0$ and $G_0$. For illustration purposes, we here consider a simple strategy where both models are assumed to be generalized linear models. However this is not the recommended practice since there is little chance those functions are actually linear, and theoretical guarantees associated with TMLE may fail to hold. We recommend instead the use of Super Learning which is exemplified in [The benefits of Super Learning](@ref).

```@example user-guide
using MLJLinearModels

η_spec = NuisanceSpec(
    LinearRegressor(), # Q model
    LogisticClassifier(lambda=0) # G model
)
nothing # hide
```

!!! note "Practical note on $Q_0$ and $G_0$"
    The models chosen for the nuisance parameters should be adapted to the outcome they target:
    - ``Q_0 = \mathbf{E}_0[Y|T=t, W=w, C=c]``, $Y$ can be either continuous or categorical, in our example it is continuous and a `LinearRegressor` is a correct choice.
    - ``G_0 = P_0(T|W)``, $T$ are always categorical variables. If T is a single treatment with only 2 levels, a logistic regression will work, if T has more than two levels a multinomial regression for instance would be suitable. If there are more than 2 treatment variables (with potentially more than 2 levels), then, the joint distribution is learnt and a multinomial regression would also work. In any case, the `LogisticClassifier` from [MLJLinearModels](https://juliaai.github.io/MLJLinearModels.jl/stable/) is a suitable choice.
    For more information on available models and their uses, we refer to the [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) documentation

The `NuisanceSpec` struct also holds a specification for the `OneHotEncoder` necessary for the encoding of treatment variables and a generalized linear model for the fluctuation model. Unless you know what you are doing, there is little chance you need to modify those.

## Parameters

### The Conditional mean

We are now ready to move to the definition of the parameters of interest. The most basic type of parameter is the conditional mean of the target given the treatment:

```math
CM_t(P) = \mathbb{E}[\mathbb{E}[Y|T=t, W]]
```

The treatment does not have to be restricted to a single variable, we can define for instance $CM_{T_1=1, T_2=1}$:

```@example user-guide
Ψ = CM(
    target      = :Y,
    treatment   = (T₁=1, T₂=1),
    confounders = [:W₁, :W₂]
)
nothing # hide
```

In this case, we can compute the exact value of the parameter:

```math
CM_{T_1=1, T_2=1} = 1 + 2\mathbb{E}[W₁] + 3\mathbb{E}[W₂] - 4\mathbb{E}[C₁] - 2\mathbb{E}[W₂] = 0.5
```

Running a targeted estimation procedure should then be as simple as:

```@example user-guide
cm_result₁₁, _, _ = tmle(Ψ, η_spec, dataset, verbosity=0)
nothing # hide
```

For now, let's ignore the two `_` outputs and focus on the `result` of type `PointTMLE`, it represents a point estimator of $CM_{T_1=1, T_2=0}$. As such, we can have a look at the value and variance of the estimator, since the estimator is asymptotically normal, a 95% confidence interval can be rougly constructed via:

```@example user-guide
Ψ̂ = TMLE.estimate(cm_result₁₁)
σ² = var(cm_result₁₁)
Ψ̂ - 1.96√σ² <= 0.5 <= Ψ̂ + 1.96√σ²
```

In fact, we can easily be more rigorous here and perform a standard T test:

```@example user-guide
OneSampleTTest(cm_result₁₁)
```

### The Average Treatment Effect

Let's now turn our attention to the Average Treatment Effect:

```math
ATE_{t_1 \rightarrow t_2}(P) = \mathbb{E}[\mathbb{E}[Y|T=t_2, W]] - \mathbb{E}[\mathbb{E}[Y|T=t_1, W]]
```

Again, from our dataset, there are many ATEs we may be interested in, let's assume we are interested in ``ATE_{T_1=0 \rightarrow 1, T_2=0 \rightarrow 1}``. Since we know the generating process, this can be computed exactly and we have:

```math
\begin{aligned}
ATE_{T_1=0 \rightarrow 1, T_2=0 \rightarrow 1} &= (1 + 2\mathbb{E}[W₁] + 3\mathbb{E}[W₂] - 4\mathbb{E}[C₁] - 2\mathbb{E}[W₂]) - (1 + 2\mathbb{E}[W₁] + 3\mathbb{E}[W₂]) \\
                                               &= -3
\end{aligned}                                    
```

Let's see what the TMLE tells us:

```@example user-guide
Ψ = ATE(
    target      = :Y,
    treatment   = (T₁=(case=1, control=0), T₂=(case=1, control=0)),
    confounders = [:W₁, :W₂]
)

ate_result, _, _ = tmle(Ψ, η_spec, dataset, verbosity=0)

OneSampleTTest(ate_result)
```

As expected.

### The Interaction Average Treatment Effect

Finally let us look at the most interesting case of interactions, we compute here the ``IATE_{T_1=0 \rightarrow 1, T_2=0 \rightarrow 1}`` since this is the highest order in our data generating process. That being said, you could go after any higher-order (3, 4, ...) interaction if you wanted. However you will inevitably decrease power and encounter positivity violations as you climb the interaction-order ladder.

Let us provide the ground truth for this pairwise interaction, you can check that:

```math
\begin{aligned}
IATE_{T_1=0 \rightarrow 1, T_2=0 \rightarrow 1} &= 1 .+ 2W₁ .+ 3W₂ .- 4C₁.*T₁ .- 2T₂.*T₁.*W₂ \\
                                                &= - 1
\end{aligned}
```

and run:

```@example user-guide
Ψ = IATE(
    target      = :Y,
    treatment   = (T₁=(case=1, control=0), T₂=(case=1, control=0)),
    confounders = [:W₁, :W₂]
)

iate_result, _, _ = tmle(Ψ, η_spec, dataset, verbosity=0)

OneSampleTTest(iate_result)
```

### Composing Parameters

By leveraging the multivariate Central Limit Theorem and Julia's automatic differentiation facilities, we can actually compute any new parameter estimate from a set of already estimated parameters. By default, TMLE.jl will use [Zygote](https://fluxml.ai/Zygote.jl/latest/) but since we are using [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) you can change the backend to your favorite AD system.

For instance, by definition of the ATE, we should be able to retrieve ``ATE_{T_1=0 \rightarrow 1, T_2=0 \rightarrow 1}`` by composing ``CM_{T_1=1, T_2=1} - CM_{T_1=0, T_2=0}``. We already have almost all of the pieces, we just need an estimate for ``CM_{T_1=0, T_2=0}``, let's get it.

```@example user-guide
Ψ = CM(
    target      = :Y,
    treatment   = (T₁=0, T₂=0),
    confounders = [:W₁, :W₂]
)
cm_result₀₀, _, _ = tmle(Ψ, η_spec, dataset, verbosity=0)
nothing # hide
```

```@example user-guide
composed_ate_result = compose(-, cm_result₁₁, cm_result₀₀)
nothing # hide
```

We can compare the estimate value, which is simply obtained by applying the function to the arguments:

```@example user-guide
TMLE.estimate(composed_ate_result), TMLE.estimate(ate_result)
```

and the variance:

```@example user-guide
var(composed_ate_result), var(ate_result)
```

## Using the cache

Oftentimes, we are interested in multiple parameters, or would like to investigate how our estimator is affected by changes in the nuisance parameters specification. In many cases, as long as the dataset under study is the same, it is possible to save some computational time by caching the previously learnt nuisance parameters. We describe below how TMLE.jl proposes to do that in some common scenarios. For that purpose let us add a new target variable (which is simply random noise) to our dataset:

```@example user-guide
dataset.Ynew = rand(1000)
nothing # hide
```

### Scenario 1: Changing the treatment values

Let us say we are interested in two ATE parameters: ``ATE_{T_1=0 \rightarrow 1, T_2=0 \rightarrow 1}`` and ``ATE_{T_1=1 \rightarrow 0, T_2=0 \rightarrow 1}`` (Notice how the setting for $T_1$ has changed).

Let us start afresh an compute the first ATE:

```@example user-guide
Ψ = ATE(
    target      = :Y,
    treatment   = (T₁=(case=1, control=0), T₂=(case=1, control=0)),
    confounders = [:W₁, :W₂]
)

ate_result₁, _, cache = tmle(Ψ, η_spec, dataset)
nothing # hide
```

Notice the logs are informing you of all the nuisance parameters that are being fitted.

Let us now investigate the second ATE by using the cache:

```@example user-guide
Ψ = ATE(
    target      = :Y,
    treatment   = (T₁=(case=0, control=1), T₂=(case=1, control=0)),
    confounders = [:W₁, :W₂]
)

ate_result₂, _, cache = tmle!(cache, Ψ)
nothing # hide
```

You should see that the logs are actually now telling you which nuisance parameters have been reused, i.e. all of them, only the targeting step needs to be done! This is because we already had nuisance estimators that matched our target parameter.

### Scenario 2: Changing the target

Let us now imagine that we are interested in another target: $Ynew$, we can say so by defining a new parameter and running the TMLE procedure using the cache:

```@example user-guide
Ψ = ATE(
    target      = :Ynew,
    treatment   = (T₁=(case=1, control=0), T₂=(case=1, control=0)),
    confounders = [:W₁, :W₂]
)

ate_result₃, _, cache = tmle!(cache, Ψ)
nothing # hide
```

As you can see, only $Q$ has been updated because the existing cached $G$ already matches our target parameter and cane be reused.

### Scenario 3: Changing the nuisance parameters specification

Another common situation is to try a new model for a given nuisance parameter (or both). Here we can try a new regularization parameter for our logistic regression:

```@example user-guide
η_spec = NuisanceSpec(
    LinearRegressor(), # Q model
    LogisticClassifier(lambda=0.001) # Updated G model
)

ate_result₄, _, cache = tmle!(cache, η_spec)
nothing # hide
```

Since we have only updated $G$'s specification, only this model is fitted again.

### Scenario N

Feel free to play around with the cache and to report any non consistent behaviour.
