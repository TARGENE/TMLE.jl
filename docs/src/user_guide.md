# User Guide

```@meta
CurrentModule = TMLE
```

```jldoctest
a = 1
b = 2
a + b

# output

3
```

## The Dataset

TMLE.jl should be compatible with any dataset respecting the [Tables.jl](https://tables.juliadata.org/stable/) interface, that is, a structure like a `NamedTuple` or a `DataFrame` from [DataFrames.jl](https://dataframes.juliadata.org/stable/) should work. In the remainder of this section, we will be working with the same dataset and see that we can ask very many questions (Parameters) from it.

```jldoctest user-guide
using Random
using Distributions
using DataFrames
using StableRNGs
using CategoricalArrays
using TMLE

function dataset(;n=100)
    rng = StableRNG(123)
    # Confounders
    W₁ = rand(rng, Uniform(), n)
    W₂ = rand(rng, Uniform(), n)
    # Covariates
    C₁ = rand(rng, Uniform(), n)
    # Treatment | Confounders
    T₁ = rand(rng, Uniform(), n) .< TMLE.expit(0.5sin.(W₁) .- 1.5W₂)
    T₂ = rand(rng, Uniform(), n) .< TMLE.expit(-3W₁ - 1.5W₂)
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
```

!!! note "Note on Treatment variables"
    It should be noted that the treatment variables **must** be categorical.

## The Nuisance Parameters

As described in the [Mathematical setting](@ref) section, we need to provide an estimation strategy for both $Q_0$ and $G_0$. For illustration purposes, we here consider a simple strategy where both models are assumed to be generalized linear models. However this is not the recommended practice since there is little chance those functions are actually linear, and theoretical guarantees associated with TMLE may fail to hold. We recommend instead the use of Super Learning which is exemplified in [The benefits of Super Learning](@ref).

```jldoctest user-guide
using MLJLinearModels

η_spec = NuisanceSpec(
    LinearRegressor(), # Q model
    LogisticClassifier(lambda=0) # G model
)
```

!!! note "Practical note on $Q_0$ and $G_0$"
    The models chosen for the nuisance parameters should be adapted to the outcome they target:
    - ``Q_0 = \mathbf{E}_0[Y|T=t, W=w, C=c]``, $Y$ can be either continuous or categorical, in our example it is continuous and a `LinearRegressor` is a correct choice.
    - ``G_0 = P_0(T|W)``, $T$ are always categorical variables. If T is a single treatment with only 2 levels, a logistic regression will work, if T has more than two levels a multinomial regression for instance would be suitable. If there are more than 2 treatment variables (with potentially more than 2 levels), then, the joint distribution is learnt and a multinomial regression would also work. In any case, the `LogisticClassifier` from [MLJLinearModels](https://juliaai.github.io/MLJLinearModels.jl/stable/) is a suitable choice.
    For more information on available models and their uses, we refer to the [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) documentation

## Parameters

### The Conditional mean

We are now ready to move to the definition of the parameters of interest. The most basic type of parameter is the conditional mean of the target given the treatment:

```math
CM_t(P) = \mathbb{E}[\mathbb{E}[Y|T=t, W]]
```

The treatment does not have to be restricted to a single variable, we can define for instance $CM_{T_1=1, T_2=1}$:

```jldoctest user-guide
Ψ = CM(
    target      = :Y,
    treatment   = (T₁=1, T₂=1),
    confounders = [:W₁, :W₂]
)
```

In this case, we can compute the exact value of the parameter:

```math
CM_{T_1=1, T_2=1} = 1 + 2\mathbb{E}[W₁] + 3\mathbb{E}[W₂] - 4\mathbb{E}[C₁] - 2\mathbb{E}[W₂] = 0.5
```

Running a targeted estimation procedure should then be as simple as:

```jldoctest user-guide
cm_result₁₁, _, _ = tmle(Ψ, η_spec, dataset(;n=100))
```

For now, let's ignore the two `_` outputs and focus on the `result` of type `PointTMLE`, it represents a point estimator of $CM_{T_1=1, T_2=0}$. As such, we can have a look at the value and variance of the estimator, since the estimator is asymptotically normal, a 95% confidence interval can be rougly constructed via:

```jldoctest user-guide
Ψ̂ = TMLE.estimate(cm_result₁₁)
σ² = var(cm_result₁₁)
Ψ̂ - 1.96√σ² <= 0.5 <= Ψ̂ + 1.96√σ²

# output

true
```

In fact, we can easily be more rigorous here and perform a standard T test:

```jldoctest user-guide
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

```jldoctest user-guide
Ψ = ATE(
    target      = :Y,
    treatment   = (T₁=(case=1, control=0), T₂=(case=1, control=0)),
    confounders = [:W₁, :W₂]
)

ate_result, _, _ = tmle(Ψ, η_spec, dataset(;n=100))

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

```jldoctest user-guide
Ψ = IATE(
    target      = :Y,
    treatment   = (T₁=(case=1, control=0), T₂=(case=1, control=0)),
    confounders = [:W₁, :W₂]
)

iate_result, _, _ = tmle(Ψ, η_spec, dataset(;n=100))

OneSampleTTest(iate_result)
```

### Composing Parameters

By leveraging the multivariate Central Limit Theorem and Julia's automatic differentiation facilities, we can actually compute any new parameter estimate from a set of already estimated parameters. By default, TMLE.jl will use [Zygote](https://fluxml.ai/Zygote.jl/latest/) but since we are using [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) you can change the backend to your favorite AD system.

For instance, by definition of the ATE, we should be able to retrieve ``ATE_{T_1=0 \rightarrow 1, T_2=0 \rightarrow 1}`` by composing ``CM_{T_1=1, T_2=1} - CM_{T_1=0, T_2=0}``. We already have almost all of the pieces, we just need an estimate for ``CM_{T_1=0, T_2=0}``, let's get it.

```jldoctest user-guide
Ψ = CM(
    target      = :Y,
    treatment   = (T₁=0, T₂=0),
    confounders = [:W₁, :W₂]
)
cm_result₀₀, _, _ = tmle(Ψ, η_spec, dataset(;n=100))
```

```jldoctest user-guide
composed_ate_result = compose(-, cm_result₁₁, cm_result₀₀)
```

We can compare the estimate value, which is simply obtained by applying the function to the arguments:

```jldoctest user-guide
TMLE.estimate(composed_ate_result), TMLE.estimate(ate_result)
```

and the variance:

```jldoctest user-guide
var(composed_ate_result), var(ate_result)
```

## Using the cache
