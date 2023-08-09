```@meta
CurrentModule = TMLE
```

# Home

## Overview

TMLE.jl is a Julia implementation of the Targeted Minimum Loss-Based Estimation ([TMLE](https://link.springer.com/book/10.1007/978-1-4419-9782-1)) framework. If you are interested in efficient and unbiased estimation of causal effects, you are in the right place. Since TMLE uses machine-learning methods to estimate nuisance estimands, the present package is based upon [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/).

## Installation

TMLE.jl can be installed via the Package Manager and supports Julia `v1.6` and greater.

```Pkg
Pkg> add TMLE
```

## Quick Start

To run an estimation procedure, we need 3 ingredients:

1. A dataset
2. A Structural Causal Model that describes the relationship between the variables.
3. An estimand of interest

For illustration, assume we know the actual data generating process is as follows:

```math
\begin{aligned}
W  &\sim \mathcal{Uniform}(0, 1) \\
T  &\sim \mathcal{Bernoulli}(logistic(1-2 \cdot W)) \\
Y  &\sim \mathcal{Normal}(1 + 3 \cdot T - T \cdot W, 0.01)
\end{aligned}
```

Because we know the data generating process, we can simulate some data accordingly:

```@example quick-start
using TMLE
using Distributions
using StableRNGs
using Random
using CategoricalArrays
using MLJLinearModels
using LogExpFunctions

rng = StableRNG(123)
n = 100
W = rand(rng, Uniform(), n)
T = rand(rng, Uniform(), n) .< logistic.(1 .- 2W)
Y = 1 .+ 3T .- T.*W .+ rand(rng, Normal(0, 0.01), n)
dataset = (Y=Y, T=categorical(T), W=W)
nothing # hide
```

### Two lines TMLE

Estimating the Average Treatment Effect can of ``T`` on ``Y`` can be as simple as:

```@example quick-start
Ψ = ATE(:Y, (T=(case=true, control = false),), :W)
result, _ = tmle!(Ψ, dataset)
result
```

### Two steps approach

Let's first define the Structural Causal Model:

```@example quick-start
scm = StaticConfoundedModel(:Y, :T, :W)
```

and second, define the Average Treatment Effect of the treatment ``T`` on the outcome ``Y``: $ATE_{T:0 \rightarrow 1}(Y)$:

```@example quick-start
Ψ = ATE(
    scm,
    outcome      = :Y,
    treatment   = (T=(case=true, control = false),),
)
nothing # hide
```

Note that in this example the ATE can be computed exactly and is given by:

```math
ATE_{0 \rightarrow 1}(P_0) = \mathbb{E}[1 + 3 - W] - \mathbb{E}[1] = 3 - \mathbb{E}[W] = 2.5
```

Running the `tmle` will produce two asymptotically linear estimators: the TMLE and the One Step Estimator. For each we can look at the associated estimate, confidence interval and p-value:

```@example quick-start
result, _ = tmle!(Ψ, dataset)
result
```

and be comforted to see that our estimators covers the ground truth! 🥳

```@example quick-start
using Test # hide
@test pvalue(OneSampleTTest(result.tmle, 2.5)) > 0.05 # hide
nothing # hide
```
