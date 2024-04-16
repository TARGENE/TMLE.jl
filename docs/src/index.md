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

### 1. A dataset: here a simulation dataset

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

### 2. A quantity of interest: here the Average Treatment Effect (ATE)

The Average Treatment Effect of ``T`` on ``Y`` confounded by ``W`` is defined as:

```@example quick-start
Ψ = ATE(
    outcome=:Y, 
    treatment_values=(T=(case=true, control = false),), 
    treatment_confounders=(T=[:W],)
)
```

### 3. An estimator: here a Targeted Maximum Likelihood Estimator (TMLE)

```@example quick-start
tmle = TMLEE()
result, _ = tmle(Ψ, dataset, verbosity=0);
result
```

We are comforted to see that our estimator covers the ground truth! 🥳

```@example quick-start
using Test # hide
@test pvalue(OneSampleTTest(result, 2.5)) > 0.05 # hide
nothing # hide
```

## Scope and Distinguishing Features

The goal of this package is to provide an entry point for semi-parametric asymptotic unbiased and efficient estimation in Julia. The two main general estimators that are known to achieve these properties are the One-Step estimator and the Targeted Maximum-Likelihood estimator. Most of the current effort as been centered around estimands that are composite of the counterfactual mean.

Distinguishing Features:

- Estimands: Counterfactual Mean, Average Treatment Effect, Interactions, Any composition thereof
- Estimators: TMLE, One-Step
- Machine-Learning: Any [MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/) compatible model
- Treatments Variables:
  - Multiple treatment variables (with their own set of confounders)
  - Categorical treatment variables (factorial analysis)

## Citing TMLE.jl

If you use TMLE.jl for your own work and would like to cite us, here are the BibTeX and APA formats:

- BibTeX

```bibtex
@software{Labayle_TMLE_jl,
    author = {Labayle, Olivier and Beentjes, Sjoerd and Khamseh, Ava and Ponting, Chris},
    title = {{TMLE.jl}},
    url = {https://github.com/olivierlabayle/TMLE.jl}
}
```

- APA

Labayle, O., Beentjes, S., Khamseh, A., & Ponting, C. TMLE.jl [Computer software]. https://github.com/olivierlabayle/TMLE.jl