```@meta
CurrentModule = TMLE
```

# Home

## Overview

TMLE.jl is a Julia implementation of the Targeted Minimum Loss-Based Estimation ([TMLE](https://link.springer.com/book/10.1007/978-1-4419-9782-1)) framework. If you are interested in leveraging the power of modern machine-learning methods while preserving interpretability and statistical inference guarantees, you are in the right place. TMLE.jl is compatible with any [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) compliant algorithm and any dataset respecting the [Tables](https://tables.juliadata.org/stable/) interface.

The following plot illustrates the bias reduction achieved by TMLE over a mis-specified linear model in the presence of confounding. Note that in this case, TMLE also uses mis-specified models but still achieves a lower bias due to the targeting step.

```@setup intro
using GLM
using Distributions
using Random
using DataFrames
using CairoMakie
using TMLE
using CategoricalArrays
using MLJLinearModels

function generate_confounded(;α = 0, β = 1, γ = -1, n = 100)
    W = rand(Normal(1, 1), n)
    Uₜ = rand(Normal(0, 1), n)
    T = Uₜ + W .< 0.7
    ϵ = rand(Normal(0, 0.1), n)
    Y = @. β * T + α * W + γ * T * W + ϵ
    return DataFrame(W=W, T=T, Y=Y)
end

function generate_unconfounded(;α = 0, β = 1, γ = -1, n = 100)
    W = rand(Normal(1, 1), n)
    Uₜ = rand(Normal(0, 1), n)
    T = rand(Bernoulli(0.2), n)
    ϵ = rand(Normal(0, 1), n)
    Y = @. β * T + α * W + γ * T * W + ϵ
    return DataFrame(W=W, T=T, Y=Y)
end

function linear_model_coef(data)
    fitted_lm = lm(@formula(Y ~ T + W), data)
    return coef(fitted_lm)[2]
end

function tmle_estimates(data)
    models = default_models(;
        Q_continuous=MLJLinearModels.LinearRegressor(),
        Q_binary=MLJLinearModels.LogisticClassifier(),
        G=MLJLinearModels.LogisticClassifier()
    )
    Ψ̂ = TMLEE(models=models, weighted=true)
    Ψ = ATE(;
        outcome=:Y, 
        treatment_values=(T=(case=true, control = false),),
        treatment_confounders=(:W,)
    )
    data.T = categorical(data.T)
    Ψ̂ₙ, cache = Ψ̂(Ψ, data;verbosity=0);
    return TMLE.estimate(Ψ̂ₙ)
end

function bootstrap_analysis(;B=100, α=0, β=1, γ=-1, n=100, ATE₀=α+γ)
    Random.seed!(123)
    β̂s_confounded = Vector{Float64}(undef, B)
    tmles_confounded = Vector{Float64}(undef, B)
    β̂s_unconfounded = Vector{Float64}(undef, B)
    tmles_unconfounded = Vector{Float64}(undef, B)
    for b in 1:B
        data_confounded = generate_confounded(;α=α, β=β, γ=γ, n=n)
        β̂s_confounded[b] = linear_model_coef(data_confounded)
        tmles_confounded[b] = tmle_estimates(data_confounded)
        data_unconfounded = generate_unconfounded(;α=α, β=β, γ=γ, n=n)
        β̂s_unconfounded[b] = linear_model_coef(data_unconfounded)
        tmles_unconfounded[b] = tmle_estimates(data_unconfounded)
    end
    return β̂s_confounded, β̂s_unconfounded, tmles_confounded, tmles_unconfounded
end

function plot(β̂s_confounded, β̂s_unconfounded, tmles_confounded, tmles_unconfounded, β, ATE₀)
    fig = Figure(size=(1000, 800))
    ax = Axis(fig[1, 1], title="Distribution of Linear Model's and TMLE's Estimates", yticks=(1:2, ["Confounded", "Unconfounded"]))
    labels = vcat(repeat(["Confounded"], length(β̂s_confounded)), repeat(["Unconfounded"], length(β̂s_unconfounded)))
    rainclouds!(ax, labels, vcat(β̂s_confounded, β̂s_unconfounded), orientation = :horizontal, color=(:blue, 0.5))
    rainclouds!(ax, labels, vcat(tmles_confounded, tmles_unconfounded), orientation = :horizontal, color=(:orange, 0.5))
    vlines!(ax, ATE₀, label="ATE", color=:green)
    vlines!(ax, β, label="β", color=:red)

    Legend(fig[1, 2], 
        [PolyElement(color = :blue), PolyElement(color = :orange), LineElement(color = :green), LineElement(color = :red)], 
        ["Linear", "TMLE", "ATE", "β"], 
        framevisible = false,
    )
    return fig
end

Random.seed!(123)
B = 1000
n = 1000
α = 0
β = 1
γ = -1
ATE₀ = β + γ
β̂s_confounded, β̂s_unconfounded, tmles_confounded, tmles_unconfounded = bootstrap_analysis(;B=B, α=α, β=β, γ=γ, n=n, ATE₀=ATE₀)
fig = plot(β̂s_confounded, β̂s_unconfounded, tmles_confounded, tmles_unconfounded, β, ATE₀)
save(joinpath("assets", "home_simulation.png"), fig)
```

![Home Illustration](assets/home_simulation.png)

## Installation

TMLE.jl can be installed via the Package Manager and supports Julia `v1.10` and greater.

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

The goal of this package is to provide an entry point for semi-parametric asymptotic unbiased and efficient estimation in Julia. The two main general estimators that are known to achieve these properties are the One-Step estimator and the Targeted Maximum-Likelihood estimator. Most of the current effort has been centered around estimands that are composite of the counterfactual mean.

Distinguishing Features:

- Estimands: Counterfactual Mean, Average Treatment Effect, Interactions, Any composition thereof
- Estimators: TMLE, One-Step, in both canonical and cross-validated versions.
- Machine-Learning: Any [MLJ](https://alan-turing-institute.github.io/MLJ.jl/stable/) compatible model
- Dataset: Any dataset respecting the [Tables](https://tables.juliadata.org/stable/) interface (e.g. [DataFrames.jl](https://dataframes.juliadata.org/stable/))
- Factorial Treatment Variables:
  - Multiple treatments
  - Categorical treatment values

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