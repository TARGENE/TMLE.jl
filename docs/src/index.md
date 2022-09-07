```@meta
CurrentModule = TMLE
```

# Home

## Overview

TMLE.jl is a Julia implementation of the Targeted Minimum Loss-Based Estimation ([TMLE](https://link.springer.com/book/10.1007/978-1-4419-9782-1)) framework. If you are interested in the efficient and unbiased estimation of some causal effect, you are in the right place. Since TMLE uses machine-learning methods under the hood, the present package is based upon [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/). This means that any model respecting the MLJ interface can be used for the estimation of nuisance parameters.

## Important Mathematical details

Even though TMLE is not restricted to causal inference, the scope of the present package is currently restricted to so called "causal effects" and the following causal graph is assumed throughout:

```@raw html
<img src="assets/causal_graph.png" alt="Causal Model" style="width:400px;"/>
```

whith the following semantics:

- Y: The target variable
- T: Treatment variables
- W: Confounding variables
- C: Extra covariates

This graph encodes a factorization of the joint probability distribution that generated the data we observed:

```math
P(Y, T, W, C) = P(Y|T, W, C) \times P(T|W) \times P(W) \times P(C)
```

Usually, we don't have much information about $P$ and don't want to make unrealistic assumptions, thus we will simply state that $P \in \mathcal{M}$ where $\mathcal{M}$ is the space of all probability distributions. It turns out that most target quantities of interest in causal inference can be expressed as real-valued functions of $P$, denoted by: $\Psi:\mathcal{M} \rightarrow \Re$.

TMLE.jl enables the estimation of the following quantities:

### The Conditional Mean

This is simply the expected value of the target $Y$ when the treatment is set to $t$.

```math
\Psi_t(P) = \mathbb{E}[\mathbb{E}[Y|T=t, W]]
```

### The Average Treatment Effect (ATE)

Probably the most famous quantity in causal inference. Under suitable assumptions, tt represents the additive effect of changing the treatment value from $t_1$ to $t_2$.

```math
ATE_{t_1 \rightarrow t_2}(P) = \mathbb{E}[\mathbb{E}[Y|T=t_2, W]] - \mathbb{E}[\mathbb{E}[Y|T=t_1, W]]
```

### Interaction Average Treatment Effect (IATE)

If you are interested in the interaction effect of multiple treatments, the IATE is for you. An example formula is displayed below for two interacting treatments whose values are both changed from $0$ to $1$:

```math
IATE_{0 \rightarrow 1, 0 \rightarrow 1}(P) = \mathbb{E}[\mathbb{E}[Y|T_1=1, T_2=1, W]] - \mathbb{E}[\mathbb{E}[Y|T_1=1, T_2=0, W]]  \\
- \mathbb{E}[\mathbb{E}[Y|T_1=0, T_2=1, W]] + \mathbb{E}[\mathbb{E}[Y|T_1=0, T_2=0, W]] 
```

### Any function of the previous Parameters

As a result of Julia's automatic differentiation facilities, given a set of already estimated parameters $(\Psi_1, ..., \Psi_k)$, we can automatically compute an estimator for $f(\Psi_1, ..., \Psi_k)$.

## Installation

TMLE.jl can be installed via the Package Manager and supports Julia `v1.6` and greater.

```julia
pkg> add TMLE
```

## Quick Start

To run an estimation procedure, we need 3 ingredients:

- A dataset
- A parameter of interest
- A nuisance parameters estimation specification

For instance, assume the following simple data generating process:

```julia
using Distributions
using Random
using CategoricalArrays

# Generate the dataset
n = 100
W = rand(Uniform(), n)
T = rand(Uniform(), n) .< TMLE.expit(1 .- 2W)
y = 1 .+ 3T .- T.*W .+ rand(Normal(0, 0.01), n)
dataset = (y=y, T=categorical(T), W=W)
```

And say we are interested in the $ATE_{0 \rightarrow 1}(P)$:

```julia
Ψ = ATE(
    target      = :y,
    treatment   = (T=(case=1, control = 0),)
    confounders = [:W]
)
```

Note that in this example the ATE can be computed exactly and is given by:

```math
ATE_{0 \rightarrow 1}(P) = \mathbb{E}[1 + 3 - W] - \mathbb{E}[1] = 3 - \mathbb{E}[W] = 2.5
```

We next need to define the strategy for learning the nuisance parameters $Q$ and $G$, here we keep things simple:

```julia
η_spec = NuisanceSpec(
    LinearRegressor(),
    LogisticClassifier(lambda=0)
)
```

And then run the TMLE procedure:

```julia
tmle(Ψ, η_spec, dataset);
```
