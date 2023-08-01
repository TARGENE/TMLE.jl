```@meta
CurrentModule = TMLE
```

# Home

## Overview

TMLE.jl is a Julia implementation of the Targeted Minimum Loss-Based Estimation ([TMLE](https://link.springer.com/book/10.1007/978-1-4419-9782-1)) framework. If you are interested in efficient and unbiased estimation of some causal effect, you are in the right place. Since TMLE uses machine-learning methods to estimate nuisance estimands, the present package is based upon [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/). This means that any model respecting the MLJ interface can be used for the estimation of nuisance estimands.

## Mathematical setting

Even though the TMLE framework is not restricted to causal inference, the scope of the present package is currently restricted to so called "causal effects" and the following causal graph is assumed throughout:

```@raw html
<div style="text-align:center">
<img src="assets/causal_graph.png" alt="Causal Model" style="width:400px;"/>
</div>
```

whith the following general interpration of variables:

- Y: The Target
- T: The Treatment
- W: Confounders
- C: Extra Covariates

This graph encodes a factorization of the joint probability distribution $P_0$ that we assume generated the observed data:

```math
P_0(Y, T, W, C) = P_0(Y|T, W, C) \cdot P_0(T|W) \cdot P_0(W) \cdot P_0(C)
```

Usually, we don't have much information about $P_0$ and don't want to make unrealistic assumptions, thus we will simply state that $P_0 \in \mathcal{M}$ where $\mathcal{M}$ is the space of all probability distributions. It turns out that most target quantities of interest in causal inference can be expressed as real-valued functions of $P_0$, denoted by: $\Psi:\mathcal{M} \rightarrow \Re$. TMLE works by finding a suitable estimator $\hat{P}_n$ of $P_0$ and then simply substituting it into $\Psi$. Fortunately, it is often non necessary to come up with an estimator of the joint distribution $P_0$, instead only subparts of it are required. Those are called nuisance estimands because they are not of direct interest and for our purposes, only two nuisance estimands are necessary:

```math
Q_0(t, w, c) = \mathbb{E}[Y|T=t, W=w, C=c]
```

and

```math
G_0(t, w) = P(T=t|W=w)
```

Specifying which machine-learning method to use for each of those nuisance estimand is the only ingredient you will need to run a targeted estimation for any of the following quantities. Those quantities have a causal interpretation under suitable assumptions that we do not discuss here.

### The Conditional Mean

This is simply the expected value of the target $Y$ when the treatment is set to $t$.

```math
CM_t(P) = \mathbb{E}[\mathbb{E}[Y|T=t, W]]
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

### Any function of the previous Estimands

As a result of Julia's automatic differentiation facilities, given a set of already estimated estimands $(\Psi_1, ..., \Psi_k)$, we can automatically compute an estimator for $f(\Psi_1, ..., \Psi_k)$.

## Installation

TMLE.jl can be installed via the Package Manager and supports Julia `v1.6` and greater.

```Pkg
Pkg> add TMLE
```

## Quick Start

To run an estimation procedure, we need 3 ingredients:

- A dataset
- A estimand of interest
- A nuisance estimands specification

For instance, assume the following simple data generating process:

```math
\begin{aligned}
W  &\sim \mathcal{Uniform}(0, 1) \\
T  &\sim \mathcal{Bernoulli}(logistic(1-2 \cdot W)) \\
Y  &\sim \mathcal{Normal}(1 + 3 \cdot T - T \cdot W, 0.01)
\end{aligned}
```

which can be simulated in Julia by:

```@example quick-start
using Distributions
using StableRNGs
using Random
using CategoricalArrays
using MLJLinearModels
using TMLE
using LogExpFunctions

rng = StableRNG(123)
n = 100
W = rand(rng, Uniform(), n)
T = rand(rng, Uniform(), n) .< logistic.(1 .- 2W)
Y = 1 .+ 3T .- T.*W .+ rand(rng, Normal(0, 0.01), n)
dataset = (Y=Y, T=categorical(T), W=W)
nothing # hide
```

And say we are interested in the $ATE_{0 \rightarrow 1}(P_0)$:

```@example quick-start
Ψ = ATE(
    target      = :Y,
    treatment   = (T=(case=true, control = false),),
    confounders = [:W]
)
nothing # hide
```

Note that in this example the ATE can be computed exactly and is given by:

```math
ATE_{0 \rightarrow 1}(P_0) = \mathbb{E}[1 + 3 - W] - \mathbb{E}[1] = 3 - \mathbb{E}[W] = 2.5
```

We next need to define the strategy for learning the nuisance estimands $Q$ and $G$, here we keep things simple and simply use generalized linear models:

```@example quick-start
η_spec = NuisanceSpec(
    LinearRegressor(),
    LogisticClassifier(lambda=0)
)
nothing # hide
```

We are now ready to run the TMLE procedure and look the associated confidence interval:

```@example quick-start
result, _ = tmle(Ψ, η_spec, dataset)
result
```

```@example quick-start
using Test # hide
@test pvalue(OneSampleTTest(result.tmle, 2.5)) > 0.05 # hide
nothing # hide
```
