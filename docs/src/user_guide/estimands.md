```@meta
CurrentModule = TMLE
```

# Estimands

Most causal questions can be translated into a causal estimand. Usually either an interventional or counterfactual quantity. What would have been the outcome if I had set this variable to this value? When identified, this causal estimand translates to a statistical estimand which can be estimated from data. For us, an estimand will be a functional, that is a function that takes as input a probability distribution, and outputs a real number or vector of real numbers.

Mathematically speaking, denoting the estimand by ``\Psi``, the set of all probability distributions by ``\mathcal{M}``:

```math
\Psi: \mathcal{M} \rightarrow \mathbb{R}^p
```

At the moment, most of the work in this package has been focused on estimands that are composite functions of the interventional conditional mean which is easily identified via backdoor adjustment and for which the efficient influence function is well known.

In what follows, ``P`` is a probability distribution generating an outcome ``Y``, a random vector of "treatment" variables ``\textbf{T}`` and a random vector of "confounding" variables ``\textbf{W}``.

## The Interventional Counterfactual Mean (CM)

- Causal Question:

What would be the mean of ``Y`` in the population if we intervened on ``\textbf{T}`` and set it to ``\textbf{t}``?

- Causal Estimand:

```math
CM_{\textbf{t}}(P) = \mathbb{E}[Y|do(\textbf{T}=\textbf{t})]
```

- Statistical Estimand (via backdoor adjustment):

```math
CM_{\textbf{t}}(P) = \mathbb{E}_{\textbf{W}}[\mathbb{E}[Y|do(\textbf{T}=\textbf{t}), \textbf{W}]]
```

- TMLE.jl Example

For a Structural Causal Model `scm`, an outcome `Y` and two treatments `T₁` and `T₂` taking values `t₁` and `t₂` respectively:

```julia
Ψ = CM(scm, outcome=:Y, treatment=(T₁=t₁, T₂=t₂))
```

## The Average Treatment Effect

- Causal Question:

What is the average difference in treatment effect on ``Y`` when the two treatment levels are set to ``\textbf{t}_1`` and ``\textbf{t}_2`` respectively?

- Causal Estimand:

```math
ATE_{\textbf{t}_1 \rightarrow \textbf{t}_2}(P) = \mathbb{E}[Y|do(\textbf{T}=\textbf{t}_2)] - \mathbb{E}[Y|do(\textbf{T}=\textbf{t}_1)]
```

- Statistical Estimand (via backdoor adjustment):

```math
\begin{aligned}
ATE_{\textbf{t}_1 \rightarrow \textbf{t}_2}(P) &= CM_{\textbf{t}_2}(P) - CM_{\textbf{t}_1}(P) \\
&= \mathbb{E}_{\textbf{W}}[\mathbb{E}[Y|\textbf{T}=\textbf{t}_2, \textbf{W}] - \mathbb{E}[Y|\textbf{T}=\textbf{t}_1, \textbf{W}]]
\end{aligned}
```

- TMLE.jl Example

For a Structural Causal Model `scm`, an outcome `Y` and two treatments differences `T₁`:`t₁₁ → t₁₂` and `T₂`:`t₂₁ → t₂₂`:

```julia
Ψ = ATE(scm, outcome=:Y, treatment=(T₁=(case=t₁₂, control=t₁₁), T₂=(case=t₂₂, control=t₂₁)))
```

Note that all treatments are not required to change, for instance the following where `T₁` is held fixed at `t₁₁` is also a valid `ATE`:

```julia
Ψ = ATE(scm, outcome=:Y, treatment=(T₁=(case=t₁₁, control=t₁₁), T₂=(case=t₂₂, control=t₂₁)))
```

## The Interaction Average Treatment Effect

- Causal Question:

Interactions can be defined up to any order but we restrict the interpretation to two variables. Is the Total Average Treatment Effect of ``T₁`` and ``T₂`` different from the sum of their respective marginal Average Treatment Effects? Is there a synergistic effect between ``T₁`` and ``T₂`` on ``Y``.

For a general higher-order definition, please refer to [Higher-order interactions in statistical physics and machine learning: A model-independent solution to the inverse problem at equilibrium](https://arxiv.org/abs/2006.06010).

- Causal Estimand:

For two points interaction with both treatment and control levels ``0`` and ``1`` for ease of notation:

```math
IATE_{0 \rightarrow 1, 0 \rightarrow 1}(P) = \mathbb{E}[Y|do(T_1=1, T_2=1)] - \mathbb{E}[Y|do(T_1=1, T_2=0)]  \\
- \mathbb{E}[Y|do(T_1=0, T_2=1)] + \mathbb{E}[Y|do(T_1=0, T_2=0)] 
```

- Statistical Estimand (via backdoor adjustment):

```math
IATE_{0 \rightarrow 1, 0 \rightarrow 1}(P) = \mathbb{E}_{\textbf{W}}[\mathbb{E}[Y|T_1=1, T_2=1, \textbf{W}]] - \mathbb{E}[Y|T_1=1, T_2=0, \textbf{W}]  \\
- \mathbb{E}[Y|T_1=0, T_2=1, \textbf{W}] + \mathbb{E}[Y|T_1=0, T_2=0, \textbf{W}]] 
```

- TMLE.jl Example

For a Structural Causal Model `scm`, an outcome `Y` and two treatments differences `T₁`:`t₁₁ → t₁₂` and `T₂`:`t₂₁ → t₂₂`:

```julia
Ψ = IATE(scm, outcome=:Y, treatment=(T₁=(case=t₁₂, control=t₁₁), T₂=(case=t₂₂, control=t₂₁)))
```

## Any function of the previous Estimands

As a result of Julia's automatic differentiation facilities, given a set of already estimated estimands ``(\Psi_1, ..., \Psi_k)``, we can automatically compute an estimator for $f(\Psi_1, ..., \Psi_k)$. This is done via the `compose` function:

```julia
compose(f, args...)
```

where args are asymptotically linear estimates (see [Composing Estimands](@ref)).
