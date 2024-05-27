```@meta
CurrentModule = TMLE
```

# Estimands

Most causal questions can be translated into a causal estimand. Usually either an interventional or counterfactual quantity. What would have been the outcome if I had set this variable to this value? When identified, this causal estimand translates to a statistical estimand which can be estimated from data. From a mathematical standpoint, an estimand (``\Psi``) is a functional, that is a function that takes as input a probability distribution (from a model ``\mathcal{M}``), and outputs a real number or vector of real numbers.

```math
\Psi: \mathcal{M} \rightarrow \mathbb{R}^p
```

At the moment, most of the work in this package has been focused on estimands that are composite functions of the counterfactual mean which is easily identified via backdoor adjustment and for which the gradient is well known.

In what follows, ``P`` is a probability distribution generating an outcome ``Y``, a random vector of "treatment" variables ``\textbf{T}`` and a random vector of "confounding" variables ``\textbf{W}``. For the examples, we will assume two treatment variables ``T₁`` and ``T₂`` taking either values 0 or 1. The ``SCM`` is given by:

```@example estimands
using TMLE
scm = StaticSCM(
    outcomes=[:Y], 
    treatments=[:T₁, :T₂], 
    confounders=[:W]
)
```

## The Counterfactual Mean (CM)

- Causal Question:

What would have been the mean of ``Y`` had we set ``\textbf{T}=\textbf{t}``?

- Causal Estimand:

```math
CM_{\textbf{t}}(P) = \mathbb{E}[Y|do(\textbf{T}=\textbf{t})]
```

- Statistical Estimand (via backdoor adjustment):

```math
CM_{\textbf{t}}(P) = \mathbb{E}_{\textbf{W}}[\mathbb{E}[Y|\textbf{T}=\textbf{t}, \textbf{W}]]
```

- TMLE.jl Example

A causal estimand is given by:

```@example estimands
causalΨ = CM(outcome=:Y, treatment_values=(T₁=1, T₂=0))
```

A corresponding statistical estimand can be identified via backdoor adjustment using the `scm`:

```@example estimands
statisticalΨ = identify(causalΨ, scm)
```

or defined directly:

```@example estimands
statisticalΨ = CM(
    outcome=:Y, 
    treatment_values=(T₁=1, T₂=0),
    treatment_confounders=(T₁=[:W], T₂=[:W])
)
```

## The Average Treatment Effect

- Causal Question:

What would be the average difference on ``Y`` if we switch the treatment levels from ``\textbf{t}_1`` to ``\textbf{t}_2``?

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

A causal estimand is given by:

```@example estimands
causalΨ = ATE(
    outcome=:Y, 
    treatment_values=(
        T₁=(case=1, control=0), 
        T₂=(case=1, control=0)
    )
)
```

A corresponding statistical estimand can be identified via backdoor adjustment using the `scm`:

```@example estimands
statisticalΨ = identify(causalΨ, scm)
```

or defined directly:

```@example estimands
statisticalΨ = ATE(
    outcome=:Y, 
    treatment_values=(
        T₁=(case=1, control=0), 
        T₂=(case=1, control=0)
    ),
    treatment_confounders=(T₁=[:W], T₂=[:W])
)
```

## The Interaction Average Treatment Effect

- Causal Question:

Interactions can be defined up to any order but we restrict the interpretation to two variables. Is the Total Average Treatment Effect of ``T₁`` and ``T₂`` different from the sum of their respective marginal Average Treatment Effects? Is there a synergistic additive effect between ``T₁`` and ``T₂`` on ``Y``.

For a general higher-order definition, please refer to [Higher-order interactions in statistical physics and machine learning: A model-independent solution to the inverse problem at equilibrium](https://arxiv.org/abs/2006.06010).

- Causal Estimand:

For two points interaction with both treatment and control levels ``0`` and ``1`` for ease of notation:

```math
IATE_{0 \rightarrow 1, 0 \rightarrow 1}(P) = \mathbb{E}[Y|do(T_1=1, T_2=1)] - \mathbb{E}[Y|do(T_1=1, T_2=0)]  \\
- \mathbb{E}[Y|do(T_1=0, T_2=1)] + \mathbb{E}[Y|do(T_1=0, T_2=0)] 
```

- Statistical Estimand (via backdoor adjustment):

```math
IATE_{0 \rightarrow 1, 0 \rightarrow 1}(P) = \mathbb{E}_{\textbf{W}}[\mathbb{E}[Y|T_1=1, T_2=1, \textbf{W}] - \mathbb{E}[Y|T_1=1, T_2=0, \textbf{W}]  \\
- \mathbb{E}[Y|T_1=0, T_2=1, \textbf{W}] + \mathbb{E}[Y|T_1=0, T_2=0, \textbf{W}]] 
```

- TMLE.jl Example

A causal estimand is given by:

```@example estimands
causalΨ = IATE(
    outcome=:Y, 
    treatment_values=(
        T₁=(case=1, control=0), 
        T₂=(case=1, control=0)
    )
)
```

A corresponding statistical estimand can be identified via backdoor adjustment using the `scm`:

```@example estimands
statisticalΨ = identify(causalΨ, scm)
```

or defined directly:

```@example estimands
statisticalΨ = IATE(
    outcome=:Y, 
    treatment_values=(
        T₁=(case=1, control=0), 
        T₂=(case=1, control=0)
    ),
    treatment_confounders=(T₁=[:W], T₂=[:W])
)
```

- Factorial Treatments

It is possible to generate a `JointEstimand` containing all linearly independent IATEs from a set of treatment values or from a dataset. For that purpose, use the `factorialEstimand` function.

## Joint And Composed Estimands

A `JointEstimand` is simply a list of one dimensional estimands that are grouped together. For instance for a treatment `T` taking three possible values ``(0, 1, 2)`` we can define the two following Average Treatment Effects and a corresponding `JointEstimand`:

```julia
ATE₁ = ATE(
    outcome = :Y, 
    treatment_values = (T = (control = 0, case = 1),),
    treatment_confounders = [:W]
    )
ATE₂ = ATE(
    outcome = :Y, 
    treatment_values = (T = (control = 1, case = 2),),
    treatment_confounders = [:W]
    )
joint_estimand = JointEstimand(ATE₁, ATE₂)
```

You can easily generate joint estimands corresponding to Counterfactual Means, Average Treatment Effects or Average Interaction Effects by using the `factorialEstimand` function.

To estimate a joint estimand you can use any of the estimators defined in this package exactly as you would do it for a one dimensional estimand, see [Estimation](@ref).

There are two main use cases for them that we now describe.

### Joint Testing

In some cases, like in factorial analyses where multiple versions of a treatment are tested, it may be of interest to know if any version of the versions has had an effect. This can be done via a Hotelling's T2 Test, which is simply a multivariate generalisation of the Student's T test. This is the default returned by the `significance_test` function provided in TMLE.jl and the result of the test is also printed to the REPL for any joint estimate.

### Composition

Once you have estimated a `JointEstimand` and have a `JointEstimate`, you may be interested to ask further questions. For instance whether two treatment versions have the same effect. This question is typically answered by testing if the difference in Average Treatment Effect is 0. Using the Delta Method and Julia's automatic differentiation, you don't need to explicitly define a semi-parametric estimator for it. You can simply call `compose`:

```julia
ATEdiff = compose(-, joint_estimate)
```
