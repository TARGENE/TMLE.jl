```@meta
CurrentModule = TMLE
```

# Structural Causal Models

Even if you don't have to, it can be useful to define a Structural Causal Model (`SCM`) for your problem. A `SCM` is a directed acyclic graph that describes the causal relationships between the random variables under study.

## Incremental Construction

All models are wrong? Well maybe not the following:

```@example scm
using TMLE # hide
scm = SCM()
```

This model does not say anything about the random variables and is thus not really useful. Let's assume that we are interested in an outcome ``Y`` and that this outcome is determined by 8 other random variables. We can add this assumption to the model

```@example scm
add_equation!(scm, :Y => [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁, :W₂₂, :W, :C])
```

Let's now assume that we have a more complete knowledge of the problem and we also know how `T₁` and `T₂` depend on the rest of the variables in the system.

```@example scm
add_equations!(scm, :T₁ => [:W₁₁, :W₁₂, :W], :T₂ => [:W₂₁, :W₂₂, :W])
```

## One Step Construction

Instead of constructing the `SCM` incrementally, one can provide all the specified equations at once:

```@example scm
scm = SCM([
    :Y  => [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁, :W₂₂, :W, :C],
    :T₁ => [:W₁₁, :W₁₂, :W],
    :T₂ => [:W₂₁, :W₂₂, :W]
])
```

## Classic Structural Causal Models

There are many cases where we are interested in estimating the causal effect of a some treatment variables on a some outcome variables. If all treatment variables share the same set of confounders, we can quickly define the associated `SCM` with the `StaticSCM` interface:

```@example scm
scm = StaticSCM(
    outcomes=[:Y₁, :Y₂], 
    treatments=[:T₁, :T₂], 
    confounders=[:W₁, :W₂];
)
```

where `outcome_extra_covariates` is a set of extra variables that are causal of the outcomes but are not of direct interest in the study.
