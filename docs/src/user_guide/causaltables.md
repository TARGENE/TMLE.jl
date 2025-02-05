```@meta
CurrentModule = TMLE
```

# CausalTables.jl Integration

The [CausalTables.jl](https://salbalkus.github.io/CausalTables.jl/dev/) package provides a simple Tables-compliant interface allowing users to wrap data and structural causal information together into one object. It also allows users to easily simulate data from a *known* structural causal model for experimentation purposes.

TMLE.jl estimators can take `CausalTable` objects as input, in which case the user does not need to identify a statistical estimand from a causal one -- it will be identified automatically from the `CausalTable`. 

## Example

Using CausalTables.jl, one can define a structural causal model where the distribution of each variable is known, and sample from it using the `rand` function. This yields a `CausalTable` object that stores the underlying causal structure (the same information as that contained in an `SCM` object in TMLE.jl). 

Estimating a causal quantity in this scenario is now simpler: one does not need to use the `identify` function or define the variables needed in the statistical estimand; just call the estimator with the `CausalTable` object!


```@example scm
using CausalTables
using Distributions
using TMLE
# Sample a random dataset endowed with causal structure
# using the CausalTables.jl package
scm = StructuralCausalModel(@dgp(
        W ~ Beta(2,2),
        A ~ Binomial.(1, W),
        Y ~ Normal.(A .+ W, 0.5)
    ); treatment = :A, response = :Y)

ct = rand(scm, 100)

# Define a causal estimand and estimate it using TMLE
Ψ = ATE(outcome = :Y, treatment_values = (A = (case = 1, control = 0),))
estimator = TMLEE()
result, cache = estimator(Ψ, ct)
```

