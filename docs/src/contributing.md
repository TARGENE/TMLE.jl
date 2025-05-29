# Contributing

## Reporting

If you have a question that is not answered by the documentation, would like a new feature or find a bug, please open an [issue on Github](https://github.com/TARGENE/TMLE.jl/issues). Be specific, in the latter case, a minimal example reproducing the bug is ideal.

## Contributing

The package is not fully mature and some of the API is subject to change. If you would like to contribute a new estimand, estimator, extension, or example for the documentation, feel free to get in [touch](https://github.com/TARGENE/TMLE.jl/issues). Breaking changes are welcome if they extend the capabilities of the package.

Here is some general information about the code base.

### New Estimands

Implementing a new estimand requires both the definition of the estimand (see `src/counterfactual_mean_based/estimands.jl` for examples) and the associated estimators.

The entry point estimators are defined within `src/counterfactual_mean_based/estimators.jl`. This is because there has been a focus on implementationsestimands based on the counterfactual mean ``E[Y(T)]``. However the estimation procedures are still quite general and could serve as a backbone for future estimands. For instance:

- `get_relevant_factors`: returns the nuisance parameters for a given parameter ``Ψ``.
- `gradient_and_estimate`: computes the gradient for a given parameter ``Ψ``.

### New Estimators

There are several interesting directions to complement this package with new estimators.

- C-TMLE: collaborative TMLE is a powerful estimation strategy for which a rough template is in place (`src/counterfactual_mean_based/collaborative_template.jl`). Many strategies following the template can then be implemented. Get in touch if needed for general directions.
- Riesz Representer Estimation: Estimation of the nuisance functions is typically made by estimation of the propensity score. In cases of positivity violations this can lead to numerical instability. It turns out that the inverse of the propensity score can be estimated directly (see [here](https://github.com/TARGENE/TMLE.jl/issues/83)).

### General Code Pattern

The package is centered around the following statistical concepts which are embodied by [Julia structs](https://docs.julialang.org/en/v1/manual/types/):

- Estimand: A quantity of interest, one-dimensional like the Average Treatment Effect (`ATE`) or infinite dimensional like a conditional distribution (`ConditionalDistribution`).
- Estimator: A method that uses data to obtain an estimate of the estimand. `Tmle` and `Ose` are valid estimators for the `ATE` and the `MLConditionalDistributionEstimator` is a valid estimator for a `ConditionalDistribution`.
- Estimate: Calling an estimator on an estimand with a dataset yields an estimate. For example a `TMLEstimate` is obtained by using a `Tmle` for the `ATE`. A `MLConditionalDistribution` is obtained from `MLConditionalDistributionEstimator` for a `ConditionalDistribution`.

The general pattern is thus:

```julia
estimand = Estimand(kwargs...)
estimator = Estimator(kwargs...)
estimate = estimator(estimand, dataset; kwargs...)
```
