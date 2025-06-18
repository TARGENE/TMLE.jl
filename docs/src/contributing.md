# Contributing

## Reporting

If you have a question that is not answered by the documentation, would like a new feature or find a bug, please open an [issue on Github](https://github.com/TARGENE/TMLE.jl/issues). Be specific, in the latter case, a minimal example reproducing the bug is ideal.

## Contributing

If you are new to Julia and are unsure where to start, I recommend this (old but good) [Youtube video](https://www.youtube.com/watch?v=QVmU29rCjaA&t=16s).

The package is not fully mature and some of the API is subject to change. If you would like to contribute a new estimand, estimator, extension, or example for the documentation, feel free to get in [touch](https://github.com/TARGENE/TMLE.jl/issues). Breaking changes are welcome if they extend the capabilities of the package.

Here is some general information about the code base.

### Guidance for New Estimands

Implementing a new estimand requires both the definition of the estimand (see `src/counterfactual_mean_based/estimands.jl` for examples) and the associated estimators.

Currently, the entry point estimators are defined within `src/counterfactual_mean_based/estimators.jl`. This is because there has been a focus on implementating estimands based on the counterfactual mean ``E[Y(T)]``. However the estimation procedures are still quite general and could serve as a backbone for future estimands. For instance:

- `get_relevant_factors`: returns the nuisance parameters for a given parameter ``Ψ``.
- `gradient_and_estimate`: computes the gradient for a given parameter ``Ψ``.

### Guidance for New Estimators

There are several interesting directions to complement this package with new estimators.

- C-TMLE: collaborative TMLE is a powerful estimation strategy for which a rough template is in place (`src/counterfactual_mean_based/collaborative_template.jl`). Many strategies following the template can then be implemented. Get in touch if needed for general directions.
- Riesz Representer Estimation: Estimation of the nuisance functions is typically made by estimation of the propensity score. In cases of positivity violations this can lead to numerical instability. It turns out that the inverse of the propensity score can be estimated directly (see [here](https://github.com/TARGENE/TMLE.jl/issues/83)).

### Estimand - Estimator - Estimate

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

### Testing

Testing is a vital part of any software development and is expected with a particular focus on correctness. 

#### Unit Testing

Basic unit-testing should be in place for most important implemented functions but you don't need to test that ``1 + 1 = 2``.

#### Statistical Guarantees

Theoretical statistical guarantees are difficult to verify in practice but simulations can help. At the moment, the main property for which most estimators are tested against, is their so-called double robustness. That is, they should converge to the true value of the parameter if only the propensity score or the outcome model is correctly specified. At the moment there is a test file for each of the two main parameters of the package, the `ATE` and the `AIE`:

- `test/counterfactual_mean_based/double_robustness_ate.jl` with simulations in `test/counterfactual_mean_based/double_robustness_ate.jl`
- `test/counterfactual_mean_based/double_robustness_aie.jl` with simulations in `test/counterfactual_mean_based/aie_simulations.jl`

Both test the set of estimators defined by the `double_robust_estimators()` function in `test/helper_fns.jl`. To add your new estimator to the testset, you just need to add it to the return list of that function. One thing to note here is that, instead of verifying the convergence with the sample size, we only verify coverage for a sufficiently large sample size in a variety of simulations. This can be improved in the future and would represent an interesting contribution to this package.

If a new estimator you provide is expected to perform better in a well defined context, writing a suitable simulation verifying this expected behaviour is highly recommended. Note that this does not have to be a test in the `test` folder but could instead be part of the documentation and will then serve two purposes at once.

### Documentation

Please provide documentation for your contribution, this documentation is two folds:

- Developer oriented: with comments in the code for example
- User oriented: 
  - At least a docstring with a brief description and potentially a usage example.
  - Ideally a contribution to the `docs` source code.