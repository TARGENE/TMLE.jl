"""
    tmle!(Ψ::CMCompositeEstimand, dataset; 
        adjustment_method=BackdoorAdjustment(), 
        verbosity=1, 
        force=false, 
        ps_lowerbound=1e-8, 
        weighted_fluctuation=false
        )

Performs Targeted Minimum Loss Based Estimation of the target estimand.

## Arguments

- Ψ: An estimand of interest.
- dataset: A table respecting the `Tables.jl` interface.
- adjustment_method: A confounding adjustment method.
- verbosity: Level of logging.
- force: To force refit of machines in the SCM .
- ps_lowerbound: The propensity score will be truncated to respect this lower bound.
- weighted_fluctuation: To use a weighted fluctuation instead of the vanilla TMLE, can improve stability.
"""
function tmle!(Ψ::CMCompositeEstimand, dataset;
    resampling=nothing,
    adjustment_method=BackdoorAdjustment(), 
    verbosity=1, 
    force=false, 
    ps_lowerbound=1e-8, 
    weighted_fluctuation=false
    )
    # Check the estimand against the dataset
    check_treatment_levels(Ψ, dataset)
    # Initial fit of the SCM's equations
    verbosity >= 1 && @info "Fitting the required equations..."
    initial_factors = fit!(Ψ, dataset;
        adjustment_method=adjustment_method, 
        verbosity=verbosity, 
        force=force
    )
    # Get propensity score truncation threshold
    n = nrows(initial_factors[outcome(Ψ)].machine.data[2])
    ps_lowerbound = ps_lower_bound(n, ps_lowerbound)
    # Fit Fluctuation
    verbosity >= 1 && @info "Performing TMLE..."
    targeted_factors = fluctuate(initial_factors, Ψ; 
        tol=nothing, 
        verbosity=verbosity, 
        weighted_fluctuation=weighted_fluctuation, 
        ps_lowerbound=ps_lowerbound
    )
    # Estimation results after TMLE
    IC, Ψ̂ = gradient_and_estimate(Ψ, targeted_factors; ps_lowerbound=ps_lowerbound)
    verbosity >= 1 && @info "Done."
    return TMLEstimate(Ψ̂, IC), targeted_factors
end

function ose!(Ψ::CMCompositeEstimand, dataset; 
    adjustment_method=BackdoorAdjustment(), 
    verbosity=1, 
    force=false, 
    ps_lowerbound=1e-8)
    # Check the estimand against the dataset
    check_treatment_levels(Ψ, dataset)
    # Initial fit of the SCM's equations
    verbosity >= 1 && @info "Fitting the required equations..."
    fit!(Ψ, dataset;
        adjustment_method=adjustment_method, 
        verbosity=verbosity, 
        force=force
    )
    # Get propensity score truncation threshold
    ps_lowerbound = ps_lower_bound(Ψ, ps_lowerbound)
    # Retrieve initial fit
    Q = get_outcome_model(Ψ)
    # Gradient and estimate
    IC, Ψ̂ = gradient_and_estimate(Ψ, Q; ps_lowerbound=ps_lowerbound)
    verbosity >= 1 && @info "Done."
    return OSEstimate(Ψ̂ + mean(IC), IC), Q
end

function naive_plugin_estimate!(Ψ::CMCompositeEstimand, dataset;
    adjustment_method=BackdoorAdjustment(), 
    verbosity=1, 
    force=false)
    # Check the estimand against the dataset
    check_treatment_levels(Ψ, dataset)
    # Initial fit of the SCM's equations
    verbosity >= 1 && @info "Fitting the required equations..."
    Q = get_or_set_conditional_distribution_from_natural!(
        Ψ.scm, outcome(Ψ), 
        outcome_parents(adjustment_method, Ψ); 
        verbosity=verbosity
    )
    fit!(Q, dataset; 
        verbosity=verbosity, 
        force=force
    )
    return mean(counterfactual_aggregate(Ψ, Q))
end

"""

# Algorithm

1. Split the dataset in v-folds splits
2. For each split, fit the relevant factors of the distribution on the training sets
3. Perform cross-validated selection of the fluctuation model
    a. For each split compute the clever covariate and offset on the validation sets
    b. Fit a fluctuate on the full validation set
    c. Iterate until epsilon = 0 (usually one step in our case)
4. Compute the targeted estimate
    a. For each validation set compute the estimate from the fluctuated factors
    b. Average those estimates to retrieve the targeted estimate
5. Compute the estimated variance
    a. For each validation set compute the variance of the IC from the fluctuated factors
    b. Average those estimates to retrieve the IC variance and divide per root n

Discussion points:

- More of a CV question, why does the cross-validated selector solves the optimization problem presented in the paper?
"""
function cvtmle!(Ψ::CMCompositeEstimand, dataset;
    resampling=nothing,
    adjustment_method=BackdoorAdjustment(), 
    verbosity=1, 
    force=false, 
    ps_lowerbound=1e-8, 
    weighted_fluctuation=false) 
end


