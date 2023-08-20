function MLJBase.fit!(Ψ::CMCompositeEstimand, dataset; adjustment_method=BackdoorAdjustment(), verbosity=1, force=false) 
    models_input_variables = get_models_input_variables(adjustment_method, Ψ)
    for (variable, input_variables) in zip(keys(models_input_variables), models_input_variables)
        fit!(Ψ.scm[variable], dataset; 
            input_variables=input_variables, 
            verbosity=verbosity, 
            force=force
        )
    end
end

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
    fit!(Ψ, dataset;
        adjustment_method=adjustment_method, 
        verbosity=verbosity, 
        force=force
    )
    # Get propensity score truncation threshold
    ps_lowerbound = ps_lower_bound(Ψ, ps_lowerbound)
    # Fit Fluctuation
    verbosity >= 1 && @info "Performing TMLE..."
    Q⁰ = getQ(Ψ)
    X, y = Q⁰.data
    Q = machine(
        Fluctuation(Ψ, 0.1, ps_lowerbound, weighted_fluctuation), 
        X, 
        y
    )
    fit!(Q, verbosity=verbosity-1)
    # Estimation results after TMLE
    Ψ̂⁰ = mean(counterfactual_aggregate(Ψ, Q⁰))
    IC, Ψ̂ = gradient_and_estimate(Ψ, Q; ps_lowerbound=ps_lowerbound)
    verbosity >= 1 && @info "Done."
    return TMLEstimate(Ψ̂⁰, Ψ̂, IC), Q
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
    Q = getQ(Ψ)
    # Gradient and estimate
    IC, Ψ̂ = gradient_and_estimate(Ψ, Q; ps_lowerbound=ps_lowerbound)
    verbosity >= 1 && @info "Done."
    return OSEstimate(Ψ̂, Ψ̂ + mean(IC), IC), Q
end

