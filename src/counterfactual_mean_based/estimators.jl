choose_initial_dataset(dataset, nomissing_dataset, resampling::Nothing) = dataset
choose_initial_dataset(dataset, nomissing_dataset, resampling) = nomissing_dataset

function tmle!(Ψ::CMCompositeEstimand, models, dataset;
    resampling=nothing,
    adjustment_method=BackdoorAdjustment(), 
    verbosity=1, 
    force=false, 
    ps_lowerbound=1e-8, 
    weighted_fluctuation=false,
    factors_cache=nothing
    )
    # Check the estimand against the dataset
    TMLE.check_treatment_levels(Ψ, dataset)
    # Initial fit of the SCM's equations
    verbosity >= 1 && @info "Fitting the required equations..."
    relevant_factors = TMLE.get_relevant_factors(Ψ; adjustment_method=adjustment_method)
    nomissing_dataset = TMLE.nomissing(dataset, TMLE.variables(relevant_factors))
    initial_factors_dataset = TMLE.choose_initial_dataset(dataset, nomissing_dataset, resampling)
    initial_factors_estimate = TMLE.estimate(relevant_factors, resampling, models, initial_factors_dataset; 
        factors_cache=factors_cache, 
        verbosity=verbosity
    )
    # Get propensity score truncation threshold
    n = nrows(nomissing_dataset)
    ps_lowerbound = TMLE.ps_lower_bound(n, ps_lowerbound)
    # Fit Fluctuation
    verbosity >= 1 && @info "Performing TMLE..."
    targeted_factors_estimate = TMLE.fluctuate(initial_factors_estimate, Ψ, nomissing_dataset;
        tol=nothing, 
        verbosity=verbosity, 
        weighted_fluctuation=weighted_fluctuation, 
        ps_lowerbound=ps_lowerbound
    )
    # Estimation results after TMLE
    IC, Ψ̂ = TMLE.gradient_and_estimate(Ψ, targeted_factors_estimate, nomissing_dataset; ps_lowerbound=ps_lowerbound)
    verbosity >= 1 && @info "Done."
    return TMLEstimate(Ψ̂, IC), targeted_factors_estimate
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