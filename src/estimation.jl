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
    tmle(Ψ::CMCompositeEstimand, dataset; 
        adjustment_method=BackdoorAdjustment(), 
        verbosity=1, 
        force=false, 
        threshold=1e-8, 
        weighted_fluctuation=false
        )

Performs Targeted Minimum Loss Based Estimation of the target estimand.

## Arguments

- Ψ: An estimand of interest.
- dataset: A table respecting the `Tables.jl` interface.
- adjustment_method: A confounding adjustment method.
- verbosity: Level of logging.
- force: To force refit of machines in the SCM .
- threshold: The balancing score will be bounded to respect this threshold.
- weighted_fluctuation: To use a weighted fluctuation instead of the vanilla TMLE, can improve stability.
"""
function tmle(Ψ::CMCompositeEstimand, dataset; 
    adjustment_method=BackdoorAdjustment(), 
    verbosity=1, 
    force=false, 
    threshold=1e-8, 
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
    # TMLE step
    verbosity >= 1 && @info "Performing TMLE..."
    fluctuation_mach = tmle_step(Ψ, verbosity=verbosity, threshold=threshold, weighted_fluctuation=weighted_fluctuation)
    # Estimation results after TMLE
    IC, Ψ̂, ICᵢ, Ψ̂ᵢ = gradient_and_estimates(Ψ, fluctuation_mach, threshold=threshold, weighted_fluctuation=weighted_fluctuation)
    tmle_result = TMLEstimate(Ψ̂, IC)
    one_step_result = OSEstimate(Ψ̂ᵢ + mean(ICᵢ), ICᵢ)

    verbosity >= 1 && @info "Done."
    TMLEResult(Ψ, tmle_result, one_step_result, Ψ̂ᵢ), fluctuation_mach 
end

function tmle_step(Ψ::CMCompositeEstimand; verbosity=1, threshold=1e-8, weighted_fluctuation=false)
    outcome_equation = Ψ.scm[outcome(Ψ)]
    X, y = outcome_equation.mach.data
    outcome_mach = outcome_equation.mach
    # Compute offset
    offset = TMLE.compute_offset(MLJBase.predict(outcome_mach))
    # Compute clever covariate and weights for weighted fluctuation mode
    T = treatments(X, Ψ)
    W = confounders(X, Ψ)
    covariate, weights = TMLE.clever_covariate_and_weights(
        Ψ.scm, W, T, indicator_fns(Ψ); 
        threshold=threshold, 
        weighted_fluctuation=weighted_fluctuation
    )
    # Fit fluctuation
    Xfluct = TMLE.fluctuation_input(covariate, offset)
    mach = machine(F_model(scitype(y)), Xfluct, y, weights, cache=true)
    MLJBase.fit!(mach, verbosity=verbosity-1)
    return mach
end

function counterfactual_aggregates(Ψ::CMCompositeEstimand, fluctuation_mach; threshold=1e-8, weighted_fluctuation=false)
    outcome_eq = outcome_equation(Ψ)
    X = outcome_eq.mach.data[1]
    Ttemplate = treatments(X, Ψ)
    n = nrows(Ttemplate)
    counterfactual_aggregateᵢ = zeros(n)
    counterfactual_aggregate = zeros(n)
    # Loop over Treatment settings
    indicators = indicator_fns(Ψ)
    for (vals, sign) in indicators
        # Counterfactual dataset for a given treatment setting
        T_ct = TMLE.counterfactualTreatment(vals, Ttemplate)
        X_ct = selectcols(merge(X, T_ct), keys(X))
        W = confounders(X_ct, Ψ)
        # Counterfactual predictions with the initial Q
        ŷᵢ = MLJBase.predict(outcome_eq.mach, X_ct)
        counterfactual_aggregateᵢ .+= sign .* expected_value(ŷᵢ)
        # Counterfactual predictions with F
        offset = compute_offset(ŷᵢ)
        covariate, _ = clever_covariate_and_weights(
            Ψ.scm, W, T_ct, indicators; 
            threshold=threshold, weighted_fluctuation=weighted_fluctuation
        )
        Xfluct_ct = fluctuation_input(covariate, offset)
        ŷ = predict(fluctuation_mach, Xfluct_ct)
        counterfactual_aggregate .+= sign .* expected_value(ŷ)
    end
    return counterfactual_aggregate, counterfactual_aggregateᵢ
end

"""
    gradient_W(counterfactual_aggregate, estimate)

∇_W = counterfactual_aggregate - Ψ
"""
gradient_W(counterfactual_aggregate, estimate) =
    counterfactual_aggregate .- estimate


"""
    gradient_Y_X(cache)

∇_YX(w, t, c) = covariate(w, t)  ̇ (y - E[Y|w, t, c])

This part of the gradient is evaluated on the original dataset. All quantities have been precomputed and cached.
"""
function gradients_Y_X(outcome_mach::Machine, fluctuation_mach::Machine)
    X, y, w = fluctuation_mach.data
    y = float(y)
    covariate = X.covariate .* w
    gradient_Y_Xᵢ = covariate .* (y .- expected_value(MLJBase.predict(outcome_mach)))
    gradient_Y_X_fluct = covariate .* (y .- expected_value(MLJBase.predict(fluctuation_mach)))
    return gradient_Y_Xᵢ, gradient_Y_X_fluct
end


function gradient_and_estimates(Ψ::CMCompositeEstimand, fluctuation_mach::Machine; threshold=1e-8, weighted_fluctuation=false)
    counterfactual_aggregate, counterfactual_aggregateᵢ = TMLE.counterfactual_aggregates(
        Ψ, 
        fluctuation_mach; 
        threshold=threshold, 
        weighted_fluctuation=weighted_fluctuation
    )
    Ψ̂, Ψ̂ᵢ = mean(counterfactual_aggregate), mean(counterfactual_aggregateᵢ)
    gradient_Y_Xᵢ, gradient_Y_X_fluct = gradients_Y_X(outcome_equation(Ψ).mach, fluctuation_mach)
    IC = gradient_Y_X_fluct .+ gradient_W(counterfactual_aggregate, Ψ̂)
    ICᵢ = gradient_Y_Xᵢ .+ gradient_W(counterfactual_aggregateᵢ, Ψ̂ᵢ)
    return IC, Ψ̂, ICᵢ, Ψ̂ᵢ
end

