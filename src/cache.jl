
function check_treatment_settings(settings::NamedTuple, levels, treatment_name)
    for (key, val) in zip(keys(settings), settings) 
        any(string(val) .== levels) || 
            throw(ArgumentError(string(
                "The '", key, "' string representation: '", val, "' for treatment ", treatment_name, 
                " in Ψ does not match any level of the corresponding variable in the dataset: ", string.(levels))))
    end
end

function check_treatment_settings(setting, levels, treatment_name)
    any(string(setting) .== levels) || 
            throw(ArgumentError(string(
                "The string representation: '", val, "' for treatment ", treatment_name, 
                " in Ψ does not match any level of the corresponding variable in the dataset: ", string.(levels))))
end

function check_treatment_values(cache, Ψ::Estimand)
    for treatment_name in treatments(Ψ)
        treatment_levels = string.(levels(Tables.getcolumn(cache.data[:source], treatment_name)))
        treatment_settings = getproperty(Ψ.treatment, treatment_name)
        check_treatment_settings(treatment_settings, treatment_levels, treatment_name)
    end
end

function tmle(Ψ::Estimand, dataset; verbosity=1, cache=true, force=false, threshold=1e-8, weighted_fluctuation=false)
    # Initial fit of the SCM's equations
    verbosity >= 1 && @info "Fitting the required equations..."
    fit!(Ψ, dataset; verbosity=verbosity, cache=cache, force=force)
    # TMLE step
    verbosity >= 1 && @info "Performing TMLE..."
    fluctuation_mach = tmle_step(Ψ, verbosity=verbosity, threshold=threshold, weighted_fluctuation=weighted_fluctuation)
    # Estimation results after TMLE
    IC, Ψ̂, ICᵢ, Ψ̂ᵢ = gradient_and_estimates(Ψ, fluctuation_mach, threshold=threshold, weighted_fluctuation=weighted_fluctuation)
    tmle_result = ALEstimate(Ψ̂, IC)
    one_step_result = ALEstimate(Ψ̂ᵢ + mean(ICᵢ), ICᵢ)

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
    W = confounders(X, Ψ)
    T = treatments(X, Ψ)
    covariate, weights = TMLE.clever_covariate_and_weights(
        Ψ.scm, T, W, indicator_fns(Ψ); 
        threshold=threshold, weighted_fluctuation=weighted_fluctuation
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
        X_ct = selectcols(merge(X, T_ct), parents(outcome_eq))
        # Counterfactual predictions with the initial Q
        ŷᵢ = MLJBase.predict(outcome_eq.mach,  X_ct)
        counterfactual_aggregateᵢ .+= sign .* expected_value(ŷᵢ)
        # Counterfactual predictions with F
        offset = compute_offset(ŷᵢ)
        covariate, _ = clever_covariate_and_weights(
            Ψ.scm, T_ct, confounders(X, Ψ), indicators; 
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

