"""
    counterfactual_aggregate(Ψ::CMCompositeEstimand, Q::Machine)

This is a counterfactual aggregate that depends on the parameter of interest.

For the ATE with binary treatment, confounded by W it is equal to:

``ctf_agg = Q(1, W) - Q(0, W)``

"""
function counterfactual_aggregate(Ψ::CMCompositeEstimand, Q, dataset)
    X = selectcols(dataset, featurenames(Q))
    Ttemplate = selectcols(X, treatments(Ψ))
    n = nrows(Ttemplate)
    ctf_agg = zeros(n)
    # Loop over Treatment settings
    for (vals, sign) in indicator_fns(Ψ)
        # Counterfactual dataset for a given treatment setting
        T_ct = TMLE.counterfactualTreatment(vals, Ttemplate)
        X_ct = merge(X, T_ct)
        # Counterfactual mean
        ctf_agg .+= sign .* expected_value(Q, X_ct)
    end
    return ctf_agg
end

"""
    ∇W(ctf_agg, Ψ̂)

∇_W = ctf_agg - Ψ̂
"""
∇W(ctf_agg, Ψ̂) = ctf_agg .- Ψ̂

"""
    gradient_Y_X(cache)

∇_YX(w, t, c) = covariate(w, t)  ̇ (y - E[Y|w, t, c])

This part of the gradient is evaluated on the original dataset. All quantities have been precomputed and cached.
"""
function ∇YX(Ψ::CMCompositeEstimand, Q, G, dataset; ps_lowerbound=1e-8)
    Qmach = Q.machine
    H = weighted_covariate(Qmach, G, Ψ, dataset; ps_lowerbound=ps_lowerbound)
    y = float(Tables.getcolumn(dataset, Q.estimand.outcome))
    gradient_Y_X_fluct = H .* (y .- training_expected_value(Qmach, dataset))
    return gradient_Y_X_fluct
end


function gradient_and_estimate(Ψ::CMCompositeEstimand, factors, dataset; ps_lowerbound=1e-8)
    Q = factors.outcome_mean
    G = factors.propensity_score
    ctf_agg = counterfactual_aggregate(Ψ, Q, dataset)
    Ψ̂ = mean(ctf_agg)
    IC = ∇YX(Ψ, Q, G, dataset; ps_lowerbound = ps_lowerbound) .+ ∇W(ctf_agg, Ψ̂)
    return IC, Ψ̂
end