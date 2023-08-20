"""
    counterfactual_aggregate(Ψ::CMCompositeEstimand, Q::Machine)

This is a counterfactual aggregate that depends on the parameter of interest.

For the ATE with binary treatment, confounded by W it is equal to:

``ctf_agg = Q(1, W) - Q(0, W)``

"""
function counterfactual_aggregate(Ψ::CMCompositeEstimand, Q::Machine)
    X = get_outcome_datas(Ψ)[1]
    Ttemplate = treatments(X, Ψ)
    n = nrows(Ttemplate)
    ctf_agg = zeros(n)
    # Loop over Treatment settings
    for (vals, sign) in indicator_fns(Ψ)
        # Counterfactual dataset for a given treatment setting
        T_ct = counterfactualTreatment(vals, Ttemplate)
        X_ct = selectcols(merge(X, T_ct), keys(X))
        # Counterfactual predictions with F
        ŷ = predict(Q, X_ct)
        ctf_agg .+= sign .* expected_value(ŷ)
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
function ∇YX(Ψ::CMCompositeEstimand, Q::Machine; ps_lowerbound=1e-8)
    X, y = get_outcome_datas(Ψ)
    H = weighted_covariate(Q, Ψ, X; ps_lowerbound=ps_lowerbound)
    y = float(y)
    gradient_Y_X_fluct = H .* (y .- training_expected_value(Q))
    return gradient_Y_X_fluct
end


function gradient_and_estimate(Ψ::CMCompositeEstimand, Q::Machine; ps_lowerbound=1e-8)
    ctf_agg = counterfactual_aggregate(Ψ, Q)
    Ψ̂ = mean(ctf_agg)
    IC = ∇YX(Ψ, Q; ps_lowerbound = ps_lowerbound) .+ ∇W(ctf_agg, Ψ̂)
    return IC, Ψ̂
end