"""
    counterfactual_aggregate(Ψ::StatisticalCMCompositeEstimand, Q::Machine)

This is a counterfactual aggregate that depends on the parameter of interest.

For the ATE with binary treatment, confounded by W it is equal to:

``ctf_agg = Q(1, W) - Q(0, W)``

"""
function counterfactual_aggregate(Ψ::StatisticalCMCompositeEstimand, Q, dataset)
    X = selectcols(dataset, Q.estimand.parents)
    Ttemplate = selectcols(X, treatments(Ψ))
    n = nrows(Ttemplate)
    ctf_agg = zeros(n)
    # Loop over Treatment settings
    for (vals, sign) in indicator_fns(Ψ)
        # Counterfactual dataset for a given treatment setting
        T_ct = counterfactualTreatment(vals, Ttemplate)
        X_ct = merge(X, T_ct)
        # Counterfactual mean
        ctf_agg .+= sign .* expected_value(Q, X_ct)
    end
    return ctf_agg
end

plugin_estimate(ctf_aggregate) = mean(ctf_aggregate)


"""
    ∇W(ctf_agg, Ψ̂)

Computes the projection of the gradient on the (W) space.
"""
∇W(ctf_agg, Ψ̂) = ctf_agg .- Ψ̂

"""
    ∇YX(H, y, Ey, w)

Computes the projection of the gradient on the (Y | X) space.

- H: Clever covariate
- y: Outcome
- Ey: Expected value of the outcome
- w: Weights
"""
∇YX(H, y, Ey, w) = H .* w .* (y .- Ey)

"""
    gradient_Y_X(cache)

∇_YX(w, t, c) = covariate(w, t)  ̇ (y - E[Y|w, t, c])

This part of the gradient is evaluated on the original dataset. All quantities have been precomputed and cached.
"""
function ∇YX(Ψ::StatisticalCMCompositeEstimand, Q, G, dataset; ps_lowerbound=1e-8)
    # Maybe can cache some results (H and E[Y|X]) to improve perf here
    H, w = clever_covariate_and_weights(Ψ, G, dataset; ps_lowerbound=ps_lowerbound)
    y = float(Tables.getcolumn(dataset, Q.estimand.outcome))
    Ey = expected_value(Q, dataset)
    return ∇YX(H, y, Ey, w)
end


function gradient_and_plugin_estimate(Ψ::StatisticalCMCompositeEstimand, factors, dataset; ps_lowerbound=1e-8)
    Q = factors.outcome_mean
    G = factors.propensity_score
    ctf_agg = counterfactual_aggregate(Ψ, Q, dataset)
    Ψ̂ = plugin_estimate(ctf_agg)
    IC = ∇YX(Ψ, Q, G, dataset; ps_lowerbound = ps_lowerbound) .+ ∇W(ctf_agg, Ψ̂)
    return IC, Ψ̂
end

train_validation_indices_from_ps(::MLConditionalDistribution) = nothing
train_validation_indices_from_ps(factor::SampleSplitMLConditionalDistribution) = factor.train_validation_indices

train_validation_indices_from_factors(factors) = 
    train_validation_indices_from_ps(first(factors.propensity_score))