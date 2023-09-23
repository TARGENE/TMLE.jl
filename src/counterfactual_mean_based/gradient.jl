"""
    counterfactual_aggregate(Ψ::CMCompositeEstimand, Q::Machine)

This is a counterfactual aggregate that depends on the parameter of interest.

For the ATE with binary treatment, confounded by W it is equal to:

``ctf_agg = Q(1, W) - Q(0, W)``

"""
function counterfactual_aggregate(Ψ::CMCompositeEstimand, Q, dataset)
    X = selectcols(dataset, Q.estimand.parents)
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

compute_estimate(ctf_aggregate, ::Nothing) = mean(ctf_aggregate)

compute_estimate(ctf_aggregate, train_validation_indices) =
    mean(compute_estimate(ctf_aggregate[val_indices], nothing) for (_, val_indices) in train_validation_indices)


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
    # Maybe can cache some results (H and E[Y|X]) to improve perf here
    H, weights = clever_covariate_and_weights(Ψ, G, dataset; ps_lowerbound=ps_lowerbound)
    y = float(Tables.getcolumn(dataset, Q.estimand.outcome))
    gradient_Y_X_fluct = H .* weights .* (y .- expected_value(Q, dataset))
    return gradient_Y_X_fluct
end


function gradient_and_estimate(Ψ::CMCompositeEstimand, factors, dataset; ps_lowerbound=1e-8)
    Q = factors.outcome_mean
    G = factors.propensity_score
    ctf_agg = TMLE.counterfactual_aggregate(Ψ, Q, dataset)
    Ψ̂ = TMLE.compute_estimate(ctf_agg, TMLE.train_validation_indices_from_factors(factors))
    IC = TMLE.∇YX(Ψ, Q, G, dataset; ps_lowerbound = ps_lowerbound) .+ TMLE.∇W(ctf_agg, Ψ̂)
    return IC, Ψ̂
end


train_validation_indices_from_ps(::MLConditionalDistribution) = nothing
train_validation_indices_from_ps(factor::SampleSplitMLConditionalDistribution) = factor.train_validation_indices

train_validation_indices_from_factors(factors) = 
    train_validation_indices_from_ps(first(factors.propensity_score))