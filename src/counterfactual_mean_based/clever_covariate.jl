
"""
    data_adaptive_ps_lower_bound(Ψ::StatisticalCMCompositeEstimand)

This startegy is from [this paper](https://academic.oup.com/aje/article/191/9/1640/6580570?login=false) 
but the study does not show strictly better behaviour of the strategy so not a default for now.
"""
data_adaptive_ps_lower_bound(n::Int; max_lb=0.1) = 
    min(5 / (√(n)*log(n/5)), max_lb)

ps_lower_bound(n::Int, lower_bound::Nothing; max_lb=0.1) = data_adaptive_ps_lower_bound(n; max_lb=max_lb)
ps_lower_bound(n::Int, lower_bound; max_lb=0.1) = min(max_lb, lower_bound)


function truncate!(v::AbstractVector, ps_lowerbound::AbstractFloat)
    for i in eachindex(v)
        v[i] = max(v[i], ps_lowerbound)
    end
end

"""
    balancing_weights(G::NamedTuple, dataset; ps_lowerbound=1e-8)

G always carries a `propensity_score` and a (possibly `nothing`) `censoring_score`. When the
latter is present, the inverse probability of censoring weights are folded in.
"""
function balancing_weights(G::NamedTuple, dataset; ps_lowerbound=1e-8)
    jointlikelihood = ones(nrows(dataset))
    for Gᵢ ∈ G.propensity_score.components
        jointlikelihood .*= likelihood(Gᵢ, dataset)
    end
    truncate!(jointlikelihood, ps_lowerbound)
    weights = 1. ./ jointlikelihood
    return update_with_ipcw_weights!(weights, G.censoring_score, dataset; ps_lowerbound=ps_lowerbound)
end

"""
    clever_covariate_and_weights(
        Ψ::StatisticalCMCompositeEstimand, 
        G::NamedTuple, 
        dataset; 
        ps_lowerbound=1e-8, 
        weighted_fluctuation=false
    )

Computes the clever covariate and weights that are used to fluctuate the initial Q.

if `weighted_fluctuation = false`:

- ``clever_covariate(t, w) = \\frac{SpecialIndicator(t)}{p(t|w)}`` 
- ``weight(t, w) = 1``

if `weighted_fluctuation = true`:

- ``clever_covariate(t, w) = SpecialIndicator(t)`` 
- ``weight(t, w) = \\frac{1}{p(t|w)}``

where SpecialIndicator(t) is defined in `indicator_fns`.
"""
function clever_covariate_and_weights(
    Ψ::StatisticalCMCompositeEstimand, 
    G::NamedTuple, 
    dataset; 
    ps_lowerbound=1e-8, 
    weighted_fluctuation=false
    )
    # Compute the indicator values
    T = selectcols(dataset, (p.estimand.outcome for p in G.propensity_score.components))
    indic_vals = indicator_values(indicator_fns(Ψ), T)
    weights = balancing_weights(G, dataset; ps_lowerbound=ps_lowerbound)
    if weighted_fluctuation
        return indic_vals, weights
    end
    # Vanilla unweighted fluctuation
    indic_vals .*= weights
    return indic_vals, ones(size(weights, 1))
end
