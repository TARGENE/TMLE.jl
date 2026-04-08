
"""
    data_adaptive_ps_lower_bound(n::Int; max_lb=0.1)

Data-adaptive propensity score truncation level from Gruber et al. (2022):
"Data-Adaptive Selection of the Propensity Score Truncation Level for 
Inverse-Probability–Weighted and Targeted Maximum Likelihood Estimators 
of Marginal Point Treatment Effects" (doi:10.1093/aje/kwac087).

This sets the propensity score lower bound to `5/(√n * log(n/5))`, capped at `max_lb`.
The paper formula is `5/(√n * ln(n))` but uses a slightly modified version here.
This is the default when `ps_lowerbound=nothing`.
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

function balancing_weights(G, dataset; ps_lowerbound=nothing)
    n = nrows(dataset)
    jointlikelihood = ones(n)
    for Gᵢ ∈ G.components
        jointlikelihood .*= likelihood(Gᵢ, dataset)
    end
    actual_lowerbound = ps_lower_bound(n, ps_lowerbound)
    truncate!(jointlikelihood, actual_lowerbound)
    return 1. ./ jointlikelihood
end

"""
    clever_covariate_and_weights(
        Ψ::StatisticalCMCompositeEstimand, 
        Gs::Tuple{Vararg{ConditionalDistributionEstimate}}, 
        dataset; 
        ps_lowerbound=nothing, 
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
    G, 
    dataset; 
    ps_lowerbound=nothing, 
    weighted_fluctuation=false
    )
    # Compute the indicator values
    T = selectcols(dataset, (p.estimand.outcome for p in G.components))
    indic_vals = indicator_values(indicator_fns(Ψ), T)
    weights = balancing_weights(G, dataset; ps_lowerbound=ps_lowerbound)
    if weighted_fluctuation
        return indic_vals, weights
    end
    # Vanilla unweighted fluctuation
    indic_vals .*= weights
    return indic_vals, ones(size(weights, 1))
end