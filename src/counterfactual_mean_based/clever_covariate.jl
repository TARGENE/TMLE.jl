
"""
    data_adaptive_ps_lower_bound(n::Int; max_lb=0.1)

Data-adaptive propensity score truncation level from Gruber et al. (2022):
"Data-Adaptive Selection of the Propensity Score Truncation Level for 
Inverse-Probability–Weighted and Targeted Maximum Likelihood Estimators 
of Marginal Point Treatment Effects" (doi:10.1093/aje/kwac087).

This is the default when `ps_lowerbound=nothing`. Here a maximum lower bound
is applied to prevent extreme truncation in small samples.
"""
data_adaptive_ps_lower_bound(n::Int; max_lb=0.1) = 
    min(5 / (√(n)*log(n)), max_lb)

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

Computes the clever covariate matrix, weights, and signs used to fluctuate the initial Q.

Returns a tuple `(H, w, signs)` where:
- `H` is an `n × K` matrix, one column per counterfactual in `indicator_fns(Ψ)`.
- `w` is a length-`n` weight vector.
- `signs` is a length-`K` vector of signs from `indicator_fns(Ψ)`.

if `weighted_fluctuation = false`:

- ``H_{k}(t, w) = \\frac{I_k(t)}{p(t|w)}`` 
- ``w(t, w) = 1``

if `weighted_fluctuation = true`:

- ``H_{k}(t, w) = I_k(t)`` 
- ``w(t, w) = \\frac{1}{p(t|w)}``

where ``I_k(t)`` is the unsigned indicator for the k-th counterfactual.
"""
function clever_covariate_and_weights(
    Ψ::StatisticalCMCompositeEstimand, 
    G, 
    dataset; 
    ps_lowerbound=nothing, 
    weighted_fluctuation=false
    )
    # Compute the indicator matrix (n×K) and signs (K,)
    T = selectcols(dataset, (p.estimand.outcome for p in G.components))
    indic_mat, signs = indicator_matrix(indicator_fns(Ψ), T)
    weights = balancing_weights(G, dataset; ps_lowerbound=ps_lowerbound)
    if weighted_fluctuation
        return indic_mat, weights, signs
    end
    # Vanilla unweighted fluctuation
    indic_mat .*= weights
    return indic_mat, ones(size(weights, 1)), signs
end