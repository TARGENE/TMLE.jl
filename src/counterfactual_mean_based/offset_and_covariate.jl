
data_adaptive_ps_lower_bound(n::Int; max_lb=0.1) = 
    min(5 / (√(n)*log(n/5)), max_lb)

"""
    data_adaptive_ps_lower_bound(Ψ::CMCompositeEstimand)

This startegy is from [this paper](https://academic.oup.com/aje/article/191/9/1640/6580570?login=false) 
but the study does not show strictly better behaviour of the strategy so not a default for now.
"""
function data_adaptive_ps_lower_bound(Ψ::CMCompositeEstimand; max_lb=0.1)
    n = nrows(get_outcome_datas(Ψ)[2])
    return data_adaptive_ps_lower_bound(n; max_lb=max_lb) 
end

ps_lower_bound(Ψ::CMCompositeEstimand, lower_bound::Nothing; max_lb=0.1) = data_adaptive_ps_lower_bound(Ψ; max_lb=max_lb)
ps_lower_bound(Ψ::CMCompositeEstimand, lower_bound; max_lb=0.1) = min(max_lb, lower_bound)


function truncate!(v::AbstractVector, ps_lowerbound::AbstractFloat)
    for i in eachindex(v)
        v[i] = max(v[i], ps_lowerbound)
    end
end

function compute_offset(ŷ::UnivariateFiniteVector{Multiclass{2}})
    μy = expected_value(ŷ)
    logit!(μy)
    return μy
end
compute_offset(ŷ::AbstractVector{<:Distributions.UnivariateDistribution}) = expected_value(ŷ)
compute_offset(ŷ::AbstractVector{<:Real}) = expected_value(ŷ)

compute_offset(Ψ::CMCompositeEstimand) = 
    compute_offset(MLJBase.predict(get_outcome_model(Ψ)))

function balancing_weights(Ψ::CMCompositeEstimand, W, T; ps_lowerbound=1e-8)
    density = ones(nrows(T))
    for colname ∈ Tables.columnnames(T)
        mach = Ψ.scm[colname].mach
        ŷ = MLJBase.predict(mach, W[colname])
        density .*= pdf.(ŷ, Tables.getcolumn(T, colname))
    end
    truncate!(density, ps_lowerbound)
    return 1. ./ density
end

"""
    clever_covariate_and_weights(jointT, W, G, indicator_fns; ps_lowerbound=1e-8, weighted_fluctuation=false)

Computes the clever covariate and weights that are used to fluctuate the initial Q.

if `weighted_fluctuation = false`:

- ``clever_covariate(t, w) = \\frac{SpecialIndicator(t)}{p(t|w)}`` 
- ``weight(t, w) = 1``

if `weighted_fluctuation = true`:

- ``clever_covariate(t, w) = SpecialIndicator(t)`` 
- ``weight(t, w) = \\frac{1}{p(t|w)}``

where SpecialIndicator(t) is defined in `indicator_fns`.
"""
function clever_covariate_and_weights(Ψ::CMCompositeEstimand, X; ps_lowerbound=1e-8, weighted_fluctuation=false)
    # Compute the indicator values
    T = treatments(X, Ψ)
    W = confounders(X, Ψ)
    indic_vals = indicator_values(indicator_fns(Ψ), T)
    weights = balancing_weights(Ψ, W, T, ps_lowerbound=ps_lowerbound)
    if weighted_fluctuation
        return indic_vals, weights
    end
    # Vanilla unweighted fluctuation
    indic_vals .*= weights
    return indic_vals, ones(size(weights, 1))
end

"""
    weighted_covariate(Q::Machine{<:Fluctuation}, Ψ, X; ps_lowerbound=1e-8)

If Q is a fluctuation and caches data, the covariate has already been computed and can be retrieved.
"""
weighted_covariate(Q::Machine{<:FluctuationModel, true}, Ψ, X; ps_lowerbound=1e-8) = Q.cache.weighted_covariate


"""
    weighted_covariate(Q::Machine, Ψ, X; ps_lowerbound=1e-8)

Computes the weighted covariate for the gradient.
"""
function weighted_covariate(Q::Machine, Ψ, X; ps_lowerbound=1e-8)
    H, weights = clever_covariate_and_weights(Ψ, X; ps_lowerbound=ps_lowerbound)
    return H .* weights
end
