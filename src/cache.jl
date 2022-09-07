"""
This structure is used as a cache for both:
    - η_spec: The specification of the learning algorithms used to estimate the nuisance parameters
    - η: nuisance parameters
    - data: dataset changes

If the input data does not change, then the nuisance parameters do not have to be estimated again.
"""
mutable struct TMLECache
    Ψ::Parameter
    η_spec::NamedTuple
    dataset
    η
    function TMLECache(Ψ, η_spec, dataset)
        dataset = Dict(
            :source => dataset,
            :no_missing => nomissing(dataset)
        )
        η = NuisanceParameters(nothing, nothing, nothing, nothing)
        new(Ψ, η_spec, dataset, η)
    end
end

function update!(cache::TMLECache, Ψ::Parameter)
    if keys(cache.Ψ.treatment) != keys(Ψ.treatment)
        cache.η.G = nothing
        cache.η.Q = nothing
        cache.η.H = nothing
    end
    if cache.Ψ.confounders != Ψ.confounders
        cache.η.G = nothing
        cache.η.Q = nothing
    end
    if cache.Ψ.covariates != Ψ.covariates
        cache.η.Q = nothing
    end
    if cache.Ψ.target != Ψ.target
        cache.η.Q = nothing
    end
    cache.η.F = nothing
    cache.Ψ = Ψ
end

function update!(cache::TMLECache, η_spec::NamedTuple)
    if cache.η_spec.G != η_spec.G
        cache.η.G = nothing
    end
    if cache.η_spec.Q != η_spec.Q
        cache.η.Q = nothing
    end
    cache.η.F = nothing
    cache.η_spec = η_spec
end

function update!(cache::TMLECache, Ψ::Parameter, η_spec::NamedTuple)
    update!(cache, Ψ)
    update!(cache, η_spec)
end

function tmle(Ψ::Parameter, η_spec::NamedTuple, dataset; verbosity=1, threshold=1e-8)
    cache = TMLECache(Ψ, η_spec, dataset)
    return tmle!(cache; verbosity=verbosity, threshold=threshold)
end

function tmle!(cache; verbosity=1, threshold=1e-8)
    Ψ, η_spec, dataset, η = cache.Ψ, cache.η_spec, cache.dataset, cache.η
    # Initial fit
    verbosity >= 1 && @info "Fitting the nuisance parameters..."
    TMLE.fit!(η, η_spec, Ψ, dataset, verbosity=verbosity)
    
    # Estimation results before TMLE
    dataset = dataset[:no_missing]
    ICᵢ = gradient(Ψ, η, dataset; threshold=threshold)
    Ψ̂ᵢ = estimate(Ψ, η, dataset; threshold=threshold)
    resultᵢ = PointTMLE(Ψ̂ᵢ, ICᵢ)
    
    # TMLE step
    verbosity >= 1 && @info "Targeting the nuisance parameters..."
    tmle!(η, Ψ, dataset, verbosity=verbosity, threshold=threshold)
    
    # Estimation results after TMLE
    IC = gradient(Ψ, η, dataset; threshold=threshold)
    Ψ̂ = estimate(Ψ, η, dataset; threshold=threshold)
    result = PointTMLE(Ψ̂, IC)

    verbosity >= 1 && @info "Thank you."
    return result, resultᵢ, cache
end

function tmle!(cache::TMLECache, Ψ::Parameter; verbosity=1, threshold=1e-8)
    update!(cache, Ψ)
    tmle!(cache, verbosity=verbosity, threshold=threshold)
end

function tmle!(cache::TMLECache, η_spec::NamedTuple; verbosity=1, threshold=1e-8)
    update!(cache, η_spec)
    tmle!(cache, verbosity=verbosity, threshold=threshold)
end

function tmle!(cache::TMLECache, Ψ::Parameter, η_spec::NamedTuple; verbosity=1, threshold=1e-8)
    update!(cache, Ψ, η_spec)
    fit!(cache, verbosity=verbosity, threshold=threshold)
end