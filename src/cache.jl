"""
This structure is used as a cache for both:
    - η_spec: The specification of the learning algorithms used to estimate the nuisance parameters
    - η: nuisance parameters
    - data: dataset changes

If the input data does not change, then the nuisance parameters do not have to be estimated again.
"""
mutable struct TMLECache
    Ψ::Parameter
    η_spec::NuisanceSpec
    dataset
    η
    function TMLECache(Ψ, η_spec, dataset)
        dataset = Dict(
            :source => dataset,
            :no_missing => nomissing(dataset, allcolumns(Ψ))
        )
        η = NuisanceParameters(nothing, nothing, nothing, nothing)
        new(Ψ, η_spec, dataset, η)
    end
end

function update!(cache::TMLECache, Ψ::Parameter)
    any_variable_changed = false
    if keys(cache.Ψ.treatment) != keys(Ψ.treatment)
        cache.η.G = nothing
        cache.η.Q = nothing
        cache.η.H = nothing
        any_variable_changed = true
    end
    if cache.Ψ.confounders != Ψ.confounders
        cache.η.G = nothing
        cache.η.Q = nothing
        any_variable_changed = true
    end
    if cache.Ψ.covariates != Ψ.covariates
        cache.η.Q = nothing
        any_variable_changed = true
    end
    if cache.Ψ.target != Ψ.target
        cache.η.Q = nothing
        any_variable_changed = true
    end
    cache.η.F = nothing
    # Update no missing dataset
    if any_variable_changed
        cache.dataset[:no_missing] = nomissing(cache.dataset[:source], allcolumns(Ψ))
    end
    cache.Ψ = Ψ
end

function update!(cache::TMLECache, η_spec::NuisanceSpec)
    if cache.η_spec.G != η_spec.G
        cache.η.G = nothing
    end
    if cache.η_spec.Q != η_spec.Q
        cache.η.Q = nothing
    end
    cache.η.F = nothing
    cache.η_spec = η_spec
end

function update!(cache::TMLECache, Ψ::Parameter, η_spec::NuisanceSpec)
    update!(cache, Ψ)
    update!(cache, η_spec)
end

"""
    tmle(Ψ::Parameter, η_spec::NuisanceSpec, dataset; verbosity=1, threshold=1e-8)

Main entrypoint to run the TMLE procedure.

# Arguments

- Ψ: The parameter of interest
- η_spec: The specification for learning `Q_0` and `G_0`
- dataset: A tabular dataset respecting the Table.jl interface
- verbosity: The logging level
- threshold: To avoid small values of Ĝ to cause the "clever covariate" to explode
"""
function tmle(Ψ::Parameter, η_spec::NuisanceSpec, dataset; verbosity=1, threshold=1e-8)
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
    tmle!(η, Ψ, η_spec, dataset, verbosity=verbosity, threshold=threshold)
    
    # Estimation results after TMLE
    IC = gradient(Ψ, η, dataset; threshold=threshold)
    Ψ̂ = estimate(Ψ, η, dataset; threshold=threshold)
    result = PointTMLE(Ψ̂, IC)

    verbosity >= 1 && @info "Thank you."
    return result, resultᵢ, cache
end

"""
    tmle!(cache::TMLECache, Ψ::Parameter; verbosity=1, threshold=1e-8)

Runs the TMLE procedure for the new parameter Ψ while potentially reusing cached nuisance parameters.
"""
function tmle!(cache::TMLECache, Ψ::Parameter; verbosity=1, threshold=1e-8)
    update!(cache, Ψ)
    tmle!(cache, verbosity=verbosity, threshold=threshold)
end

"""
    tmle!(cache::TMLECache, η_spec::NuisanceSpec; verbosity=1, threshold=1e-8)

Runs the TMLE procedure for the new nuisance parameters specification η_spec while potentially reusing cached nuisance parameters.
"""
function tmle!(cache::TMLECache, η_spec::NuisanceSpec; verbosity=1, threshold=1e-8)
    update!(cache, η_spec)
    tmle!(cache, verbosity=verbosity, threshold=threshold)
end

"""
    tmle!(cache::TMLECache, Ψ::Parameter, η_spec::NuisanceSpec; verbosity=1, threshold=1e-8)

Runs the TMLE procedure for the new parameter Ψ and the new nuisance parameters specification η_spec 
while potentially reusing cached nuisance parameters.
"""
function tmle!(cache::TMLECache, Ψ::Parameter, η_spec::NuisanceSpec; verbosity=1, threshold=1e-8)
    update!(cache, Ψ, η_spec)
    tmle!(cache, verbosity=verbosity, threshold=threshold)
end