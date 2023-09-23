mutable struct Fluctuation <: MLJBase.Supervised
    Ψ::CMCompositeEstimand
    initial_factors::TMLE.MLCMRelevantFactors
    tol::Union{Nothing, Float64}
    ps_lowerbound::Float64
    weighted::Bool
end

Fluctuation(Ψ, initial_factors; tol=nothing, ps_lowerbound=1e-8, weighted=false) =
    Fluctuation(Ψ, initial_factors, tol, ps_lowerbound, weighted)

one_dimensional_path(target_scitype::Type{T}) where T <: AbstractVector{<:MLJBase.Continuous} = LinearRegressor(fit_intercept=false, offsetcol = :offset)
one_dimensional_path(target_scitype::Type{T}) where T <: AbstractVector{<:Finite} = LinearBinaryClassifier(fit_intercept=false, offsetcol = :offset)

fluctuation_input(covariate::AbstractVector{T}, offset::AbstractVector{T}) where T = (covariate=covariate, offset=offset)

"""

The GLM models require inputs of the same type
"""
fluctuation_input(covariate::AbstractVector{T1}, offset::AbstractVector{T2}) where {T1, T2} = 
    (covariate=covariate, offset=convert(Vector{T1}, offset))

training_expected_value(Q::Machine{<:Fluctuation, }, dataset) = Q.cache.training_expected_value

function clever_covariate_offset_and_weights(model::Fluctuation, X)
    Q⁰ = model.initial_factors.outcome_mean
    G⁰ = model.initial_factors.propensity_score
    offset = compute_offset(MLJBase.predict(Q⁰, X))
    covariate, weights = clever_covariate_and_weights(
        model.Ψ, G⁰, X;
        ps_lowerbound=model.ps_lowerbound,
        weighted_fluctuation=model.weighted
    )
    Xfluct = fluctuation_input(covariate, offset) 
    return Xfluct, weights
end

function MLJBase.fit(model::Fluctuation, verbosity, X, y)
    clever_covariate_and_offset, weights = 
        clever_covariate_offset_and_weights(model, X)
    mach = machine(
        one_dimensional_path(scitype(y)), 
        clever_covariate_and_offset, 
        y, 
        weights, 
        )
    fit!(mach, verbosity=verbosity)

    fitresult = (
        one_dimensional_path   = mach,
        )
    cache = (
        weighted_covariate = clever_covariate_and_offset.covariate .* weights,
        training_expected_value = expected_value(predict(mach))
        )
    return fitresult, cache, nothing
end

function MLJBase.predict(model::Fluctuation, fitresult, X) 
    clever_covariate_and_offset, weights = 
        clever_covariate_offset_and_weights(model, X)
    return MLJBase.predict(fitresult.one_dimensional_path, clever_covariate_and_offset)
end