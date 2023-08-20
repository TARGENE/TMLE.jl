
mutable struct ContinuousFluctuation <: MLJBase.Probabilistic
    Ψ::CMCompositeEstimand
    tol::Float64
    ps_lowerbound::Float64
    weighted::Bool
end

mutable struct BinaryFluctuation <: MLJBase.Probabilistic
    Ψ::CMCompositeEstimand
    tol::Float64
    ps_lowerbound::Float64
    weighted::Bool
end

FluctuationModel = Union{ContinuousFluctuation, BinaryFluctuation}

function Fluctuation(Ψ, tol, ps_lowerbound, weighted)
    outcome_scitype = scitype(get_outcome_datas(Ψ)[2])
    if outcome_scitype <: AbstractVector{<:MLJBase.Continuous}
        return ContinuousFluctuation(Ψ, tol, ps_lowerbound, weighted)
    elseif outcome_scitype <: AbstractVector{<:Finite}
        return BinaryFluctuation(Ψ, tol, ps_lowerbound, weighted)
    else
        throw(ArgumentError("Cannot proceed with outcome with target_scitype: $outcome_scitype"))
    end
end

epsilon(Qstar) = fitted_params(fitted_params(Qstar).fitresult).coef[1]

one_dimensional_path(model::ContinuousFluctuation) = LinearRegressor(fit_intercept=false, offsetcol = :offset)
one_dimensional_path(model::BinaryFluctuation) = LinearBinaryClassifier(fit_intercept=false, offsetcol = :offset)

fluctuation_input(covariate::AbstractVector{T}, offset::AbstractVector{T}) where T = (covariate=covariate, offset=offset)

"""

The GLM models require inputs of the same type
"""
fluctuation_input(covariate::AbstractVector{T1}, offset::AbstractVector{T2}) where {T1, T2} = 
    (covariate=covariate, offset=convert(Vector{T1}, offset))


training_expected_value(Q::Machine{<:FluctuationModel}) = Q.cache.training_expected_value

function clever_covariate_offset_and_weights(Ψ, Q, X; 
    ps_lowerbound=1e-8, 
    weighted_fluctuation=false
    )
    offset = compute_offset(MLJBase.predict(Q, X))
    covariate, weights = TMLE.clever_covariate_and_weights(
        Ψ, X; 
        ps_lowerbound=ps_lowerbound, 
        weighted_fluctuation=weighted_fluctuation
    )
    Xfluct = fluctuation_input(covariate, offset) 
    return Xfluct, weights
end

function MLJBase.fit(model::FluctuationModel, verbosity, X, y)
    Ψ = model.Ψ
    Q = get_outcome_model(Ψ)
    clever_covariate_and_offset, weights = 
        clever_covariate_offset_and_weights(
            Ψ, Q, X; 
            ps_lowerbound=model.ps_lowerbound, 
            weighted_fluctuation=model.weighted
    )
    mach = machine(
        one_dimensional_path(model), 
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

function MLJBase.predict(model::FluctuationModel, fitresult, X) 
    Ψ = model.Ψ
    Q = get_outcome_model(Ψ)
    clever_covariate_and_offset, weights = 
        clever_covariate_offset_and_weights(
            Ψ, Q, X; 
            ps_lowerbound=model.ps_lowerbound, 
            weighted_fluctuation=model.weighted
    )
    return MLJBase.predict(fitresult.one_dimensional_path, clever_covariate_and_offset)
end