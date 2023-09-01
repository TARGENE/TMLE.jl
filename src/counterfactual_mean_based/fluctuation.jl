
mutable struct ContinuousFluctuation <: MLJBase.Probabilistic
    Ψ::CMCompositeEstimand
    initial_factors::NamedTuple
    tol::Union{Nothing, Float64}
    ps_lowerbound::Float64
    weighted::Bool
end

mutable struct BinaryFluctuation <: MLJBase.Probabilistic
    Ψ::CMCompositeEstimand
    initial_factors::NamedTuple
    tol::Union{Nothing, Float64}
    ps_lowerbound::Float64
    weighted::Bool
end

FluctuationModel = Union{ContinuousFluctuation, BinaryFluctuation}

function Fluctuation(Ψ, initial_factors; tol=nothing, ps_lowerbound=1e-8, weighted=false)
    outcome_scitype = scitype(initial_factors[outcome(Ψ)].machine.data[2])
    if outcome_scitype <: AbstractVector{<:MLJBase.Continuous}
        return ContinuousFluctuation(Ψ, initial_factors, tol, ps_lowerbound, weighted)
    elseif outcome_scitype <: AbstractVector{<:Finite}
        return BinaryFluctuation(Ψ, initial_factors, tol, ps_lowerbound, weighted)
    else
        throw(ArgumentError("Cannot proceed with outcome with target_scitype: $outcome_scitype"))
    end
end

function fluctuate(initial_factors, Ψ; 
        tol=nothing, 
        verbosity=1, 
        weighted_fluctuation=false, 
        ps_lowerbound=1e-8
    )
    Qfluct_model = Fluctuation(Ψ, initial_factors; 
        tol=tol, 
        ps_lowerbound=ps_lowerbound, 
        weighted=weighted_fluctuation
    )
    Q⁰, G⁰ = splitQG(initial_factors, Ψ)
    # Retrieve dataset from Q⁰ 
    X, y = Q⁰.machine.data
    dataset = merge(X, NamedTuple{(Q⁰.outcome, )}([y]))
    Qfluct = ConditionalDistribution(
        Q⁰.outcome, 
        Q⁰.parents,
        Qfluct_model
    )
    fit!(Qfluct, dataset, verbosity=verbosity-1)
    return merge(NamedTuple{(Q⁰.outcome,)}([Qfluct]), G⁰)
end

function splitQG(factors, Ψ)
    Q = factors[outcome(Ψ)]
    G = (; (key => factors[key] for key in treatments(Ψ))...)
    return Q, G
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

training_expected_value(Q::Machine{<:FluctuationModel, }) = Q.cache.training_expected_value

function clever_covariate_offset_and_weights(model::FluctuationModel, X)
    Ψ = model.Ψ
    Q⁰, G⁰ = splitQG(model.initial_factors, Ψ)
    offset = compute_offset(MLJBase.predict(Q⁰, X))
    covariate, weights = clever_covariate_and_weights(
        Ψ, G⁰, X;
        ps_lowerbound=model.ps_lowerbound,
        weighted_fluctuation=model.weighted
    )
    Xfluct = fluctuation_input(covariate, offset) 
    return Xfluct, weights
end

function MLJBase.fit(model::FluctuationModel, verbosity, X, y)
    clever_covariate_and_offset, weights = 
        clever_covariate_offset_and_weights(model, X)
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
    clever_covariate_and_offset, weights = 
        clever_covariate_offset_and_weights(model, X)
    return MLJBase.predict(fitresult.one_dimensional_path, clever_covariate_and_offset)
end