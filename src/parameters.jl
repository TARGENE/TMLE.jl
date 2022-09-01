"""
A parameter is a functional, it takes as input a distribution and outputs a real number.
Since the full distribution is usually not necessary, it is replaced by nuisance parameters.
"""
abstract type Parameter end

"""
For a treatment specification T=t and a set of confounding variables and covariates X: 
    Ψₜ = Eₓ[E[Y|T=t, X]]
"""
struct ConditionalMean <: Parameter
    target::Symbol
    treatment::NamedTuple
    confounders::AbstractVector{Symbol}
    covariates::AbstractVector{Symbol}
end

ConditionalMean(;target, treatment, confounders, covariates=[]) = 
    ConditionalMean(target, treatment, confounders, covariates)

struct ATE <: Parameter
    target::Symbol
    treatment::NamedTuple
    confounders::AbstractVector{Symbol}
    covariates::AbstractVector{Symbol}
end

ATE(;target, treatment, confounders, covariates=[]) = 
    ATE(target, treatment, confounders, covariates)


selectcols(data, cols) = data |> TableOperations.select(cols...) |> Tables.columntable

confounders(Ψ::Parameter) = Ψ.confounders
confounders(dataset, Ψ) = selectcols(dataset, confounders(Ψ))

covariates(Ψ::Parameter) = Ψ.covariates
covariates(dataset, Ψ) = selectcols(dataset, covariates(Ψ))

treatments(Ψ::Parameter) = collect(keys(Ψ.treatment))
treatments(dataset, Ψ) = selectcols(dataset, treatments(Ψ))

target(Ψ::Parameter) = Ψ.target
target(dataset, Ψ) = Tables.getcolumn(dataset, target(Ψ))

treatment_and_confounders(Ψ::Parameter) = vcat(confounders(Ψ), treatments(Ψ))

confounders_and_covariates(Ψ::Parameter) = vcat(confounders(Ψ), covariates(Ψ))
confounders_and_covariates(dataset, Ψ) = selectcols(dataset, confounders_and_covariates(Ψ))

Qinputs(dataset, Ψ::Parameter) = 
    selectcols(dataset, vcat(confounders_and_covariates(Ψ), treatments(Ψ)))


mutable struct NuisanceParameters
    Q::Union{Nothing, MLJBase.Machine}
    G::Union{Nothing, MLJBase.Machine}
    H::Union{Nothing, MLJBase.Machine}
    F::Union{Nothing, MLJBase.Machine}
end

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

function nomissing(table)
    sch = Tables.schema(table)
    for type in sch.types
        if nonmissingtype(type) != type
            return table |> 
                   TableOperations.dropmissing |> 
                   Tables.columntable
        end
    end
    return table
end

function nomissing(table, columns)
    columns = selectcols(table, columns)
    return nomissing(columns)
end

"""
This structure stores everything there is to know about an estimation result:
    - Ψ: The parameter of interest
    - estimate: The targeted estimate for that parameter
    - eic: The efficient influence curve
"""
struct EstimationResult
    Ψ::Parameter
    Ψ̂::AbstractFloat
    IC::AbstractVector
end


function update!(cache::TMLECache, Ψ::Parameter)
    if keys(cache.Ψ.treatment) != keys(Ψ.treatment)
        cache.η.G = nothing
        cache.η.Q = nothing
        cache.η.H = nothing
    end
    if cache.Ψ.confounders != Ψ.confounders || cache.Ψ.covariates != Ψ.covariates
        cache.η.G = nothing
        cache.η.Q = nothing
    end
    if cache.Ψ.target != Ψ.target
        cache.η.Q = nothing
    end
    cache.Ψ = Ψ
end

function update!(cache::TMLECache, η_spec::NamedTuple)
    if cache.η_spec.G != η_spec.G
        cache.η.G = nothing
    end
    if cache.η_spec.Q != η_spec.Q
        cache.η.Q = nothing
    end
    cache.η_spec = η_spec
end

function tmle(Ψ::Parameter, η_spec::NamedTuple, dataset; verbosity=1, threshold=1e-8)
    cache = TMLECache(Ψ, η_spec, dataset)
    Ψ, η_spec, dataset, η = cache.Ψ, cache.η_spec, cache.dataset, cache.η
    
    # Initial fit
    verbosity >= 1 && @info "Fitting the nuisance parameters..."
    TMLE.fit!(η, η_spec, Ψ, dataset, verbosity=verbosity)
    
    # Estimation results before TMLE
    dataset = dataset[:no_missing]
    ICᵢ = gradient(Ψ, η, dataset; threshold=threshold)
    Ψ̂ᵢ = estimate(Ψ, η, dataset; threshold=threshold)
    resultᵢ = EstimationResult(Ψ, Ψ̂ᵢ, ICᵢ)
    
    # TMLE step
    verbosity >= 1 && @info "Targeting the nuisance parameters..."
    tmle!(η, Ψ, dataset, verbosity=verbosity, threshold=threshold)
    
    # Estimation results after TMLE
    IC = gradient(Ψ, η, dataset; threshold=threshold)
    Ψ̂ = estimate(Ψ, η, dataset; threshold=threshold)
    result = EstimationResult(Ψ, Ψ̂, IC)

    verbosity >= 1 && @info "Thank you."
    return result, resultᵢ, cache
end

function update!(cache::TMLECache, Ψ::Parameter, η_spec::NamedTuple)
    update!(cache, Ψ)
    update!(cache, η_spec)
    tmle!(cache, verbosity=verbosity)
end

function tmle!(cache::TMLECache; Ψ::Parameter, verbosity=1)
    update!(cache, Ψ)
    tmle!(cache, verbosity=verbosity)
end

function tmle!(cache::TMLECache; η_spec::NamedTuple, verbosity=1)
    update!(cache, η_spec)
    tmle!(cache, verbosity=verbosity)
end

function tmle!(cache::TMLECache; Ψ::Parameter, η_spec::NamedTuple, verbosity=1)
    update!(cache, Ψ, η_spec)
    dataset = TMLE.selectcols(cache.dataset, TMLE.variables(cache.Ψ))
    fit!(cache.η, cache.Ψ, dataset, verbosity=verbosity)
    fluctuate!(cache.η, dataset, verbosity=verbosity)
end

function fit!(η::NuisanceParameters, η_spec, Ψ::Parameter, dataset; verbosity=1)
    # Fitting P(T|W)
    # Only rows with missing values in either W or Tₜ are removed
    if η.G === nothing
        verbosity >= 1 && @info "→ Fitting P(T|W)"
        nomissing_WT = nomissing(dataset[:source], treatment_and_confounders(Ψ))
        W = confounders(nomissing_WT, Ψ)
        T = treatments(nomissing_WT, Ψ)
        mach = machine(η_spec.G, W, adapt(T))
        MLJBase.fit!(mach, verbosity=verbosity-1)
        η.G = mach
    end

    # Fitting E[Y|X]
    if η.Q === nothing
        verbosity >= 1 && @info "→ Fitting E[Y|X]"
        # Data
        X = Qinputs(dataset[:no_missing], Ψ)
        y = target(dataset[:no_missing], Ψ)

        # Fitting the Encoder
        if η.H === nothing
            mach = machine(OneHotEncoder(features=treatments(Ψ), drop_last=true, ordered_factor=false), X)
            MLJBase.fit!(mach, verbosity=verbosity-1)
            η.H = mach
        end

        mach = machine(η_spec.Q, MLJBase.transform(η.H, X), y)
        MLJBase.fit!(mach, verbosity=verbosity-1)
        η.Q = mach
    end
end

function fluctuation_input(dataset, η, Ψ; threshold=1e-8)
    X = Qinputs(dataset, Ψ)
    offset = compute_offset(η.Q, MLJBase.transform(η.H, X))
    indicators = indicator_fns(Ψ)
    W = confounders(X, Ψ)
    T = treatments(X, Ψ)
    covariate = compute_covariate(η.G, W, T, indicators; 
                                        threshold=threshold)
    return fluctuation_input(covariate, offset)
end

interaction_combinations(Ψ::Parameter) =
    Iterators.product((values(Ψ.treatment[T]) for T in treatments(Ψ))...)

function indicator_fns(Ψ::Parameter)
    N = length(treatments(Ψ))
    indicators = Dict()
    for comb in interaction_combinations(Ψ)
        indicators[comb] = (-1)^(N - sum(comb[i] == Ψ.treatment[i].case for i in 1:N))
    end
    return indicators
end

function tmle!(η::NuisanceParameters, Ψ, dataset; verbosity=1, threshold=1e-8)
    X = fluctuation_input(dataset, η, Ψ, threshold=threshold)
    y = target(dataset, Ψ)
    mach = machine(TMLE.fluctuation_model(η.Q.model), X, y)
    MLJBase.fit!(mach, verbosity=verbosity-1)
    η.F = mach
end

function fluctuation_model(Q)
    if Q isa Probabilistic
        return LinearBinaryClassifier(fit_intercept=false, offsetcol = :offset)
    elseif Q isa Deterministic
        return LinearRegressor(fit_intercept=false, offsetcol = :offset)
    else
        throw(ArgumentError("Q should be either Probabilistic or Deterministic"))
    end
end

function outcome_mean(η, Ψ, dataset; threshold=1e-8)
    if η.F isa Nothing
        X = Qinputs(dataset, Ψ)
        return expected_value(MLJBase.predict(η.Q, X), typeof(η.Q.model), target_scitype(η.Q.model))
    else
        X = fluctuation_input(dataset, η, Ψ, threshold=threshold)
        return predict_mean(η.F, X)
    end
end

function counterfactual_aggregate(Ψ, η, dataset; threshold=1e-8)
    WC = TMLE.confounders_and_covariates(dataset, Ψ)
    T = TMLE.treatments(dataset, Ψ)
    counterfactual_aggregate_ = zeros(nrows(T))
    for (vals, sign) in TMLE.indicator_fns(Ψ)
        Tc = TMLE.counterfactualTreatment(vals, T)
        Xc = merge(WC, Tc)
        counterfactual_aggregate_ .+= sign.* TMLE.outcome_mean(η, Ψ, Xc, threshold=threshold)
    end
    return counterfactual_aggregate_
end

function gradient_W(Ψ, η, dataset; threshold=1e-8)
    counterfactual_aggregate_ = counterfactual_aggregate(Ψ, η, dataset; threshold=threshold)
    return counterfactual_aggregate_ .- mean(counterfactual_aggregate_)
end

function gradient_Y_X(Ψ, η, dataset; threshold=1e-8)
    indicators = TMLE.indicator_fns(Ψ)
    W = confounders(dataset, Ψ)
    T = treatments(dataset, Ψ)
    covariate = TMLE.compute_covariate(η.G, W, T, indicators; 
                                        threshold=threshold)
    return covariate .* (target(dataset, Ψ) .- outcome_mean(η, Ψ, dataset, threshold=threshold))
end

gradient(Ψ, η, dataset; threshold=1e-8) = gradient_Y_X(Ψ, η, dataset; threshold=threshold) .+ gradient_W(Ψ, η, dataset; threshold=threshold)

estimate(Ψ, η, dataset; threshold=1e-8) = mean(counterfactual_aggregate(Ψ, η, dataset; threshold=threshold))
