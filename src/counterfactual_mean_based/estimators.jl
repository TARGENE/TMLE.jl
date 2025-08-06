#####################################################################
###                            TMLE                               ###
#####################################################################

mutable struct Tmle <: Estimator
    models::Dict
    resampling::Union{Nothing, ResamplingStrategy}
    collaborative_strategy::Union{Nothing, CollaborativeStrategy}
    ps_lowerbound::Union{Float64, Nothing}
    weighted::Bool
    tol::Union{Float64, Nothing}
    max_iter::Int
    machine_cache::Bool
    prevalence::Union{Nothing, Float64}
    function Tmle(
        models, 
        resampling, 
        collaborative_strategy, 
        ps_lowerbound, 
        weighted, 
        tol, 
        max_iter, 
        machine_cache,
        prevalence
    )
        if resampling === nothing && collaborative_strategy !== nothing
            @warn("Collaborative TMLE requires a resampling strategy but none was provided. Using the default resampling strategy.")
            resampling = default_resampling(collaborative_strategy)
        end
        return new(
            models, 
            resampling, 
            collaborative_strategy, 
            ps_lowerbound, 
            weighted, tol, 
            max_iter, 
            machine_cache,
            prevalence
        )
    end
end

"""
    Tmle(;models=default_models(), resampling=nothing, ps_lowerbound=1e-8, weighted=false, tol=nothing, machine_cache=false)

Defines a TMLE estimator using the specified models for estimation of the nuisance parameters. The estimator is a 
function that can be applied to estimate estimands for a dataset.

# Constructor Arguments

- models (default: `default_models()`): A Dict(variable => model, ...) where the `variables` are the outcome variables modeled by the `models`.
- collaborative_strategy (default: nothing): A collaborative strategy to use for the estimation. Then the resampling strategy is used  to evaluate the candidates.
- resampling (default: `default_resampling(collaborative_strategy)`): Outer resampling strategy. Setting it to `nothing` (default) falls back to vanilla TMLE while 
any valid `MLJ.ResamplingStrategy` will result in CV-TMLE.
- ps_lowerbound (default: 1e-8): Lowerbound for the propensity score to avoid division by 0. The special value `nothing` will 
result in a data adaptive definition as described in [here](https://pubmed.ncbi.nlm.nih.gov/35512316/).
- weighted (default: false): Whether the fluctuation model is a classig GLM or a weighted version. The weighted fluctuation has 
been show to be more robust to positivity violation in practice.
- tol (default: nothing): Convergence threshold for the TMLE algorithm iterations. If nothing (default), 1/(sample size) will be used. See also `max_iter`.
- max_iter (default: 1): Maximum number of iterations for the TMLE algorithm.
- machine_cache (default: false): Whether MLJ.machine created during estimation should cache data.
- prevalence (default: nothing): If provided, the prevalence weights will be used to weight the observations to match the true prevalence of the source population. 

# Run Argument

- Ψ: parameter of interest
- dataset: A DataFrame 
- cache (default: Dict()): A dictionary to store nuisance function fits.
- verbosity (default: 1): Verbosity level.
- acceleration (default: `CPU1()`): acceleration strategy for parallelised estimation of nuisance functions.

# Example

```julia
using MLJLinearModels
tmle = Tmle()
Ψ̂ₙ, cache = tmle(Ψ, dataset)
```
"""
function Tmle(;
    models=default_models(), 
    collaborative_strategy=nothing,
    resampling=default_resampling(collaborative_strategy), 
    ps_lowerbound=1e-8, 
    weighted=true, 
    tol=nothing, 
    max_iter=1, 
    machine_cache=false,
    prevalence=nothing
    )
    Tmle(
        models, 
        resampling, 
        collaborative_strategy, 
        ps_lowerbound, 
        weighted, tol, 
        max_iter, 
        machine_cache,
        prevalence
    )
end

function (tmle::Tmle)(Ψ::StatisticalCMCompositeEstimand, dataset; cache=Dict(), verbosity=1, acceleration=CPU1())
    # Check the estimand against the dataset
    check_treatment_levels(Ψ, dataset)
    # Make train-validation pairs
    train_validation_indices = get_train_validation_indices(tmle.resampling, Ψ, dataset)
    # Initial fit of the SCM's relevant factors
    relevant_factors = get_relevant_factors(Ψ, collaborative_strategy=tmle.collaborative_strategy)
    # Check if the dataset is suitable for CCW-TMLE if prevalence is provided
    ccw_check(tmle.prevalence, dataset, relevant_factors)
    nomissing_dataset = nomissing(dataset, variables(relevant_factors))
    initial_factors_dataset = choose_initial_dataset(dataset, nomissing_dataset, train_validation_indices)
    prevalence_weights = get_weights_from_prevalence(tmle.prevalence, collect(skipmissing(initial_factors_dataset[!, relevant_factors.outcome_mean.outcome])))

    initial_factors_estimator = CMRelevantFactorsEstimator(tmle.collaborative_strategy; 
        train_validation_indices=train_validation_indices, 
        models=tmle.models,
        prevalence_weights=prevalence_weights
    )
    verbosity >= 1 && @info "Estimating nuisance parameters."
    
    initial_factors_estimate = initial_factors_estimator(relevant_factors, initial_factors_dataset; 
        cache=cache, 
        verbosity=verbosity-1,
        machine_cache=tmle.machine_cache,
        acceleration=acceleration
    )
    # Get prevalence weights may be different from the initial factors estimate depending on the strategy
    prevalence_weights = get_weights_from_prevalence(tmle.prevalence, nomissing_dataset[!, relevant_factors.outcome_mean.outcome])
    # Get propensity score truncation threshold
    n = nrows(nomissing_dataset)
    ps_lowerbound = ps_lower_bound(n, tmle.ps_lowerbound)
    # Fluctuation initial factors
    targeted_factors_estimator = get_targeted_estimator(
        Ψ, 
        tmle.collaborative_strategy, 
        train_validation_indices,
        initial_factors_estimate;
        tol=tmle.tol,
        max_iter=tmle.max_iter,
        ps_lowerbound=ps_lowerbound,
        weighted=tmle.weighted,
        machine_cache=tmle.machine_cache,
        models=tmle.models,
        prevalence_weights=prevalence_weights
    )
    targeted_factors_estimate = targeted_factors_estimator(relevant_factors, nomissing_dataset; 
        cache=cache, 
        verbosity=verbosity,
        machine_cache=tmle.machine_cache,
        acceleration=acceleration
    )
    # Estimation results after TMLE
    cache[:targeted_factors] = targeted_factors_estimate
    estimation_report = report(targeted_factors_estimate)

    IC = last(estimation_report.gradients)
    Ψ̂ = last(estimation_report.estimates)
    σ̂ = std(IC)
    n = size(IC, 1)
    verbosity >= 1 && @info "Done."
    return TMLEstimate(Ψ, Ψ̂, σ̂, n, IC), cache
end

gradient_and_estimate(::Tmle, Ψ, factors, dataset; ps_lowerbound=1e-8) = 
    gradient_and_plugin_estimate(Ψ, factors, dataset; ps_lowerbound=ps_lowerbound)

#####################################################################
###                            OSE                                ###
#####################################################################

mutable struct Ose <: Estimator
    models::Dict
    resampling::Union{Nothing, ResamplingStrategy}
    ps_lowerbound::Union{Float64, Nothing}
    machine_cache::Bool
end

"""
    Ose(;models=default_models(), resampling=nothing, ps_lowerbound=1e-8, machine_cache=false)

Defines a One Step Estimator using the specified models for estimation of the nuisance parameters. The estimator is a 
function that can be applied to estimate estimands for a dataset.

# Constructor Arguments

- models: A Dict(variable => model, ...) where the `variables` are the outcome variables modeled by the `models`.
- resampling: Outer resampling strategy. Setting it to `nothing` (default) falls back to vanilla estimation while 
any valid `MLJ.ResamplingStrategy` will result in CV-OSE.
- ps_lowerbound: Lowerbound for the propensity score to avoid division by 0. The special value `nothing` will 
result in a data adaptive definition as described in [here](https://pubmed.ncbi.nlm.nih.gov/35512316/).
- machine_cache: Whether MLJ.machine created during estimation should cache data.

# Run Argument

- Ψ: parameter of interest
- dataset: A DataFrame 
- cache (default: Dict()): A dictionary to store nuisance function fits.
- verbosity (default: 1): Verbosity level.
- acceleration (default: `CPU1()`): acceleration strategy for parallelised estimation of nuisance functions.

# Example

```julia
using MLJLinearModels
models = Dict(:Y => LinearRegressor(), :T => LogisticClassifier())
ose = Ose()
Ψ̂ₙ, cache = ose(Ψ, dataset)
```
"""
Ose(;models=default_models(), resampling=nothing, ps_lowerbound=1e-8, machine_cache=false) = 
    Ose(models, resampling, ps_lowerbound, machine_cache)

function (ose::Ose)(Ψ::StatisticalCMCompositeEstimand, dataset; cache=Dict(), verbosity=1, acceleration=CPU1())
    # Check the estimand against the dataset
    check_treatment_levels(Ψ, dataset)
    # Make train-validation pairs
    train_validation_indices = get_train_validation_indices(ose.resampling, Ψ, dataset)
    # Initial fit of the SCM's relevant factors
    initial_factors = get_relevant_factors(Ψ)
    nomissing_dataset = nomissing(dataset, variables(initial_factors))
    initial_factors_dataset = choose_initial_dataset(dataset, nomissing_dataset, ose.resampling)
    initial_factors_estimator = CMRelevantFactorsEstimator(train_validation_indices, ose.models)
    initial_factors_estimate = initial_factors_estimator(
        initial_factors, 
        initial_factors_dataset;
        cache=cache, 
        verbosity=verbosity,
        acceleration=acceleration
    )
    # Get propensity score truncation threshold
    n = nrows(nomissing_dataset)
    ps_lowerbound = ps_lower_bound(n, ose.ps_lowerbound)

    # Gradient and estimate
    IC, Ψ̂ = gradient_and_estimate(ose, Ψ, initial_factors_estimate, nomissing_dataset; ps_lowerbound=ps_lowerbound)
    σ̂ = std(IC)
    n = size(IC, 1)
    verbosity >= 1 && @info "Done."
    return OSEstimate(Ψ, Ψ̂, σ̂, n, IC), cache
end

function gradient_and_estimate(::Ose, Ψ, factors, dataset; ps_lowerbound=1e-8)
    IC, Ψ̂ = gradient_and_plugin_estimate(Ψ, factors, dataset; ps_lowerbound=ps_lowerbound)
    IC_mean = mean(IC)
    IC .-= IC_mean
    return IC, Ψ̂ + IC_mean
end

#####################################################################
###                          PLUGIN                               ###
#####################################################################

mutable struct Plugin <: Estimator
    model::MLJBase.Supervised
end

function (estimator::Plugin)(Ψ::StatisticalCMCompositeEstimand, dataset; cache=Dict(), verbosity=1)
    # Check the estimand against the dataset
    check_treatment_levels(Ψ, dataset)
    # Initial fit of the SCM's relevant factors
    relevant_factors = get_relevant_factors(Ψ)
    nomissing_dataset = nomissing(dataset, variables(relevant_factors))
    outcome_mean_estimate = MLConditionalDistributionEstimator(estimator.model)(
        relevant_factors.outcome_mean, 
        dataset;
        cache=cache,
        verbosity=verbosity
    )
    Ψ̂ = mean(counterfactual_aggregate(Ψ, outcome_mean_estimate, nomissing_dataset))
    return Ψ̂, cache
end

#####################################################################
###                Causal Estimand Estimation                     ###
#####################################################################


function (estimator::Union{Plugin, Ose, Tmle})(causalΨ::CausalCMCompositeEstimands, scm, dataset;
    identification_method=BackdoorAdjustment(),
    cache=Dict(), 
    verbosity=1,
    acceleration=CPU1()
    )
    Ψ = identify(identification_method, causalΨ, scm)
    return estimator(Ψ, dataset; cache=cache, verbosity=verbosity, acceleration=acceleration)
end

#####################################################################
###                        Deprecated                             ###
#####################################################################


@deprecate TMLEE(args...;kwargs...) Tmle(args...;kwargs...)
@deprecate OSE(args...;kwargs...) Ose(args...;kwargs...)
@deprecate NAIVE(args...;kwargs...) Plugin(args...;kwargs...)