#####################################################################
###                            TMLE                               ###
#####################################################################

mutable struct TMLEE <: Estimator
    models::Dict
    resampling::Union{Nothing, ResamplingStrategy}
    collaborative_strategy::Union{Nothing, CollaborativeStrategy}
    ps_lowerbound::Union{Float64, Nothing}
    weighted::Bool
    tol::Union{Float64, Nothing}
    max_iter::Int
    machine_cache::Bool
end

"""
    TMLEE(;models=default_models(), resampling=nothing, ps_lowerbound=1e-8, weighted=false, tol=nothing, machine_cache=false)

Defines a TMLE estimator using the specified models for estimation of the nuisance parameters. The estimator is a 
function that can be applied to estimate estimands for a dataset.

# Arguments

- models (default: `default_models()`): A Dict(variable => model, ...) where the `variables` are the outcome variables modeled by the `models`.
- resampling (default: nothing): Outer resampling strategy. Setting it to `nothing` (default) falls back to vanilla TMLE while 
any valid `MLJ.ResamplingStrategy` will result in CV-TMLE.
- ps_lowerbound (default: 1e-8): Lowerbound for the propensity score to avoid division by 0. The special value `nothing` will 
result in a data adaptive definition as described in [here](https://pubmed.ncbi.nlm.nih.gov/35512316/).
- weighted (default: false): Whether the fluctuation model is a classig GLM or a weighted version. The weighted fluctuation has 
been show to be more robust to positivity violation in practice.
- tol (default: nothing): Convergence threshold for the TMLE algorithm iterations. If nothing (default), 1/(sample size) will be used. See also `max_iter`.
- max_iter (default: 1): Maximum number of iterations for the TMLE algorithm.
- machine_cache (default: false): Whether MLJ.machine created during estimation should cache data.

# Example

```julia
using MLJLinearModels
tmle = TMLEE()
Ψ̂ₙ, cache = tmle(Ψ, dataset)
```
"""
TMLEE(;models=default_models(), resampling=nothing, collaborative_strategy=nothing, ps_lowerbound=1e-8, weighted=false, tol=nothing, max_iter=1, machine_cache=false) = 
    TMLEE(models, resampling, collaborative_strategy, ps_lowerbound, weighted, tol, max_iter, machine_cache)

function (tmle::TMLEE)(Ψ::StatisticalCMCompositeEstimand, dataset; cache=Dict(), verbosity=1)
    # Check the estimand against the dataset
    check_treatment_levels(Ψ, dataset)
    # Initial fit of the SCM's relevant factors
    relevant_factors = get_relevant_factors(Ψ)
    nomissing_dataset = nomissing(dataset, variables(relevant_factors))
    initial_factors_dataset = choose_initial_dataset(dataset, nomissing_dataset, tmle.resampling)
    initial_factors_estimator = CMRelevantFactorsEstimator(tmle.resampling, tmle.collaborative_strategy, tmle.models)
    initial_factors_estimate = initial_factors_estimator(relevant_factors, initial_factors_dataset; 
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=tmle.machine_cache
    )
    # Get propensity score truncation threshold
    n = nrows(nomissing_dataset)
    ps_lowerbound = ps_lower_bound(n, tmle.ps_lowerbound)
    # Fluctuation initial factors
    verbosity >= 1 && @info "Performing TMLE..."
    targeted_factors_estimator = TargetedCMRelevantFactorsEstimator(
        Ψ, 
        initial_factors_estimate;
        collaborative_strategy=tmle.collaborative_strategy,
        tol=tmle.tol,
        max_iter=tmle.max_iter,
        ps_lowerbound=ps_lowerbound,
        weighted=tmle.weighted,
        machine_cache=tmle.machine_cache
    )
    targeted_factors_estimate = targeted_factors_estimator(relevant_factors, nomissing_dataset; 
        cache=cache, 
        verbosity=verbosity,
        machine_cache=tmle.machine_cache
        )
    # Estimation results after TMLE
    estimation_report = report(targeted_factors_estimate)

    IC = last(estimation_report.gradients)
    Ψ̂ = last(estimation_report.estimates)
    σ̂ = std(IC)
    n = size(IC, 1)
    verbosity >= 1 && @info "Done."
    return TMLEstimate(Ψ, Ψ̂, σ̂, n, IC), cache
end

gradient_and_estimate(::TMLEE, Ψ, factors, dataset; ps_lowerbound=1e-8) = 
    gradient_and_plugin_estimate(Ψ, factors, dataset; ps_lowerbound=ps_lowerbound)

#####################################################################
###                            OSE                                ###
#####################################################################

mutable struct OSE <: Estimator
    models::Dict
    resampling::Union{Nothing, ResamplingStrategy}
    collaborative_strategy::Union{Nothing, CollaborativeStrategy}
    ps_lowerbound::Union{Float64, Nothing}
    machine_cache::Bool
end

"""
    OSE(;models=default_models(), resampling=nothing, ps_lowerbound=1e-8, machine_cache=false)

Defines a One Step Estimator using the specified models for estimation of the nuisance parameters. The estimator is a 
function that can be applied to estimate estimands for a dataset.

# Arguments

- models: A Dict(variable => model, ...) where the `variables` are the outcome variables modeled by the `models`.
- resampling: Outer resampling strategy. Setting it to `nothing` (default) falls back to vanilla estimation while 
any valid `MLJ.ResamplingStrategy` will result in CV-OSE.
- ps_lowerbound: Lowerbound for the propensity score to avoid division by 0. The special value `nothing` will 
result in a data adaptive definition as described in [here](https://pubmed.ncbi.nlm.nih.gov/35512316/).
- machine_cache: Whether MLJ.machine created during estimation should cache data.

# Example

```julia
using MLJLinearModels
models = Dict(:Y => LinearRegressor(), :T => LogisticClassifier())
ose = OSE()
Ψ̂ₙ, cache = ose(Ψ, dataset)
```
"""
OSE(;models=default_models(), resampling=nothing, collaborative_strategy=nothing, ps_lowerbound=1e-8, machine_cache=false) = 
    OSE(models, resampling, collaborative_strategy, ps_lowerbound, machine_cache)

function (ose::OSE)(Ψ::StatisticalCMCompositeEstimand, dataset; cache=Dict(), verbosity=1)
    # Check the estimand against the dataset
    check_treatment_levels(Ψ, dataset)
    # Initial fit of the SCM's relevant factors
    initial_factors = get_relevant_factors(Ψ)
    nomissing_dataset = nomissing(dataset, variables(initial_factors))
    initial_factors_dataset = choose_initial_dataset(dataset, nomissing_dataset, ose.resampling)
    initial_factors_estimator = CMRelevantFactorsEstimator(ose.resampling, ose.collaborative_strategy, ose.models)
    initial_factors_estimate = initial_factors_estimator(
        initial_factors, 
        initial_factors_dataset;
        cache=cache, 
        verbosity=verbosity
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

function gradient_and_estimate(::OSE, Ψ, factors, dataset; ps_lowerbound=1e-8)
    IC, Ψ̂ = gradient_and_plugin_estimate(Ψ, factors, dataset; ps_lowerbound=ps_lowerbound)
    IC_mean = mean(IC)
    IC .-= IC_mean
    return IC, Ψ̂ + IC_mean
end

#####################################################################
###                           NAIVE                               ###
#####################################################################

mutable struct NAIVE <: Estimator
    model::MLJBase.Supervised
end

function (estimator::NAIVE)(Ψ::StatisticalCMCompositeEstimand, dataset; cache=Dict(), verbosity=1)
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


function (estimator::Union{NAIVE, OSE, TMLEE})(causalΨ::CausalCMCompositeEstimands, scm, dataset;
    identification_method=BackdoorAdjustment(),
    cache=Dict(), 
    verbosity=1
    )
    Ψ = identify(identification_method, causalΨ, scm)
    return estimator(Ψ, dataset; cache=cache, verbosity=verbosity)
end