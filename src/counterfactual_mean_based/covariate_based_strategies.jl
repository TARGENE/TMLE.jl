#####################################################################
###                      Greedy                           ###
#####################################################################

struct Greedy <: CollaborativeStrategy
    patience::Int
    max_confounders::Union{Nothing, Int}
    remaining_confounders::Set{Symbol}
    current_confounders::Set{Symbol}
    Greedy(patience, max_confounders) = new(patience, max_confounders, Set{Symbol}(), Set{Symbol}())
end

Greedy(;patience=10, max_confounders=nothing) = Greedy(patience, max_confounders)


function propensity_score_from_state!(state, it)
    new_candidate_confounder = pop!(state)
    confounders_list = (it.collaborative_strategy.current_confounders..., new_candidate_confounder)
    return propensity_score(it.Ψ, confounders_list)
end

function Base.iterate(it::StepKPropensityScoreIterator{Greedy})
    state = copy(it.collaborative_strategy.remaining_confounders)
    g = propensity_score_from_state!(state, it)
    ĝ = build_propensity_score_estimator(g, it.models, it.dataset; train_validation_indices=nothing) 
    return (g, ĝ), state
end

function Base.iterate(it::StepKPropensityScoreIterator{Greedy}, state)
    isempty(state) && return nothing
    g = propensity_score_from_state!(state, it)
    ĝ = build_propensity_score_estimator(g, it.models, it.dataset; train_validation_indices=nothing)
    return (g, ĝ), state
end

#####################################################################
###                AdaptiveCorrelationOrdering                    ###
#####################################################################

"""
    AdaptiveCorrelationOrdering()

This strategy adaptively selects the confounding variable that is the most correlated with the last residuals of the outcome mean estimator.
"""
struct AdaptiveCorrelationOrdering <: CollaborativeStrategy
    patience::Int
    max_confounders::Union{Nothing, Int}
    remaining_confounders::Set{Symbol}
    current_confounders::Set{Symbol}
    AdaptiveCorrelationOrdering(patience, max_confounders) = new(patience, max_confounders, Set{Symbol}(), Set{Symbol}())
end

AdaptiveCorrelationOrdering(;patience=10, max_confounders=nothing) = AdaptiveCorrelationOrdering(patience, max_confounders)

function find_confounder_most_correlated_with_residuals(last_targeted_η̂ₙ, dataset, remaining_confounders)
    residuals = compute_loss(last_targeted_η̂ₙ.outcome_mean, dataset)
    max_cor = 0.
    best_confounder = :nothing
    for confounder in remaining_confounders
        confounder_col = unwrap.(dataset[!, confounder])
        σ = abs(cor(confounder_col, residuals))
        if σ > max_cor
            max_cor = σ
            best_confounder = confounder
        end
    end
    return best_confounder
end

function Base.iterate(it::StepKPropensityScoreIterator{AdaptiveCorrelationOrdering})
    # Find confounder most correlated with residuals
    best_confounder = find_confounder_most_correlated_with_residuals(it.last_targeted_η̂ₙ, it.dataset, it.collaborative_strategy.remaining_confounders)
    # Create a new propensity score with the best confounder and the current list
    confounders_list = (it.collaborative_strategy.current_confounders..., best_confounder)
    g = propensity_score(it.Ψ, confounders_list)
    ĝ = build_propensity_score_estimator(
        g,
        it.models,
        it.dataset;
        train_validation_indices=nothing,
    )
    return (g, ĝ), nothing
end

Base.iterate(it::StepKPropensityScoreIterator{AdaptiveCorrelationOrdering}, state) = nothing

#####################################################################
###             Shared AdaptiveCorrelationOrdering                ###
#####################################################################


function initialise!(strategy::Union{Greedy, AdaptiveCorrelationOrdering}, Ψ)
    empty!(strategy.remaining_confounders)
    union!(strategy.remaining_confounders, Set(Iterators.flatten(values(Ψ.treatment_confounders))))
    empty!(strategy.current_confounders)
    return nothing
end

function update!(strategy::Union{Greedy, AdaptiveCorrelationOrdering}, g, ĝ)
    treatments = (g_.outcome for g_ in g)
    parents = union((g_.parents for g_ in g)...)
    current_confounders = setdiff(parents, treatments)
    setdiff!(strategy.remaining_confounders, current_confounders)
    union!(strategy.current_confounders, current_confounders)
    return nothing
end

finalise!(strategy::Union{Greedy, AdaptiveCorrelationOrdering}) = nothing

function exhausted(strategy::Union{Greedy, AdaptiveCorrelationOrdering})
    if length(strategy.remaining_confounders) == 0
        return true
    elseif strategy.max_confounders !== nothing && 
           length(strategy.current_confounders) >= strategy.max_confounders
        return true
    end
    return false
end

function propensity_score(Ψ, confounders_list)
    Ψtreatments = TMLE.treatments(Ψ)
    return Tuple(map(eachindex(Ψtreatments)) do index
        T = Ψtreatments[index]
        T_confounders = intersect(confounders_list, Ψ.treatment_confounders[T])
        T_parents = (T_confounders..., Ψtreatments[index+1:end]...)
        ConditionalDistribution(T, T_parents)
    end)
end

propensity_score(Ψ::StatisticalCMCompositeEstimand, collaborative_strategy::Union{Greedy, AdaptiveCorrelationOrdering}) =
    propensity_score(Ψ, collaborative_strategy.current_confounders)