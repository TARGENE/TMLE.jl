struct GreedyStrategy <: CollaborativeStrategy
    patience::Int
    remaining_confounders::Set{Symbol}
    current_confounders::Set{Symbol}
    GreedyStrategy(patience) = new(patience, Set{Symbol}(), Set{Symbol}())
end

GreedyStrategy(;patience=10) = GreedyStrategy(patience)

function initialise!(strategy::GreedyStrategy, Ψ)
    empty!(strategy.remaining_confounders)
    union!(strategy.remaining_confounders, Set(Iterators.flatten(values(Ψ.treatment_confounders))))
    empty!(strategy.current_confounders)
    return nothing
end

function update!(strategy::GreedyStrategy, g, ĝ)
    treatments = (g_.outcome for g_ in g)
    parents = union((g_.parents for g_ in g)...)
    current_confounders = setdiff(parents, treatments)
    setdiff!(strategy.remaining_confounders, current_confounders)
    union!(strategy.current_confounders, current_confounders)
    return nothing
end

finalise!(strategy::GreedyStrategy) = nothing

propensity_score(Ψ::StatisticalCMCompositeEstimand, collaborative_strategy::GreedyStrategy) =
    propensity_score(Ψ, collaborative_strategy.current_confounders)

exhausted(strategy::GreedyStrategy) = length(strategy.remaining_confounders) == 0

function propensity_score(Ψ, confounders_list)
    Ψtreatments = TMLE.treatments(Ψ)
    return Tuple(map(eachindex(Ψtreatments)) do index
        T = Ψtreatments[index]
        T_confounders = intersect(confounders_list, Ψ.treatment_confounders[T])
        T_parents = (T_confounders..., Ψtreatments[index+1:end]...)
        ConditionalDistribution(T, T_parents)
    end)
end

function propensity_score_from_state!(state, it)
    new_candidate_confounder = pop!(state)
    confounders_list = (it.collaborative_strategy.current_confounders..., new_candidate_confounder)
    return propensity_score(it.Ψ, confounders_list)
end

function Base.iterate(it::StepKPropensityScoreIterator{GreedyStrategy})
    state = copy(it.collaborative_strategy.remaining_confounders)
    g = propensity_score_from_state!(state, it)
    ĝ = build_propensity_score_estimator(g, it.models, it.dataset; train_validation_indices=nothing) 
    return (g, ĝ), state
end

function Base.iterate(it::StepKPropensityScoreIterator{GreedyStrategy}, state)
    isempty(state) && return nothing
    g = propensity_score_from_state!(state, it)
    ĝ = build_propensity_score_estimator(g, it.models, it.dataset; train_validation_indices=nothing)
    return (g, ĝ), state
end
