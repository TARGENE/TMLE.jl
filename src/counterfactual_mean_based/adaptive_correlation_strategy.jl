"""
    AdaptiveCorrelationOrdering()

This strategy adaptively selects the confounding variable that is the most correlated with the last residuals of the outcome mean estimator.
"""
struct AdaptiveCorrelationOrdering <: CollaborativeStrategy
    patience::Int
    remaining_confounders::Set{Symbol}
    current_confounders::Set{Symbol}
    AdaptiveCorrelationOrdering(patience) = new(patience, Set{Symbol}(), Set{Symbol}())
end

AdaptiveCorrelationOrdering(;patience=10) = AdaptiveCorrelationOrdering(patience)

function initialise!(strategy::AdaptiveCorrelationOrdering, Ψ)
    empty!(strategy.remaining_confounders)
    union!(strategy.remaining_confounders, Set(Iterators.flatten(values(Ψ.treatment_confounders))))
    empty!(strategy.current_confounders)
    return nothing
end

function update!(strategy::AdaptiveCorrelationOrdering, g, ĝ)
    treatments = (g_.outcome for g_ in g)
    parents = union((g_.parents for g_ in g)...)
    current_confounders = setdiff(parents, treatments)
    setdiff!(strategy.remaining_confounders, current_confounders)
    union!(strategy.current_confounders, current_confounders)
    return nothing
end

function finalise!(strategy::AdaptiveCorrelationOrdering)
    return nothing
end

exhausted(strategy::AdaptiveCorrelationOrdering) = length(strategy.remaining_confounders) == 0

propensity_score(Ψ::StatisticalCMCompositeEstimand, collaborative_strategy::AdaptiveCorrelationOrdering) =
    propensity_score(Ψ, collaborative_strategy.current_confounders)

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
