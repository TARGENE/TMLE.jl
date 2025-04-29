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

function update!(strategy::AdaptiveCorrelationOrdering, last_candidate, dataset)
    residuals = compute_loss(last_candidate.outcome_mean, dataset)
    max_cor = 0.
    best_confounder = :nothing
    for confounder in strategy.remaining_confounders
        confounder_col = unwrap.(Tables.getcolumn(dataset, confounder))
        σ = abs(cor(confounder_col, residuals))
        if σ > max_cor
            max_cor = σ
            best_confounder = confounder
        end
    end
    delete!(strategy.remaining_confounders, best_confounder)
    push!(strategy.current_confounders, best_confounder)
    return nothing
end


function finalise!(strategy::AdaptiveCorrelationOrdering)
    empty!(strategy.remaining_confounders)
    empty!(strategy.current_confounders)
    return nothing
end

function propensity_score(Ψ::StatisticalCMCompositeEstimand, collaborative_strategy::AdaptiveCorrelationOrdering)
    Ψtreatments = TMLE.treatments(Ψ)
    return Tuple(map(eachindex(Ψtreatments)) do index
        T = Ψtreatments[index]
        confounders = (Ψtreatments[index+1:end]..., collaborative_strategy.current_confounders..., :COLLABORATIVE_INTERCEPT)
        ConditionalDistribution(T, confounders)
    end)
end

exhausted(strategy::AdaptiveCorrelationOrdering) = length(strategy.remaining_confounders) == 0