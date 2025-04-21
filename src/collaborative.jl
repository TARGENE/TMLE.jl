abstract type CollaborativeStrategy end

"""
    GreedyCollaboration()

This strategy can be used to adaptively select the best confounding variables for the propensity score fit. It works as follows:

1. The propensity score is fitted with no confounding variables
2. Until convergence (or all confounding variables have been added): for each remaining confounding variable, a new propensity score is trained and an associated targeted estiamtor is built. The estimator with the lowest error is selected.
3. The sequence of models is evaluated via penalized cross-validation.
"""
struct GreedyCollaboration <: CollaborativeStrategy end

function initialise_propensity_score(collaborative_strategy::GreedyCollaboration, Ψ)
    Ψtreatments = TMLE.treatments(Ψ)
    return Tuple(map(eachindex(Ψtreatments)) do index
        T = Ψtreatments[index]
        ConditionalDistribution(T, Ψtreatments[index+1:end])
    end)
end

struct GreedyPSEstimator <: Estimator
    cd_estimators::Dict
    confounders::Vector{Symbol}
end

CollaborativePSEstimator(collaborative_strategy::GreedyCollaboration, cd_estimators) =
    GreedyPSEstimator(cd_estimators, Symbol[])


function (estimator::GreedyPSEstimator)(conditional_distributions, dataset; cache=Dict(), verbosity=1, machine_cache=false)
    # Add Intercept to dataset
    dataset = merge(dataset, (;GREEDY_INTERCEPT=ones(nrows(dataset))))

    # Define the restricted conditional distributions
    treatment_variables = Set([cd.outcome for cd in conditional_distributions])

    restricted_conditional_distributions = map(conditional_distributions) do conditional_distribution
        # Only keep parents in the restricted set or being a treatment variable
        new_parents = filter(x -> x ∈ union(estimator.confounders, treatment_variables), conditional_distribution.parents)
        # Add the intercept
        new_parents = (new_parents..., :GREEDY_INTERCEPT)
        ConditionalDistribution(conditional_distribution.outcome, new_parents)
    end

    return fit_conditional_distributions(estimator.cd_estimators, restricted_conditional_distributions, dataset; 
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=machine_cache
    )
end