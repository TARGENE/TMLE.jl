module TestCollaborative

using Test
using TMLE
using MLJLinearModels

TEST_DIR = joinpath(pkgdir(TMLE), "test")
include(joinpath(TEST_DIR, "counterfactual_mean_based", "interactions_simulations.jl"))

@testset "Integration Test AdaptiveCorrelationOrdering" begin
    dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(n=100_000)
    Ψ = AIE(
        outcome = :Y,
        treatment_values = (
            T₁=(case=true, control=false), 
            T₂=(case=true, control=false)
        ),
        treatment_confounders = (
            T₁=[:W₁, :W₂, :W₃],
            T₂=[:W₁, :W₂, :W₃],
        )
    )
    # Define the estimator
    cache = Dict()
    
    resampling = StratifiedCV()
    machine_cache = false
    verbosity = 1
    collaborative_strategy = AdaptiveCorrelationOrdering(resampling=StratifiedCV())
    tmle = TMLEE(;
        models = default_models(;
            Q_continuous = LinearRegressor(),
            G = LogisticClassifier(lambda=0.)
        ),
        collaborative_strategy = collaborative_strategy
    )
    # Initialize the relevant factors: no confounder is present in the propensity score
    η = TMLE.get_relevant_factors(Ψ, collaborative_strategy=collaborative_strategy)
    @test η.propensity_score == (TMLE.ConditionalDistribution(:T₁, (:COLLABORATIVE_INTERCEPT, :T₂)), TMLE.ConditionalDistribution(:T₂, (:COLLABORATIVE_INTERCEPT,)))
    # Estimate the initial factors: check models have been fitted correctly anc cahche is updated
    initial_factors_estimator = TMLE.CMRelevantFactorsEstimator(tmle.resampling, tmle.models)
    η̂ₙ = initial_factors_estimator(η, dataset; 
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=tmle.machine_cache
    )
    for ps_component in η̂ₙ.propensity_score.components
        fp = fitted_params(ps_component.machine)
        fitted_variables = first.(fp.logistic_classifier.coefs)
        @test issubset(fitted_variables, [:COLLABORATIVE_INTERCEPT, :T₂__false])
    end
    @test haskey(cache, η.outcome_mean)
    for ps_component in η.propensity_score
        @test haskey(cache, ps_component)
    end
    # Collaborative Targeted Estimation
    estimator = TMLE.TargetedCMRelevantFactorsEstimator(
        Ψ, 
        η̂ₙ;
        collaborative_strategy=tmle.collaborative_strategy,
        tol=tmle.tol,
        max_iter=tmle.max_iter,
        ps_lowerbound=tmle.ps_lowerbound,
        weighted=tmle.weighted,
        machine_cache=tmle.machine_cache
    )
    ## Check models are correctly retrieved
    models = TMLE.retrieve_models(estimator)
    @test models == Dict(
        :T₁ => tmle.models[:G_default],
        :T₂ => tmle.models[:G_default],
        :Y => tmle.models[:Q_continuous_default]
    )
    ## Check initialisation of the collaborative strategy
    TMLE.initialise!(estimator.collaborative_strategy, Ψ)
    @test estimator.collaborative_strategy.remaining_confounders == Set()
    @test estimator.collaborative_strategy.current_confounders == Set()
    ## Check candidates initialisation: 
    ## - η corresponds to the propensity score with no confounders
    ## - The fluctuation is fitted on the whole dataset
    ## - A vector of (candidate, loss) is returned
    ## - A candidate is a targeted estimate
    candidates = TMLE.initialise_candidates(η, estimator, dataset;
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    )
    candidate, loss = only(candidates)
    @test candidate isa TMLE.MLCMRelevantFactors
    @test candidate.outcome_mean.machine.model isa TMLE.Fluctuation
    @test candidate.propensity_score === η̂ₙ.propensity_score
    @test loss == sum((abs.(TMLE.expected_value(candidate.outcome_mean, dataset) .- dataset.Y))) # The RMSE

    # Check CV candidates initialisation: 
    cv_candidates = TMLE.initialise_cv_candidates(η, dataset, estimator, models;
        cache=Dict(),
        verbosity=verbosity-1,
        machine_cache=machine_cache
        )

    candidates = estimator(η, dataset; 
        cache=cache, 
        verbosity=verbosity,
        machine_cache=tmle.machine_cache
    );

    getfield.(candidates, :loss)

end

end

true