module TestEstimandOrdering

using TMLE
using Test
using StableRNGs

scm = SCM([
    :Y₁ => [:T₁, :T₂, :W₁, :W₂, :C],
    :Y₂ => [:T₁, :T₃, :W₃, :W₂, :C],
    :T₁ => [:W₁, :W₂],
    :T₂ => [:W₂],
    :T₃ => [:W₃, :W₂,],
])
causal_estimands = [
    ATE(
        outcome=:Y₁, 
        treatment_values=(T₁=(case=1, control=0),)
    ),
    ATE(
        outcome=:Y₁, 
        treatment_values=(T₁=(case=2, control=0),)
    ),
    CM(
        outcome=:Y₁, 
        treatment_values=(T₁=(case=2, control=0), T₂=(case=1, control=0))
    ),
    CM(
        outcome=:Y₁, 
        treatment_values=(T₂=(case=1, control=0),)
    ),
    AIE(
        outcome=:Y₁, 
        treatment_values=(T₁=(case=2, control=0), T₂=(case=1, control=0))
    ),
    ATE(
        outcome=:Y₂, 
        treatment_values=(T₁=(case=2, control=0),)
    ),
    ATE(
        outcome=:Y₂, 
        treatment_values=(T₃=(case=2, control=0),)
    ),
    ATE(
        outcome=:Y₂, 
        treatment_values=(T₁=(case=1, control=0), T₃=(case=2, control=0),)
    ),
]
statistical_estimands = [identify(x, scm) for x in causal_estimands]

@testset "Test misc" begin
    # Test groupby_by_propensity_score
    groups = TMLE.groupby_by_propensity_score(statistical_estimands)
    @test groups[((:T₂, :W₂),)] == [statistical_estimands[4]]
    @test groups[((:T₃, :W₂, :W₃),)] == [statistical_estimands[7]]
    @test groups[((:T₁, :T₃, :W₁, :W₂), (:T₃, :W₂, :W₃))] == [statistical_estimands[end]]
    @test Set(groups[((:T₁, :W₁, :W₂),)]) == Set([statistical_estimands[1], statistical_estimands[2], statistical_estimands[6]])
    @test Set(groups[((:T₁, :T₂, :W₁, :W₂), (:T₂, :W₂))]) == Set([statistical_estimands[3], statistical_estimands[5]])
    @test size(vcat(values(groups)...)) == size(statistical_estimands)
 
    # Test PS groups permutations
    factorial(5)
    permutations = TMLE.propensity_score_group_based_permutation_generator(groups)
    @test length(permutations) == factorial(length(groups))
    for permutation in permutations
        @test Set(permutation) == Set(statistical_estimands)
    end
end
@testset "Test ordering strategies" begin
    # Estimand ID || Required models   
    # 1           || (T₁, Y₁|T₁)       
    # 2           || (T₁, Y₁|T₁)       
    # 3           || (T₁, T₂, Y₁|T₁,T₂)
    # 4           || (T₂, Y₁|T₂)       
    # 5           || (T₁, T₂, Y₁|T₁,T₂)
    # 6           || (T₁, Y₂|T₁)       
    # 7           || (T₃, Y₂|T₃)       
    # 8           || (T₁, T₃, Y₂|T₁,T₃)
    # ----------------------------------
    η_counts = TMLE.nuisance_function_counts(statistical_estimands)
    @test η_counts == Dict(
        TMLE.ConditionalDistribution(:Y₂, (:T₁, :W₁, :W₂))           => 1,
        TMLE.ConditionalDistribution(:Y₂, (:T₁, :T₃, :W₁, :W₂, :W₃)) => 1,
        TMLE.ConditionalDistribution(:Y₁, (:T₁, :T₂, :W₁, :W₂))      => 2,
        TMLE.ConditionalDistribution(:T₁, (:W₁, :W₂))                => 3,
        TMLE.ConditionalDistribution(:T₁, (:T₃, :W₁, :W₂))           => 1,
        TMLE.ConditionalDistribution(:T₁, (:T₂, :W₁, :W₂))           => 2,
        TMLE.ConditionalDistribution(:T₃, (:W₂, :W₃))                => 2,
        TMLE.ConditionalDistribution(:T₂, (:W₂,))                    => 3,
        TMLE.ConditionalDistribution(:Y₁, (:T₁, :W₁, :W₂))           => 2,
        TMLE.ConditionalDistribution(:Y₁, (:T₂, :W₂))                => 1,
        TMLE.ConditionalDistribution(:Y₂, (:T₃, :W₂, :W₃))           => 1
    )
    @test TMLE.evaluate_proxy_costs(statistical_estimands, η_counts) == (5, 11)
    @test TMLE.get_min_maxmem_lowerbound(statistical_estimands) == 3

    # The brute force solution returns the optimal solution
    optimal_ordering = @test_logs (:info, "Lower bound reached, stopping.") TMLE.brute_force_ordering(statistical_estimands, verbosity=1, rng=StableRNG(123))
    @test TMLE.evaluate_proxy_costs(optimal_ordering, η_counts) == (3, 11)
    # Creating a bad ordering
    bad_ordering = statistical_estimands[[1, 7, 3, 6, 2, 5, 8, 4]]
    @test TMLE.evaluate_proxy_costs(bad_ordering, η_counts) == (7, 11)
    # Without the brute force on groups, the solution is not necessarily optimal
    # but still improved
    ordering_from_groups = TMLE.groups_ordering(bad_ordering)
    @test TMLE.evaluate_proxy_costs(ordering_from_groups, η_counts) == (4, 11)
    # Adding a layer of brute forcing results in an optimal ordering
    ordering_from_groups_with_brute_force = TMLE.groups_ordering(bad_ordering, brute_force=true)
    @test TMLE.evaluate_proxy_costs(ordering_from_groups_with_brute_force, η_counts) == (3, 11)
end

@testset "Test ordering strategies with Joint Estimands" begin
    ATE₁ = ATE(
        outcome=:Y₁, 
        treatment_values=(T₁=(case=1, control=0),)
    )
    ATE₂ = ATE(
        outcome=:Y₁, 
        treatment_values=(T₁=(case=2, control=1),)
    )
    joint = JointEstimand(ATE₁, ATE₂)
    ATE₃ = ATE(
        outcome=:Y₁, 
        treatment_values=(T₂=(case=1, control=0),)
    )
    estimands = [identify(x, scm) for x in [ATE₁, ATE₃, joint, ATE₂]]
    η_counts = TMLE.nuisance_function_counts(estimands)
    η_counts == Dict(
        TMLE.ConditionalDistribution(:Y₁, (:T₂, :W₂))      => 1,
        TMLE.ConditionalDistribution(:T₁, (:W₁, :W₂))      => 4,
        TMLE.ConditionalDistribution(:Y₁, (:T₁, :W₁, :W₂)) => 4,
        TMLE.ConditionalDistribution(:T₂, (:W₂,))          => 1
    )
    @test TMLE.get_min_maxmem_lowerbound(estimands) == 2
    @test TMLE.evaluate_proxy_costs(estimands, η_counts) == (4, 4)
    # Brute Force
    optimal_ordering = TMLE.brute_force_ordering(estimands)
    @test TMLE.evaluate_proxy_costs(optimal_ordering, η_counts) == (2, 4)
    # PS Group
    grouped_ordering = TMLE.groups_ordering(estimands, brute_force=true)
    @test TMLE.evaluate_proxy_costs(grouped_ordering, η_counts) == (2, 4)

end

end

true