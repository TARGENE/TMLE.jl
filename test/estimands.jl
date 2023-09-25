module TestEstimands

using TMLE
using Test

@testset "Test ConditionalDistribution" begin
    distr = TMLE.ConditionalDistribution("Y", ["C", 1, :A, ])
    @test distr.outcome === :Y
    @test distr.parents === (Symbol("1"), :A, :C)
    @test TMLE.variables(distr) == (:Y, Symbol("1"), :A, :C)
end

@testset "Test CMRelevantFactors" begin
    η = TMLE.CMRelevantFactors(
        outcome_mean=TMLE.ExpectedValue(:Y, [:T, :W]),
        propensity_score=TMLE.ConditionalDistribution(:T, [:W])
    )
    @test TMLE.variables(η) == (:Y, :T, :W)

    η = TMLE.CMRelevantFactors(
        outcome_mean=TMLE.ExpectedValue(:Y, [:T, :W]),
        propensity_score=(
            TMLE.ConditionalDistribution(:T₁, [:W₁]),
            TMLE.ConditionalDistribution(:T₂, [:W₂₁, :W₂₂])
        )
    )
    @test TMLE.variables(η) == (:Y, :T, :W, :T₁, :W₁, :T₂, :W₂₁, :W₂₂)
end

end

true