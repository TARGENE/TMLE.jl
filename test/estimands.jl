module TestEstimands

using TMLE
using Test


@testset "Test ConditionalDistribution" begin
    distr = ConditionalDistribution("Y", ["C", 1, :A, ])
    @test distr.outcome === :Y
    @test distr.parents === (Symbol("1"), :A, :C)
    @test TMLE.variables(distr) == (:Y, Symbol("1"), :A, :C)
end

@testset "Test CMRelevantFactors" begin
    η = TMLE.CMRelevantFactors(
        outcome_mean=ExpectedValue(:Y, [:T, :W]),
        propensity_score=ConditionalDistribution(:T, [:W])
    )
    @test TMLE.variables(η) == (:Y, :T, :W)

    η = TMLE.CMRelevantFactors(
        outcome_mean=ExpectedValue(:Y, [:T, :W]),
        propensity_score=(
            ConditionalDistribution(:T₁, [:W₁]),
            ConditionalDistribution(:T₂, [:W₂₁, :W₂₂])
        )
    )
    @test TMLE.variables(η) == (:Y, :T, :W, :T₁, :W₁, :T₂, :W₂₁, :W₂₂)
end

end

true