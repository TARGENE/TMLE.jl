module TestEstimands

using TMLE
using Test

scm = SCM(
    SE(:Y, [:T, :W]),
    SE(:T₁, [:W]),
    SE(:T₂, [:W])
    )
Q̄₀  = TMLE.ExpectedValue(scm, :Y, [:T₁, :T₂, :W])
G₀₁ = TMLE.ConditionalDistribution(scm, :T₁, [:W])
G₀₂ = TMLE.ConditionalDistribution(scm, :T₂, [:W])

@testset "Test ConditionalDistribution" begin
    @test TMLE.estimand_key(Q̄₀) == (:Y, Set([:W, :T₁, :T₂]))  
    @test TMLE.featurenames(Q̄₀) == [:T₁, :T₂, :W]
    @test TMLE.variables(Q̄₀) == Set([:Y, :W, :T₁, :T₂])
    @test TMLE.variables(G₀₁) == Set([:W, :T₁])
end

@testset "Test CMRelevantFactors" begin
    expected_key = (
        (:Y, Set([:W, :T₁, :T₂])), 
        (:T₁, Set([:W])), 
        (:T₂, Set([:W]))
    )
    Q₀ = TMLE.CMRelevantFactors(scm, Q̄₀, (G₀₁, G₀₂))
    @test TMLE.estimand_key(Q₀) == expected_key
    Q₀ = TMLE.CMRelevantFactors(scm, Q̄₀, (G₀₂, G₀₁))
    @test TMLE.estimand_key(Q₀) == expected_key

    @test TMLE.variables(Q₀) == Set([:Y, :W, :T₁, :T₂])
end

end

true