module TestSCM

using Test
using TMLE
using MLJGLMInterface
using CategoricalArrays
using Random
using StableRNGs
using MLJBase

@testset "Test SE" begin
    # Incorrect Structural Equations
    @test_throws TMLE.SelfReferringEquationError(:Y) SE(:Y, [:Y])
    # Correct Structural Equations
    # - outcome is a Symbol
    # - parents is a Set of Symbols
    eq = SE("Y", [:T, "W"])
    @test TMLE.outcome(eq) == :Y
    @test TMLE.parents(eq) == Set([:T, :W])
    @test TMLE.string_repr(eq) == "Y = f(T, W)"
end

@testset "Test SCM" begin
    scm = SCM()
    # setequation!
    Yeq = SE(:Y, [:T, :W, :C])
    setequation!(scm, Yeq)
    Teq = SE(:T, [:W])
    setequation!(scm, Teq)
    # getequation
    @test getequation(scm, :Y) == Yeq
    @test getequation(scm, :T) == Teq
    # set and get conditional_distribution with implied natural parents
    cond_dist = set_conditional_distribution!(scm, :T, LinearBinaryClassifier())
    @test get_conditional_distribution(scm, :T) == cond_dist
    # set and get conditional_distribution with provided parents
    cond_dist = set_conditional_distribution!(scm, :Y, LinearBinaryClassifier())
    @test_throws KeyError get_conditional_distribution(scm, :Y, Set([:W, :T]))    
    cond_dist = TMLE.get_or_set_conditional_distribution_from_natural!(scm, :Y, Set([:W, :T]))
    @test cond_dist.model == get_conditional_distribution(scm, :Y).model
    # string representation
    @test TMLE.string_repr(scm) == "Structural Causal Model:\n-----------------------\nT = f₁(W)\nY = f₂(T, W, C)\n"
end

@testset "Test StaticConfoundedModel" begin
    # With covariates
    scm = StaticConfoundedModel(:Y, :T, [:W₁, :W₂], covariates=[:C₁])
    @test TMLE.parents(scm, :Y) == Set([:T, :W₁, :W₂, :C₁])
    @test TMLE.parents(scm, :T) == Set([:W₁, :W₂])
    @test get_conditional_distribution(scm, :T).model isa LinearBinaryClassifier
    @test get_conditional_distribution(scm, :Y).model isa ProbabilisticPipeline

    # Without covariates
    scm = StaticConfoundedModel(:Y, :T, :W₁, outcome_model=LinearBinaryClassifier(), treatment_model=LinearBinaryClassifier(fit_intercept=false))
    @test TMLE.parents(scm, :Y) == Set([:T, :W₁])
    @test TMLE.parents(scm, :T) == Set([:W₁])
    @test get_conditional_distribution(scm, :Y).model isa LinearBinaryClassifier
    @test get_conditional_distribution(scm, :T).model.fit_intercept === false

    # With 1 covariate
    scm = StaticConfoundedModel(:Y, :T, :W₁, covariates=:C₁)
    @test TMLE.parents(scm, :Y) == Set([:T, :W₁, :C₁])
    @test TMLE.parents(scm, :T) == Set([:W₁])

    # With multiple outcomes and treatments
    scm = StaticConfoundedModel(
        [:Y₁, :Y₂, :Y₃],
        [:T₁, :T₂],
        [:W₁, :W₂, :W₃],
        covariates=[:C]
        )
    
    Yparents = Set([:T₁, :T₂, :W₁, :W₂, :W₃, :C])
    for Y in [:Y₁, :Y₂, :Y₃]
        @test parents(scm, Y) == Yparents
        @test get_conditional_distribution(scm, Y, Yparents).model isa ProbabilisticPipeline
    end

    Tparents = Set([:W₁, :W₂, :W₃])
    for T in [:T₁, :T₂]
        @test parents(scm, T) == Tparents
        @test get_conditional_distribution(scm, T, Tparents).model isa LinearBinaryClassifier
    end
end

@testset "Test is_upstream" begin
    scm = SCM(
        SE(:T₁, [:W₁]),
        SE(:T₂, [:W₂]),
        SE(:Y, [:T₁, :T₂]),
    )
    @test TMLE.is_upstream(:T₁, :Y, scm) === true
    @test TMLE.is_upstream(:W₁, :Y, scm) === true
    @test TMLE.is_upstream(:Y, :T₁, scm) === false
end

end
true