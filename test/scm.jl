module TestSCM

using Test
using TMLE
using MLJGLMInterface

@testset "Test SE" begin
    # Incorrect Structural Equations
    @test_throws TMLE.SelfReferringEquationError(:Y) SE(:Y, [:Y])
    # Correct Structural Equations
    eq = SE(:Y, [:T, :W], LinearBinaryClassifier())
    @test eq isa SE
    @test string(eq) == "Y = f(T, W), LinearBinaryClassifier"

    eq = SE(:Y, [:T, :W])
    @test !isdefined(eq, :model)
    @test string(eq) == "Y = f(T, W)"
end

@testset "Test SCM" begin
    scm = SCM()
    # setindex!
    Yeq = SE(:Y, [:T, :W, :C])
    scm[:Y] = Yeq
    # push!
    Teq = SE(:T, [:W])
    push!(scm, Teq)
    # getindex
    @test scm[:Y] === Yeq
    # getproperty
    @test scm.T === Teq
    # Already defined equation
    @test_throws TMLE.AlreadyAssignedError(:Y) push!(scm, SE(:Y, [:T]))
    # string representation
    @test string(scm) == "Structural Causal Model:\n-----------------------\nT = f₁(W)\nY = f₂(T, W, C)\n"
end

@testset "Test StaticConfoundedModel" begin
    # With covariates
    scm = StaticConfoundedModel(:Y, :T, [:W₁, :W₂], covariates=[:C₁])
    @test TMLE.parents(scm, :Y) == [:C₁, :T, :W₁, :W₂]
    @test TMLE.parents(scm, :T) == [:W₁, :W₂]
    @test scm.Y.model isa LinearRegressor
    @test scm.T.model isa LinearBinaryClassifier
    @test scm.T.model.fit_intercept === true

    # Without covariates
    scm = StaticConfoundedModel(:Y, :T, :W₁, outcome_spec=LinearBinaryClassifier(), treatment_spec=LinearBinaryClassifier(fit_intercept=false))
    @test TMLE.parents(scm, :Y) == [:T, :W₁]
    @test TMLE.parents(scm, :T) == [:W₁]
    @test scm.Y.model isa LinearBinaryClassifier
    @test scm.T.model.fit_intercept === false

    # With 1 covariate
    scm = StaticConfoundedModel(:Y, :T, :W₁, covariates=:C₁)
    @test TMLE.parents(scm, :Y) == [:C₁, :T, :W₁]
    @test TMLE.parents(scm, :T) == [:W₁]
end

end
true