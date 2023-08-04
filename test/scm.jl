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
    eq = SE(:Y, [:T, :W], LinearBinaryClassifier())
    @test eq isa SE
    @test TMLE.string_repr(eq) == "Y = f(T, W), LinearBinaryClassifier, fitted=false"

    eq = SE(:Y, [:T, :W])
    @test TMLE.string_repr(eq) == "Y = f(T, W)"
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
    @test TMLE.string_repr(scm) == "Structural Causal Model:\n-----------------------\nT = f₁(W)\nY = f₂(T, W, C)\n"
end

@testset "Test StaticConfoundedModel" begin
    # With covariates
    scm = StaticConfoundedModel(:Y, :T, [:W₁, :W₂], covariates=[:C₁])
    @test TMLE.parents(scm, :Y) == [:T, :W₁, :W₂, :C₁]
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
    @test TMLE.parents(scm, :Y) == [:T, :W₁, :C₁]
    @test TMLE.parents(scm, :T) == [:W₁]

    # With multiple outcomes and treatments
    scm = StaticConfoundedModel(
        [:Y₁, :Y₂, :Y₃],
        [:T₁, :T₂],
        [:W₁, :W₂, :W₃],
        covariates=[:C]
        )
    
    Yparents = [:T₁, :T₂, :W₁, :W₂, :W₃, :C]
    @test parents(scm.Y₁) == Yparents
    @test parents(scm.Y₂) == Yparents
    @test parents(scm.Y₃) == Yparents

    Tparents = [:W₁, :W₂, :W₃]
    @test parents(scm.T₁) == Tparents
    @test parents(scm.T₂) == Tparents

end

@testset "Test fit!" begin
    rng = StableRNG(123)
    n = 100
    dataset = (
        Ycat = categorical(rand(rng, [0, 1], n)),
        Ycont = rand(rng, n),
        T₁ = categorical(rand(rng, [0, 1], n)),
        T₂ = categorical(rand(rng, [0, 1], n)),
        W₁₁ = rand(rng, n),
        W₁₂ = rand(rng, n),
        W₂₁ = rand(rng, n),
        W₂₂ = rand(rng, n),
        C = rand(rng, n)
    )

    scm = SCM(
        SE(:Ycat, [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁, :W₂₂, :C], TreatmentTransformer() |> LinearBinaryClassifier()),
        SE(:Ycont, [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁, :W₂₂, :C],  TreatmentTransformer() |> LinearRegressor()),
        SE(:T₁, [:W₁₁, :W₁₂], LinearBinaryClassifier()),
        SE(:T₂, [:W₂₁, :W₂₂], LinearBinaryClassifier()),
    )

    # Fits all equations in SCM
    fit_log_sequence = (
        (:info, "Fitting Structural Equation corresponding to variable Ycont."),
        (:info, "Fitting Structural Equation corresponding to variable Ycat."),
        (:info, "Fitting Structural Equation corresponding to variable T₁."),
        (:info, "Fitting Structural Equation corresponding to variable T₂."),
    )
    @test_logs fit_log_sequence... fit!(scm, dataset, verbosity = 1)
    for (key, eq) in equations(scm)
        @test eq.mach isa Machine
        @test isdefined(eq.mach, :data)
    end
    # Refit will not do anything
    nofit_log_sequence = (
        (:info, "Structural Equation corresponding to variable Ycont already fitted, skipping. Set `force=true` to force refit."),
        (:info, "Structural Equation corresponding to variable Ycat already fitted, skipping. Set `force=true` to force refit."),
        (:info, "Structural Equation corresponding to variable T₁ already fitted, skipping. Set `force=true` to force refit."),
        (:info, "Structural Equation corresponding to variable T₂ already fitted, skipping. Set `force=true` to force refit."),
    )
    @test_logs nofit_log_sequence... fit!(scm, dataset, verbosity = 1)
    # Force refit and set cache to false
    @test_logs fit_log_sequence... fit!(scm, dataset, verbosity = 1, force=true, cache=false)
    for (key, eq) in equations(scm)
        @test eq.mach isa Machine
        @test !isdefined(eq.mach, :data)
    end

    # Reset scm
    reset!(scm)
    for (key, eq) in equations(scm)
        @test eq.mach isa Nothing
    end
end

end
true