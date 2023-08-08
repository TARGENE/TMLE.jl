module TestEstimands

using Test
using TMLE
using CategoricalArrays
using Random
using StableRNGs
using MLJBase
using MLJGLMInterface

@testset "Test various methods" begin
    n = 5
    dataset = (
        Y = rand(n),
        T = categorical([0, 1, 1, 0, 0]),
        W₁ = rand(n),
        W₂ = rand(n)
    )
    # CM
    scm = StaticConfoundedModel(
        :Y, :T, [:W₁, :W₂], 
        covariates=:C₁
    )
    # Parameter validation
    @test_throws TMLE.VariableNotAChildInSCMError("UndefinedOutcome") CM(
        scm,
        treatment=(T=1,),
        outcome=:UndefinedOutcome
    )
    @test_throws TMLE.VariableNotAChildInSCMError("UndefinedTreatment") CM(
        scm,
        treatment=(UndefinedTreatment=1,),
        outcome=:Y
    )
    @test_throws TMLE.TreatmentMustBeInOutcomeParentsError("Y") CM(
        scm,
        treatment=(Y=1,),
        outcome=:T
    )

    Ψ = CM(
        scm=scm,
        treatment=(T=1,),
        outcome =:Y
    )
    
    @test TMLE.treatments(Ψ) == [:T]
    @test TMLE.treatments(dataset, Ψ) == (T=dataset.T,)
    @test TMLE.outcome(Ψ) == :Y
    
    # ATE
    scm = SCM(
        SE(:Z, [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁]),
        SE(:T₁, [:W₁₁, :W₁₂]),
        SE(:T₂, [:W₂₁]),
    )
    Ψ = ATE(
        scm=scm,
        treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
        outcome =:Z,
    )
    @test TMLE.treatments(Ψ) == [:T₁, :T₂]
    @test TMLE.outcome(Ψ) == :Z

    # IATE
    Ψ = IATE(
        scm=scm,
        treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
        outcome =:Z,
    )
    @test TMLE.treatments(Ψ) == [:T₁, :T₂]
    @test TMLE.outcome(Ψ) == :Z
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

    # Fits CM required equations
    Ψ = CM(
        scm=scm,
        treatment=(T₁=1,),
        outcome=:Ycat
    )
    fit!(Ψ, dataset; adjustment_method=BackdoorAdjustment([:C]), verbosity=0)
    @test scm.Ycat.mach isa Machine
    @test keys(scm.Ycat.mach.data[1]) == (:T₁, :W₁₁, :W₁₂, :C)
    @test scm.T₁.mach isa Machine
    @test keys(scm.T₁.mach.data[1]) == (:W₁₁, :W₁₂)
    @test scm.Ycont.mach isa Nothing
    @test scm.T₂.mach isa Nothing

    # Fits ATE required equations that haven't been fitted yet.
    Ψ = ATE(
        scm=scm,
        treatment=(T₁=(case=1, control=0), T₂=(case=1, control=0)),
        outcome =:Ycat,
    )
    log_sequence = (
        (:info, "Fitting Structural Equation corresponding to variable T₂."),
        (:info, "Fitting Structural Equation corresponding to variable Ycat.")
    )
    @test_logs log_sequence... fit!(Ψ, dataset, verbosity=1)
    @test scm.Ycat.mach isa Machine
    keys(scm.Ycat.mach.data[1])
    @test scm.T₁.mach isa Machine
    @test scm.T₂.mach isa Machine
    @test scm.Ycont.mach isa Nothing
    
    # Fits IATE required equations that haven't been fitted yet.
    Ψ = IATE(
        scm=scm,
        treatment=(T₁=(case=1, control=0), T₂=(case=1, control=0)),
        outcome =:Ycont,
    )
    log_sequence = (
        (:info, "Fitting Structural Equation corresponding to variable Ycont."),
    )
    @test_logs log_sequence... fit!(Ψ, dataset, verbosity=1)
    @test scm.Ycat.mach isa Machine
    @test scm.T₁.mach isa Machine
    @test scm.T₂.mach isa Machine
    @test scm.Ycont.mach isa Machine

    # Change a model
    scm.Ycont.model = TreatmentTransformer() |> LinearRegressor(fit_intercept=false)
    log_sequence = (
        (:info, "Fitting Structural Equation corresponding to variable Ycont."),
    )
    @test_logs log_sequence... fit!(Ψ, dataset, verbosity=1)
end

@testset "Test optimize_ordering" begin
    rng = StableRNG(123)
    scm = SCM(
        SE(:T₁, [:W₁, :W₂]),
        SE(:T₂, [:W₁, :W₂, :W₃]),
        SE(:Y₁, [:T₁, :T₂, :W₁, :W₂, :C₁]),
        SE(:Y₂, [:T₂, :W₁, :W₂, :W₃]),
    )
    estimands = [
        ATE(
            scm=scm,
            outcome=:Y₁, 
            treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
        ),
        IATE(
            scm=scm,
            outcome=:Y₁, 
            treatment=(T₁=(case=1, control=0), T₂=(case="AA", control="CC")),
        ),
        ATE(
            scm=scm,
            outcome=:Y₁, 
            treatment=(T₁=(case=1, control=0),),
        ),
        ATE(
            scm=scm,
            outcome=:Y₂, 
            treatment=(T₂=(case="AC", control="CC"),),
        ),
        CM(
            scm=scm,
            outcome=:Y₂, 
            treatment=(T₂="AC",),
        ),
        IATE(
            scm=scm,
            outcome=:Y₁, 
            treatment=(T₁=(case=0, control=1), T₂=(case="AC", control="CC"),),
        ),
        CM(
            scm=scm,
            outcome=:Y₂, 
            treatment=(T₂="CC",),
        ),
        ATE(
            scm=scm,
            outcome=:Y₂, 
            treatment=(T₂=(case="AA", control="CC"),),
        ),
        ATE(
            scm=scm,
            outcome=:Y₂, 
            treatment=(T₂=(case="AA", control="AC"),),
        ),
    ]
    # Test param_key
    @test TMLE.param_key(estimands[1]) == ("T₁_T₂", "Y₁")
    @test TMLE.param_key(estimands[end]) == ("T₂", "Y₂")
    # Non mutating function
    estimands = shuffle(rng, estimands)
    ordered_estimands = optimize_ordering(estimands)
    expected_ordering = [
        # Y₁
        ATE(scm=scm, outcome=:Y₁, treatment=(T₁ = (case = 1, control = 0),)),
        ATE(scm=scm, outcome=:Y₁, treatment=(T₁ = (case = 1, control = 0),T₂ = (case = "AC", control = "CC"))),
        IATE(scm=scm, outcome=:Y₁, treatment=(T₁ = (case = 0, control = 1), T₂ = (case = "AC", control = "CC"),)),
        IATE(scm=scm, outcome=:Y₁, treatment=(T₁ = (case = 1, control = 0),T₂ = (case = "AA", control = "CC"))),
        # Y₂
        ATE(scm=scm, outcome=:Y₂, treatment=(T₂ = (case = "AA", control = "AC"),)),
        CM(scm=scm, outcome=:Y₂, treatment=(T₂ = "CC",)),
        ATE(scm=scm, outcome=:Y₂, treatment=(T₂ = (case = "AA", control = "CC"),)),
        CM(scm=scm, outcome=:Y₂, treatment=(T₂ = "AC",)),
        ATE(scm=scm, outcome=:Y₂, treatment=(T₂ = (case = "AC", control = "CC"),)),
    ]
    @test ordered_estimands == expected_ordering
    # Mutating function
    optimize_ordering!(estimands)
    @test estimands == ordered_estimands
end


@testset "Test structs are concrete types" begin
    @test isconcretetype(ATE)
    @test isconcretetype(IATE)
    @test isconcretetype(CM)
    @test isconcretetype(TMLE.TMLEstimate{Float64})
    @test isconcretetype(TMLE.OSEstimate{Float64})
    @test isconcretetype(TMLE.TMLEResult{ATE, Float64})
    @test isconcretetype(TMLE.TMLEResult{IATE, Float64})
    @test isconcretetype(TMLE.TMLEResult{CM, Float64})
end

end

true