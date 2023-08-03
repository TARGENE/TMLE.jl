module TestEstimands

using Test
using TMLE
using CategoricalArrays
using Random
using StableRNGs
using MLJBase
using MLJGLMInterface

@testset "Test methods" begin
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
    Ψ = CM(
        scm=scm,
        treatment=(T=1,),
        outcome =:Y
    )
    
    @test TMLE.treatments(Ψ) == [:T]
    @test TMLE.outcome(Ψ) == :Y
    @test TMLE.confounders(Ψ) == (T=[:W₁, :W₂],)
    identified, reasons = isidentified(Ψ, dataset)
    @test identified === true
    @test reasons == []
    ηs = TMLE.equations_to_fit(Ψ)
    @test ηs === (scm.Y, scm.T)
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
    @test TMLE.confounders(Ψ) == (T₁=[:W₁₁, :W₁₂], T₂=[:W₂₁])
    identified, reasons = isidentified(Ψ, dataset)
    @test identified === false
    expected_reasons = [
        "Outcome variable: Z is not in the dataset.",
        "Treatment variable: T₁ is not in the dataset.",
        "Treatment variable: T₂ is not in the dataset.",
        "Confounding variable: W₁₂ is not in the dataset.",
        "Confounding variable: W₂₁ is not in the dataset.",
        "Confounding variable: W₁₁ is not in the dataset."
    ]
    @test reasons == expected_reasons
    ηs = TMLE.equations_to_fit(Ψ)
    @test ηs === (scm.Z, scm.T₁, scm.T₂)
    # IATE
    Ψ = IATE(
        scm=scm,
        treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
        outcome =:Z,
    )
    @test TMLE.treatments(Ψ) == [:T₁, :T₂]
    @test TMLE.outcome(Ψ) == :Z
    @test TMLE.confounders(Ψ) == (T₁=[:W₁₁, :W₁₂], T₂=[:W₂₁])
    identified, reasons = isidentified(Ψ, dataset)
    @test identified === false
    @test reasons == expected_reasons
    ηs = TMLE.equations_to_fit(Ψ)
    @test ηs === (scm.Z, scm.T₁, scm.T₂)
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
    fit!(Ψ, dataset, verbosity=0)
    @test scm.Ycat.mach isa Machine
    @test scm.T₁.mach isa Machine
    @test scm.Ycont.mach isa Nothing
    @test scm.T₂.mach isa Nothing

    # Fits ATE required equations that haven't been fitted yet.
    Ψ = ATE(
        scm=scm,
        treatment=(T₁=(case=1, control=0), T₂=(case=1, control=0)),
        outcome =:Ycat,
    )
    log_sequence = (
        (:info, "Structural Equation corresponding to variable Ycat already fitted, skipping. Set `force=true` to force refit."),
        (:info, "Structural Equation corresponding to variable T₁ already fitted, skipping. Set `force=true` to force refit."),
        (:info, "Fitting Structural Equation corresponding to variable T₂."),
    )
    @test_logs log_sequence... fit!(Ψ, dataset, verbosity=1)
    @test scm.Ycat.mach isa Machine
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
        (:info, "Structural Equation corresponding to variable T₁ already fitted, skipping. Set `force=true` to force refit."),
        (:info, "Structural Equation corresponding to variable T₂ already fitted, skipping. Set `force=true` to force refit."),
    )
    @test_logs log_sequence... fit!(Ψ, dataset, verbosity=1)
    @test scm.Ycat.mach isa Machine
    @test scm.T₁.mach isa Machine
    @test scm.T₂.mach isa Machine
    @test scm.Ycont.mach isa Machine
end

@testset "Test optimize_ordering" begin
    estimands = [
        ATE(
            outcome=:Y, 
            treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
            confounders=[:W₁, :W₂]
        ),
        ATE(
            outcome=:Y, 
            treatment=(T₁=(case=1, control=0),),
            confounders=[:W₁]
        ),
        ATE(
            outcome=:Y₂, 
            treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
            confounders=[:W₁, :W₂],
        ),
        CM(
            outcome=:Y, 
            treatment=(T₁=0,),
            confounders=[:W₁],
            covariates=[:C₁]
        ),
        IATE(
            outcome=:Y, 
            treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
            confounders=[:W₁, :W₂]
        ),
        CM(
            outcome=:Y, 
            treatment=(T₁=0,),
            confounders=[:W₁]
        ),
        ATE(
            outcome=:Y₂, 
            treatment=(T₁=(case=1, control=0),),
            confounders=[:W₁],
            covariates=[:C₁]
        ),
        ATE(
            outcome=:Y₂, 
            treatment=(T₁=(case=0, control=1), T₂=(case="AC", control="CC")),
            confounders=[:W₁, :W₂],
            covariates=[:C₂]
        ),
    ]
    # Non mutating function
    ordered_estimands = optimize_ordering(estimands)
    expected_ordering = [
        ATE(:Y, (T₁ = (case = 1, control = 0),), [:W₁], Symbol[]),
        CM(:Y, (T₁ = 0,), [:W₁], Symbol[]),
        CM(:Y, (T₁ = 0,), [:W₁], [:C₁]),
        ATE(:Y₂, (T₁ = (case = 1, control = 0),), [:W₁], [:C₁]),
        ATE(:Y, (T₁ = (case = 1, control = 0), T₂ = (case = "AC", control = "CC")), [:W₁, :W₂], Symbol[]),
        IATE(:Y, (T₁ = (case = 1, control = 0), T₂ = (case = "AC", control = "CC")), [:W₁, :W₂], Symbol[]),
        ATE(:Y₂, (T₁ = (case = 1, control = 0), T₂ = (case = "AC", control = "CC")), [:W₁, :W₂], Symbol[]),
        ATE(:Y₂, (T₁ = (case = 0, control = 1), T₂ = (case = "AC", control = "CC")), [:W₁, :W₂], [:C₂])
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
    @test isconcretetype(TMLE.ALEstimate{Float64})
    @test isconcretetype(TMLE.TMLEResult{ATE, Float64})
    @test isconcretetype(TMLE.TMLEResult{IATE, Float64})
    @test isconcretetype(TMLE.TMLEResult{CM, Float64})
end

end

true