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
        scm,
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
        scm,
        treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
        outcome =:Z,
    )
    @test TMLE.treatments(Ψ) == [:T₁, :T₂]
    @test TMLE.outcome(Ψ) == :Z

    # IATE
    Ψ = IATE(
        scm,
        treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
        outcome =:Z,
    )
    @test TMLE.treatments(Ψ) == [:T₁, :T₂]
    @test TMLE.outcome(Ψ) == :Z
end

@testset "Test constructors" begin
    Ψ = CM(outcome=:Y, treatment=(T=1,), confounders=[:W])
    @test parents(Ψ.scm, :Y) == Set([:T, :W])

    Ψ = ATE(outcome=:Y, treatment=(T₁=(case=1, control=0), T₂=(case=1, control=0)), confounders=:W)
    @test parents(Ψ.scm, :Y) == Set([:T₁, :T₂, :W])

    Ψ = IATE(outcome=:Y, treatment=(T₁=(case=1, control=0), T₂=(case=1, control=0)), confounders=:W)
    @test parents(Ψ.scm, :Y) == Set([:T₁, :T₂, :W])
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
        SE(:Ycat, [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁, :W₂₂, :C]),
        SE(:Ycont, [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁, :W₂₂, :C]),
        SE(:T₁, [:W₁₁, :W₁₂]),
        SE(:T₂, [:W₂₁, :W₂₂]),
    )
    set_conditional_distribution!(scm, :Ycat, with_encoder(LinearBinaryClassifier()))
    set_conditional_distribution!(scm, :Ycont, with_encoder(LinearRegressor()))
    set_conditional_distribution!(scm, :T₁, LinearBinaryClassifier())
    set_conditional_distribution!(scm, :T₂, LinearBinaryClassifier())
    # Fits CM required factors
    Ψ = CM(
        scm,
        treatment=(T₁=1,),
        outcome=:Ycat
    )
    log_sequence = (
        (:info, TMLE.UsingNaturalDistributionLog(scm, :Ycat, Set([:W₁₂, :W₁₁, :T₁, :C]))),
        (:info, "Fitting Conditional Distribution Factor: Ycat | W₁₂, W₁₁, T₁, C"),
        (:info, "Fitting Conditional Distribution Factor: T₁ | W₁₂, W₁₁"),
    )
    @test_logs log_sequence... fit!(Ψ, dataset; adjustment_method=BackdoorAdjustment([:C]), verbosity=1)
    outcome_dist = get_conditional_distribution(scm, :Ycat, Set([:W₁₂, :W₁₁, :T₁, :C]))
    @test keys(fitted_params(outcome_dist.machine)) == (:linear_binary_classifier, :treatment_transformer)
    @test keys(outcome_dist.machine.data[1]) == (:W₁₂, :W₁₁, :T₁, :C)
    treatment_dist = get_conditional_distribution(scm, :T₁, Set([:W₁₁, :W₁₂]))
    @test fitted_params(treatment_dist.machine).features == [:W₁₂, :W₁₁]
    @test keys(treatment_dist.machine.data[1]) == (:W₁₂, :W₁₁)
    # The natural outcome distribution has not been fitted for instance
    natural_outcome_dist = get_conditional_distribution(scm, :Ycat, Set([:T₂, :W₁₂, :W₂₂, :W₂₁, :W₁₁, :T₁, :C]))
    @test !isdefined(natural_outcome_dist, :machine)

    # Fits ATE required equations that haven't been fitted yet.
    Ψ = ATE(
        scm,
        treatment=(T₁=(case=1, control=0), T₂=(case=1, control=0)),
        outcome =:Ycat,
    )
    conditioning_set = Set([:W₁₂, :W₂₂, :W₂₁, :W₁₁, :T₁, :T₂])
    log_sequence = (
        (:info, TMLE.UsingNaturalDistributionLog(scm, :Ycat, conditioning_set)),
        (:info, "Fitting Conditional Distribution Factor: Ycat | W₁₂, W₂₂, W₂₁, W₁₁, T₁, T₂"),
        (:info, "Reusing or Updating Conditional Distribution Factor: T₁ | W₁₂, W₁₁"),
        (:info, "Fitting Conditional Distribution Factor: T₂ | W₂₂, W₂₁"),
    )
    @test_logs log_sequence... fit!(Ψ, dataset, verbosity=1)

    outcome_dist = get_conditional_distribution(scm, :Ycat, conditioning_set)
    @test keys(fitted_params(outcome_dist.machine)) == (:linear_binary_classifier, :treatment_transformer)
    @test Set(keys(outcome_dist.machine.data[1])) == conditioning_set
    treatment_dist = get_conditional_distribution(scm, :T₂, Set([:W₂₂, :W₂₁]))
    @test fitted_params(treatment_dist.machine).features == [:W₂₂, :W₂₁]
    @test keys(treatment_dist.machine.data[1]) == (:W₂₂, :W₂₁)
    
    # Fits IATE required equations that haven't been fitted yet.
    Ψ = IATE(
        scm,
        treatment=(T₁=(case=1, control=0), T₂=(case=1, control=0)),
        outcome =:Ycont,
    )
    conditioning_set = Set([:W₁₂, :W₂₂, :W₂₁, :W₁₁, :T₁, :T₂])
    log_sequence = (
        (:info, TMLE.UsingNaturalDistributionLog(scm, :Ycont, conditioning_set)),
        (:info, "Fitting Conditional Distribution Factor: Ycont | W₁₂, W₂₂, W₂₁, W₁₁, T₁, T₂"),
        (:info, "Reusing or Updating Conditional Distribution Factor: T₁ | W₁₂, W₁₁"),
        (:info, "Reusing or Updating Conditional Distribution Factor: T₂ | W₂₂, W₂₁"),
    )
    @test_logs log_sequence... fit!(Ψ, dataset, verbosity=1)
    outcome_dist = get_conditional_distribution(scm, :Ycont, conditioning_set)
    @test keys(fitted_params(outcome_dist.machine)) == (:linear_regressor, :treatment_transformer)
    @test Set(keys(outcome_dist.machine.data[1])) == conditioning_set

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
            scm,
            outcome=:Y₁, 
            treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
        ),
        IATE(
            scm,
            outcome=:Y₁, 
            treatment=(T₁=(case=1, control=0), T₂=(case="AA", control="CC")),
        ),
        ATE(
            scm,
            outcome=:Y₁, 
            treatment=(T₁=(case=1, control=0),),
        ),
        ATE(
            scm,
            outcome=:Y₂, 
            treatment=(T₂=(case="AC", control="CC"),),
        ),
        CM(
            scm,
            outcome=:Y₂, 
            treatment=(T₂="AC",),
        ),
        IATE(
            scm,
            outcome=:Y₁, 
            treatment=(T₁=(case=0, control=1), T₂=(case="AC", control="CC"),),
        ),
        CM(
            scm,
            outcome=:Y₂, 
            treatment=(T₂="CC",),
        ),
        ATE(
            scm,
            outcome=:Y₂, 
            treatment=(T₂=(case="AA", control="CC"),),
        ),
        ATE(
            scm,
            outcome=:Y₂, 
            treatment=(T₂=(case="AA", control="AC"),),
        ),
    ]
    # Test estimand_key
    @test TMLE.estimand_key(estimands[1]) == ("T₁_T₂", "Y₁")
    @test TMLE.estimand_key(estimands[end]) == ("T₂", "Y₂")
    # Non mutating function
    estimands = shuffle(rng, estimands)
    ordered_estimands = optimize_ordering(estimands)
    expected_ordering = [
        # Y₁
        ATE(scm, outcome=:Y₁, treatment=(T₁ = (case = 1, control = 0),)),
        ATE(scm, outcome=:Y₁, treatment=(T₁ = (case = 1, control = 0),T₂ = (case = "AC", control = "CC"))),
        IATE(scm, outcome=:Y₁, treatment=(T₁ = (case = 0, control = 1), T₂ = (case = "AC", control = "CC"),)),
        IATE(scm, outcome=:Y₁, treatment=(T₁ = (case = 1, control = 0),T₂ = (case = "AA", control = "CC"))),
        # Y₂
        ATE(scm, outcome=:Y₂, treatment=(T₂ = (case = "AA", control = "AC"),)),
        CM(scm, outcome=:Y₂, treatment=(T₂ = "CC",)),
        ATE(scm, outcome=:Y₂, treatment=(T₂ = (case = "AA", control = "CC"),)),
        CM(scm, outcome=:Y₂, treatment=(T₂ = "AC",)),
        ATE(scm, outcome=:Y₂, treatment=(T₂ = (case = "AC", control = "CC"),)),
    ]
    @test ordered_estimands == expected_ordering
    # Mutating function
    optimize_ordering!(estimands)
    @test estimands == ordered_estimands
end

@testset "Test indicator_fns & indicator_values" begin
    scm = StaticConfoundedModel(
        [:Y],
        [:T₁, :T₂, :T₃],
        [:W]
    )
    dataset = (
        W  = [1, 2, 3, 4, 5, 6, 7, 8],
        T₁ = ["A", "B", "A", "B", "A", "B", "A", "B"],
        T₂ = [0, 0, 1, 1, 0, 0, 1, 1],
        T₃ = ["C", "C", "C", "C", "D", "D", "D", "D"],
        Y =  [1, 1, 1, 1, 1, 1, 1, 1]
    )
    # Counterfactual Mean
    Ψ = CM(
        scm,
        outcome=:Y, 
        treatment=(T₁="A", T₂=1),
    )
    indicator_fns = TMLE.indicator_fns(Ψ)
    @test indicator_fns == Dict(("A", 1) => 1.)
    indic_values = TMLE.indicator_values(indicator_fns, TMLE.treatments(dataset, Ψ))
    @test indic_values == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # ATE
    Ψ = ATE(
        scm,
        outcome=:Y, 
        treatment=(T₁=(case="A", control="B"), T₂=(control=0, case=1)),
    )
    indicator_fns = TMLE.indicator_fns(Ψ)
    @test indicator_fns == Dict(
        ("A", 1) => 1.0,
        ("B", 0) => -1.0
    )
    indic_values = TMLE.indicator_values(indicator_fns, TMLE.treatments(dataset, Ψ))
    @test indic_values == [0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0]
    # 2-points IATE
    Ψ = IATE(
        scm,
        outcome=:Y, 
        treatment=(T₁=(case="A", control="B"), T₂=(case=1, control=0)),
    )
    indicator_fns = TMLE.indicator_fns(Ψ)
    @test indicator_fns == Dict(
        ("A", 1) => 1.0,
        ("A", 0) => -1.0,
        ("B", 1) => -1.0,
        ("B", 0) => 1.0
    )
    indic_values = TMLE.indicator_values(indicator_fns, TMLE.treatments(dataset, Ψ))
    @test indic_values == [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
    # 3-points IATE
    Ψ = IATE(
        scm,
        outcome=:Y, 
        treatment=(T₁=(case="A", control="B"), T₂=(case=1, control=0), T₃=(control="D", case="C")),
    )
    indicator_fns = TMLE.indicator_fns(Ψ)
    @test indicator_fns == Dict(
        ("A", 1, "D") => -1.0,
        ("A", 1, "C") => 1.0,
        ("B", 0, "D") => -1.0,
        ("B", 0, "C") => 1.0,
        ("B", 1, "C") => -1.0,
        ("A", 0, "D") => 1.0,
        ("B", 1, "D") => 1.0,
        ("A", 0, "C") => -1.0
    )
    indic_values = TMLE.indicator_values(indicator_fns, TMLE.treatments(dataset, Ψ))
    @test indic_values == [-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
end

@testset "Test structs are concrete types" begin
    @test isconcretetype(ATE)
    @test isconcretetype(IATE)
    @test isconcretetype(CM)
    @test isconcretetype(TMLE.TMLEstimate{Float64})
    @test isconcretetype(TMLE.OSEstimate{Float64})
end

end

true