module TestEstimands

using Test
using TMLE
using OrderedCollections
using DataFrames

@testset "Test StatisticalCMCompositeEstimand" begin
    dataset = DataFrame(
        W  = [1, 2, 3, 4, 5, 6, 7, 8],
        T₁ = ["A", "B", "A", "B", "A", "B", "A", "B"],
        T₂ = [0, 0, 1, 1, 0, 0, 1, 1],
        T₃ = ["C", "C", "C", "C", "D", "D", "D", "D"],
        Y =  [1, 1, 1, 1, 1, 1, 1, 1]
    )
    # Counterfactual Mean
    Ψ = CM(
        outcome=:Y, 
        treatment_values=Dict("T₁" => "A", :T₂ => 1),
        treatment_confounders=(T₁=[:W], T₂=["W"])
    )
    indicator_fns = TMLE.indicator_fns(Ψ)
    @test indicator_fns == Dict(("A", 1) => 1.)
    indic_values = TMLE.indicator_values(indicator_fns, TMLE.selectcols(dataset, TMLE.treatments(Ψ)))
    @test indic_values == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ## Check the propensity score does not assume an independent decomposition
    propensity_score = TMLE.propensity_score(Ψ)
    @test propensity_score.components[1].outcome == :T₁
    @test propensity_score.components[1].parents == (:T₂, :W)
    # ATE
    Ψ = ATE(
        outcome=:Y, 
        treatment_values=(
            T₁=(case="A", control="B"), 
            T₂=(control=0, case=1)
        ),
        treatment_confounders=(T₁=[:W], T₂=[:W])
    )
    indicator_fns = TMLE.indicator_fns(Ψ)
    @test indicator_fns == Dict(
        ("A", 1) => 1.0,
        ("B", 0) => -1.0
    )
    indic_values = TMLE.indicator_values(indicator_fns, TMLE.selectcols(dataset, TMLE.treatments(Ψ)))
    @test indic_values == [0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0]
    # 2-points AIE
    Ψ = AIE(
        outcome=:Y, 
        treatment_values=Dict(
            :T₁ => (case="A", control="B"), 
            :T₂ => (case=1, control=0)
        ),
        treatment_confounders=Dict(:T₁ => [:W], :T₂ => [:W])
    )
    indicator_fns = TMLE.indicator_fns(Ψ)
    @test indicator_fns == Dict(
        ("A", 1) => 1.0,
        ("A", 0) => -1.0,
        ("B", 1) => -1.0,
        ("B", 0) => 1.0
    )
    indic_values = TMLE.indicator_values(indicator_fns, TMLE.selectcols(dataset, TMLE.treatments(Ψ)))
    @test indic_values == [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
    # 3-points AIE
    Ψ = AIE(
        outcome=:Y, 
        treatment_values=(
            T₁=(case="A", control="B"), 
            T₂=(case=1, control=0), 
            T₃=(control="D", case="C")
        ),
        treatment_confounders=(T₁=[:W], T₂=[:W], T₃=[:W])
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
    indic_values = TMLE.indicator_values(indicator_fns, TMLE.selectcols(dataset, TMLE.treatments(Ψ)))
    @test indic_values == [-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
end

@testset "Test standardization of StatisticalCMCompositeEstimand" begin
    # ATE
    Ψ₁ = ATE(
        outcome="Y",
        treatment_values = (T₁=(case=1, control=0), T₂=(control=1, case=0)),
        treatment_confounders = (
            T₂ = [:W₂, "W₁"],
            T₁ = ["W₀"]
            ),
        outcome_extra_covariates = ["Z", :A]
    )
    Ψ₂ = ATE(
        outcome=:Y,
        treatment_values = Dict("T₁"=>(case=1, control=0), "T₂"=>(case=0, control=1)),
        treatment_confounders = Dict(
            :T₁ => [:W₀],
            :T₂ => [:W₂, :W₁]
            ),
        outcome_extra_covariates = [:A, :Z]
    )
    @test Ψ₁ == Ψ₂
    @test Ψ₁.outcome == :Y
    @test Ψ₁.treatment_values == OrderedDict(
        :T₁ => (control=0, case=1), 
        :T₂ => (control=1, case=0)
    )
    @test Ψ₁.treatment_confounders == OrderedDict(:T₁ => (:W₀,), :T₂ => (:W₁, :W₂))
    @test Ψ₁.outcome_extra_covariates == (:A, :Z)
    
    # CM
    Ψ₁ = CM(
        outcome=:Y,
        treatment_values=(T₁=1, T₂=0),
        treatment_confounders = (T₂ = ["W₁", "W₂"], T₁ = [:W₀]),
    )
    Ψ₂ = CM(
        outcome=:Y,
        treatment_values=(T₂=0, T₁=1),
        treatment_confounders = (T₂ = [:W₂, "W₁"],T₁ = ["W₀"]),
    )
    @test Ψ₁ == Ψ₂
    @test Ψ₁.outcome == :Y
    @test Ψ₁.treatment_values == OrderedDict(:T₁ => 1, :T₂ => 0)
    @test Ψ₁.treatment_confounders == OrderedDict(:T₁ => (:W₀,), :T₂ => (:W₁, :W₂))
    @test Ψ₁.outcome_extra_covariates == ()
end

@testset "Test structs are concrete types" begin
    for type in Base.uniontypes(TMLE.StatisticalCMCompositeEstimand)
        @test isconcretetype(type)
    end
end

@testset "Test dictionary conversion" begin
    # Causal CM
    Ψ = CM(
        outcome=:y,
        treatment_values = (T₁=1, T₂="AC")
    )
    d = TMLE.to_dict(Ψ)
    @test d == Dict(
        :type             => "CM",
        :treatment_values => Dict(:T₁ => 1, :T₂ => "AC"),
        :outcome          => :y
    )
    Ψreconstructed = TMLE.from_dict!(d)
    @test Ψreconstructed == Ψ
    # Causal ATE
    Ψ = ATE(
        outcome=:y, 
        treatment_values=Dict(
            "T₁" => (case="A", control="B"), 
            "T₂" => (case=1, control=0), 
            "T₃" => (case="C", control="D")
        ),
    )
    d = TMLE.to_dict(Ψ)
    @test d == Dict(
        :type => "ATE",
        :treatment_values => Dict(
            :T₁ => Dict(:case => "A", :control => "B"),
            :T₂ => Dict(:case => 1, :control => 0), 
            :T₃ => Dict(:control => "D", :case => "C")
        ),
        :outcome => :y
    )
    Ψreconstructed = TMLE.from_dict!(d)
    @test Ψ == Ψ
    
    # Statistical CM
    Ψ = CM(
        outcome=:y,
        treatment_values = (T₁=1, T₂="AC"),
        treatment_confounders = (T₁=[:W₁₁, :W₁₂], T₂=[:W₁₂, :W₂₂])
    )
    d = TMLE.to_dict(Ψ)
    @test d == Dict(
        :outcome_extra_covariates => [],
        :type                     => "CM",
        :treatment_values         => Dict(:T₁=>1, :T₂=>"AC"),
        :outcome                  => :y,
        :treatment_confounders    => Dict(:T₁=>[:W₁₁, :W₁₂], :T₂=>[:W₁₂, :W₂₂])
    )
    Ψreconstructed = TMLE.from_dict!(d)
    @test Ψreconstructed == Ψ

    # Statistical AIE
    Ψ = AIE(
        outcome=:y,
        treatment_values = (T₁=1, T₂="AC"),
        treatment_confounders = (T₁=[:W₁₁, :W₁₂], T₂=[:W₁₂, :W₂₂]),
        outcome_extra_covariates=[:C]
    )
    d = TMLE.to_dict(Ψ)
    @test d == Dict(
        :outcome_extra_covariates => [:C],
        :type                     => "AIE",
        :treatment_values         => Dict(:T₁=>1, :T₂=>"AC"),
        :outcome                  => :y,
        :treatment_confounders    => Dict(:T₁=>[:W₁₁, :W₁₂], :T₂=>[:W₁₂, :W₂₂])
    )
    Ψreconstructed = TMLE.from_dict!(d)
    @test Ψreconstructed == Ψ
end

@testset "Test control_case_settings" begin
    treatments_unique_values = (T₁=(1, 0, 2),)
    @test TMLE.get_treatment_settings(ATE, treatments_unique_values) == OrderedDict(:T₁ => [(1, 0), (0, 2)],)
    @test TMLE.get_treatment_settings(AIE, treatments_unique_values) == OrderedDict(:T₁ => [(1, 0), (0, 2)],)
    @test TMLE.get_treatment_settings(CM, treatments_unique_values) == OrderedDict(:T₁ => (1, 0, 2), )
    treatments_unique_values = Dict(:T₁ => (1, 0, 2), :T₂ => ["AC", "CC"])
    @test TMLE.get_treatment_settings(ATE, treatments_unique_values) == OrderedDict(
        :T₁ => [(1, 0), (0, 2)], 
        :T₂ => [("AC", "CC")]
    )
    @test TMLE.get_treatment_settings(AIE, treatments_unique_values) == OrderedDict(
        :T₁ => [(1, 0), (0, 2)], 
        :T₂ => [("AC", "CC")]
    )
    @test TMLE.get_treatment_settings(CM, treatments_unique_values) == OrderedDict(
        :T₁ => (1, 0, 2), 
        :T₂ => ["AC", "CC"]
    )
end

@testset "Test unique_treatment_values" begin
    dataset = DataFrame(
        T₁ = ["AC", missing, "AC", "CC", "CC", "AA", "CC"],
        T₂ = [1, missing, 1, 2, 2, 3, 2]
    )
    # most frequent to least frequent and sorted by keys
    infered_values = TMLE.unique_treatment_values(dataset, (:T₂, :T₁))
    @test infered_values == OrderedDict(
        :T₁ => ["CC", "AC", "AA"],
        :T₂ => [2, 1, 3],
    )
    @test collect(keys(infered_values)) == [:T₁, :T₂]
end

@testset "factorialEstimand errors" begin
    dataset = DataFrame(
        T₁ = [0, 1, 2, missing], 
        T₂ = ["AC", "CC", missing, "AA"],
    )
    # Levels not in dataset
    msg = "Not all levels provided for treatment T₁ were found in the dataset: [3]"
    @test_throws ArgumentError(msg) factorialEstimand(
        CM, (T₁=(0, 3),), :Y₁, 
        dataset=dataset, 
        verbosity=0
    )
    # No dataset and no levels
    msg = "No dataset from which to infer treatment levels was provided. Either provide a `dataset` or a NamedTuple `treatments` e.g. (T=[0, 1, 2],)"
    @test_throws ArgumentError(msg) factorialEstimand(
        CM, (:T₁, :T₂), :Y₁, 
        verbosity=0
    )
end

@testset "Test factorial CM" begin
    dataset = DataFrame(
        T₁ = [0, 1, 2, missing], 
        T₂ = ["AC", "CC", missing, "AA"],
        W₁ = [1, 2, 3, 4],
        W₂ = [1, 2, 3, 4],
        C  = [1, 2, 3, 4],
        Y₁ = [1, 2, 3, 4],
        Y₂ = [1, 2, 3, 4]
    )
    jointCM = factorialEstimand(CM, [:T₁], :Y₁, dataset=dataset, verbosity=0)
    @test jointCM == TMLE.JointEstimand(
            TMLE.CausalCM(:Y₁, (T₁ = 0,)),
            TMLE.CausalCM(:Y₁, (T₁ = 1,)),
            TMLE.CausalCM(:Y₁, (T₁ = 2,))
    )

    jointCM = factorialEstimand(CM, (:T₁, :T₂), :Y₁, dataset=dataset, verbosity=0)
    @test jointCM == TMLE.JointEstimand(
            TMLE.CausalCM(:Y₁, (T₁ = 0, T₂ = "AC")),
            TMLE.CausalCM(:Y₁, (T₁ = 1, T₂ = "AC")),
            TMLE.CausalCM(:Y₁, (T₁ = 2, T₂ = "AC")),
            TMLE.CausalCM(:Y₁, (T₁ = 0, T₂ = "CC")),
            TMLE.CausalCM(:Y₁, (T₁ = 1, T₂ = "CC")),
            TMLE.CausalCM(:Y₁, (T₁ = 2, T₂ = "CC")),
            TMLE.CausalCM(:Y₁, (T₁ = 0, T₂ = "AA")),
            TMLE.CausalCM(:Y₁, (T₁ = 1, T₂ = "AA")),
            TMLE.CausalCM(:Y₁, (T₁ = 2, T₂ = "AA"))
    )
end

@testset "Test factorial ATE" begin
    dataset = DataFrame(
        T₁ = [0, 1, 2, missing], 
        T₂ = ["AC", "CC", missing, "AA"],
        W₁ = [1, 2, 3, 4],
        W₂ = [1, 2, 3, 4],
        C  = [1, 2, 3, 4],
        Y₁ = [1, 2, 3, 4],
        Y₂ = [1, 2, 3, 4]
    )
    # No confounders, 1 treatment, no extra covariate: 3 causal ATEs
    jointATE = factorialEstimand(ATE, [:T₁], :Y₁, dataset=dataset, verbosity=0)
    @test jointATE == JointEstimand(
        TMLE.CausalATE(:Y₁, (T₁ = (case = 1, control = 0),)),
        TMLE.CausalATE(:Y₁, (T₁ = (case = 2, control = 1),))
    )
    # 2 treatments
    jointATE = factorialEstimand(ATE, [:T₁, :T₂], :Y₁;
        dataset=dataset, 
        confounders=[:W₁, :W₂],
        outcome_extra_covariates=[:C],
        verbosity=0
    )
    ## 4 expected different treatment settings
    @test jointATE == JointEstimand(
            TMLE.StatisticalATE(
                outcome = :Y₁, 
                treatment_values = (T₁ = (case = 1, control = 0), T₂ = (case = "CC", control = "AC")),
                treatment_confounders = (:W₁, :W₂),
                outcome_extra_covariates=[:C]
            ),
            TMLE.StatisticalATE(
                outcome = :Y₁, 
                treatment_values = (T₁ = (case = 2, control = 1), T₂ = (case = "CC", control = "AC")),
                treatment_confounders = (:W₁, :W₂),
                outcome_extra_covariates=[:C]
            ),
            TMLE.StatisticalATE(
                outcome = :Y₁, 
                treatment_values = (T₁ = (case = 1, control = 0), T₂ = (case = "AA", control = "CC")),
                treatment_confounders = (:W₁, :W₂),
                outcome_extra_covariates=[:C]
            ),
            TMLE.StatisticalATE(
                outcome = :Y₁, 
                treatment_values = (T₁ = (case = 2, control = 1), T₂ = (case = "AA", control = "CC")),
                treatment_confounders = (:W₁, :W₂),
                outcome_extra_covariates=[:C]
            ),
    )
    # positivity constraint
    jointATE = factorialEstimand(ATE, [:T₁, :T₂], :Y₁;
        dataset=dataset,
        confounders=[:W₁, :W₂],
        outcome_extra_covariates=[:C],
        positivity_constraint=0.1,
        verbosity=0
    )
    @test length(jointATE.args) == 1
    # Given treatment levels
    ate = factorialEstimand(ATE, (T₁ = [0, 1], T₂ = ["AA", "AC", "CC"]), :Y₁;
        dataset=dataset, 
        confounders=[:W₁, :W₂],
        outcome_extra_covariates=[:C],
        verbosity=0
    )
    @test ate == JointEstimand(
        TMLE.StatisticalATE(
            outcome = :Y₁, 
            treatment_values = (T₁ = (case = 1, control = 0), T₂ = (case = "AC", control = "AA")),
            treatment_confounders = (:W₁, :W₂),
            outcome_extra_covariates=[:C]
        ),
        TMLE.StatisticalATE(
            outcome = :Y₁, 
            treatment_values = (T₁ = (case = 1, control = 0), T₂ = (case = "CC", control = "AC")),
            treatment_confounders = (:W₁, :W₂),
            outcome_extra_covariates=[:C]
        )
    )
end

@testset "Test factorial AIE" begin
    dataset = DataFrame(
        T₁ = [0, 1, 2, missing], 
        T₂ = ["AC", "CC", missing, "AA"],
        W₁ = [1, 2, 3, 4],
        W₂ = [1, 2, 3, 4],
        C  = [1, 2, 3, 4],
        Y₁ = [1, 2, 3, 4],
        Y₂ = [1, 2, 3, 4]
    )
    # From dataset
    jointAIE = factorialEstimand(AIE, [:T₁, :T₂], :Y₁;
        dataset=dataset, 
        confounders=[:W₁], 
        outcome_extra_covariates=[:C],
        verbosity=0
    )
    @test jointAIE == JointEstimand(
            TMLE.StatisticalAIE(
                outcome = :Y₁, 
                treatment_values = (T₁ = (case = 1, control = 0), T₂ = (case = "CC", control = "AC")), 
                treatment_confounders = (:W₁,), 
                outcome_extra_covariates = (:C,)
            ),
            TMLE.StatisticalAIE(
                outcome = :Y₁, 
                treatment_values = (T₁ = (case = 2, control = 1), T₂ = (case = "CC", control = "AC")), 
                treatment_confounders = (:W₁,), 
                outcome_extra_covariates = (:C,)
            ),
            TMLE.StatisticalAIE(
                outcome = :Y₁, 
                treatment_values = (T₁ = (case = 1, control = 0), T₂ = (case = "AA", control = "CC")), 
                treatment_confounders = (:W₁,), 
                outcome_extra_covariates = (:C,)
            ),
            TMLE.StatisticalAIE(
                outcome = :Y₁, 
                treatment_values = (T₁ = (case = 2, control = 1), T₂ = (case = "AA", control = "CC")), 
                treatment_confounders = (:W₁,), 
                outcome_extra_covariates = (:C,)
            )
    )
    # From unique values
    jointAIE_from_dict = factorialEstimand(AIE, Dict(:T₁ => (0, 1), :T₂ => (0, 1, 2), :T₃ => (0, 1, 2)), :Y₁, verbosity=0)
    jointAIE_from_nt = factorialEstimand(AIE, (T₁ = (0, 1), T₂ = (0, 1, 2), T₃ = (0, 1, 2)), :Y₁, verbosity=0)
    @test jointAIE_from_dict == jointAIE_from_nt == JointEstimand(
            TMLE.CausalAIE(
                outcome = :Y₁, 
                treatment_values = (T₁ = (control = 0, case = 1), T₂ = (control = 0, case = 1), T₃ = (control = 0, case = 1))
            ),
            TMLE.CausalAIE(
                outcome = :Y₁, 
                treatment_values = (T₁ = (control = 0, case = 1), T₂ = (control = 1, case = 2), T₃ = (control = 0, case = 1))
            ),
            TMLE.CausalAIE(
                outcome = :Y₁, 
                treatment_values = (T₁ = (control = 0, case = 1), T₂ = (control = 0, case = 1), T₃ = (control = 1, case = 2))
            ),
            TMLE.CausalAIE(
                outcome = :Y₁, 
                treatment_values = (T₁ = (control = 0, case = 1), T₂ = (control = 1, case = 2), T₃ = (control = 1, case = 2))
            )
    )

    # positivity constraint
    @test_throws ArgumentError("No component passed the positivity constraint.") factorialEstimand(AIE, [:T₁, :T₂], :Y₁;
        dataset=dataset,  
        confounders=[:W₁], 
        outcome_extra_covariates=[:C],
        positivity_constraint=0.1,
        verbosity=0
    )
end

@testset "Test factorialEstimands" begin
    dataset = DataFrame(
        T₁ = [0, 1, 2, missing], 
        T₂ = ["AC", "CC", missing, "AA"],
        W₁ = [1, 2, 3, 4],
        W₂ = [1, 2, 3, 4],
        C  = [1, 2, 3, 4],
        Y₁ = [1, 2, 3, 4],
        Y₂ = [1, 2, 3, 4]
    )
    factorial_ates = factorialEstimands(ATE, [:T₁, :T₂], [:Y₁, :Y₂], 
        dataset=dataset,
        confounders=[:W₁, :W₂], 
        outcome_extra_covariates=[:C],
        positivity_constraint=0.1,
        verbosity=0
    )
    @test length(factorial_ates) == 2
    # Nothing passes the threshold
    @test_throws ArgumentError("No component passed the positivity constraint.") factorialEstimands(ATE, [:T₁, :T₂], [:Y₁, :Y₂], 
        dataset=dataset,
        confounders=[:W₁, :W₂], 
        outcome_extra_covariates=[:C],
        positivity_constraint=0.3,
        verbosity=0
    )
end
end

true