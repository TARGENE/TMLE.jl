module TestUtils

using Test
using TMLE
using MLJBase
using CategoricalArrays
using MLJLinearModels
using MLJModels
using DataFrames

@testset "Test expected_value" begin
    n = 100
    X = MLJBase.table(rand(n, 3))

    # Probabilistic Classifier
    y = categorical(rand([0, 1], n))
    mach = machine(ConstantClassifier(), X, y)
    fit!(mach; verbosity=0)
    proba = mach.fitresult[2][2]
    ŷ = MLJBase.predict(mach)
    expectation = TMLE.expected_value(ŷ)
    @test expectation == fill(proba, n)

    # Probabilistic Regressor
    y = rand(n)
    mach = machine(ConstantRegressor(), X, y)
    fit!(mach; verbosity=0)
    ŷ = MLJBase.predict(mach)
    expectation = TMLE.expected_value(ŷ)
    @test expectation ≈ fill(mean(y), n) atol=1e-10

    # Deterministic Regressor
    mach = machine(LinearRegressor(), X, y)
    fit!(mach; verbosity=0)
    ŷ = MLJBase.predict(mach)
    expectation = TMLE.expected_value(ŷ)
    @test expectation == ŷ
end

@testset "Test counterfactualTreatment" begin
    vals = (true, "a")
    T = DataFrame(
        T₁ = categorical([true, false, false], ordered=true),
        T₂ = categorical(["a", "a", "c"])
    )
    cfT = TMLE.counterfactualTreatment(vals, T)
    @test cfT == DataFrame(
        T₁ = categorical([true, true, true]),
        T₂ = categorical(["a", "a", "a"])
    )
    @test isordered(cfT.T₁)
    @test !isordered(cfT.T₂)
end

@testset "Test positivity_constraint & get_frequency_table" begin
    # get_frequency_table
    ## When no positivity constraint is provided then get_frequency_table returns nothing
    @test TMLE.get_frequency_table(nothing, nothing, [1, 2]) === nothing
    @test TMLE.get_frequency_table(nothing, "toto", [1, 2]) === nothing
    ## An error is thrown if no dataset is provided but a positivity constraint is given
    @test_throws ArgumentError("A dataset should be provided to enforce a positivity constraint.") TMLE.get_frequency_table(0.1, nothing, [1, 2])
    ## when both positivity constraint and datasets are provided
    dataset = DataFrame(
        A = [1, 1, 0, 1, 0, 2, 2, 1],
        B = ["AC", "CC", "AA", "AA", "AA", "AA", "AA", "AA"]
    ) 
    ### One variable
    frequency_table = TMLE.get_frequency_table(0.1, dataset, [:A])
    @test frequency_table[(0,)] == 0.25
    @test frequency_table[(1,)] == 0.5
    @test frequency_table[(2,)] == 0.25

    Ψ = CM(
        outcome = :toto, 
        treatment_values = (A=1,), 
        treatment_confounders = (A=[],)
    )
    @test TMLE.joint_levels(Ψ) == ((1,),)
    @test TMLE.satisfies_positivity(Ψ, frequency_table, positivity_constraint=0.4) == true
    @test TMLE.satisfies_positivity(Ψ, frequency_table, positivity_constraint=0.6) == false

    Ψ = ATE(
        outcome = :toto, 
        treatment_values= (A = (case=1, control=0),), 
        treatment_confounders = (A=[],)
    )
    @test collect(TMLE.joint_levels(Ψ)) == [(0,), (1,)]
    @test TMLE.satisfies_positivity(Ψ, frequency_table, positivity_constraint=0.2) == true
    @test TMLE.satisfies_positivity(Ψ, frequency_table, positivity_constraint=0.3) == false

    ## Two variables
    ### Treatments are sorted: [:B, :A] -> [:A, :B]
    frequency_table = TMLE.get_frequency_table(dataset, [:B, :A])
    @test frequency_table[(1, "CC")] == 0.125
    @test frequency_table[(1, "AA")] == 0.25
    @test frequency_table[(0, "AA")] == 0.25
    @test frequency_table[(1, "AC")] == 0.125
    @test frequency_table[(2, "AA")] == 0.25

    Ψ = CM(
        outcome = :toto, 
        treatment_values = (B = "CC", A = 1), 
        treatment_confounders = (B = [], A = [])
    )
    @test TMLE.joint_levels(Ψ) == ((1, "CC"),)
    @test TMLE.satisfies_positivity(Ψ, frequency_table, positivity_constraint=0.1) == true
    @test TMLE.satisfies_positivity(Ψ, frequency_table, positivity_constraint=0.15) == false
    
    Ψ = ATE(
        outcome = :toto, 
        treatment_values = (B=(case="AA", control="AC"), A=(case=1, control=1),), 
        treatment_confounders = (B = (), A = (),)
    )
    @test collect(TMLE.joint_levels(Ψ)) == [(1, "AC"), (1, "AA")]
    @test TMLE.satisfies_positivity(Ψ, frequency_table, positivity_constraint=0.1) == true
    @test TMLE.satisfies_positivity(Ψ, frequency_table, positivity_constraint=0.2) == false
    
    Ψ = AIE(
        outcome = :toto, 
        treatment_values = (B=(case="AC", control="AA"), A=(case=1, control=0),), 
        treatment_confounders = (B=(), A=()), 
    )
    @test collect(TMLE.joint_levels(Ψ)) == [
        (0, "AA") (0, "AC")  
        (1, "AA")  (1, "AC")]
    @test TMLE.satisfies_positivity(Ψ, frequency_table, positivity_constraint=1.) == false
    
    frequency_table = Dict(
        (1, "CC") => 0.125,
        (1, "AA") => 0.25,
        (0, "AA") => 0.25,
        (0, "AC") => 0.25,
        (1, "AC") => 0.125,
        (2, "AA") => 0.25
    )
    @test TMLE.satisfies_positivity(Ψ, frequency_table, positivity_constraint=0.3) == false
    @test TMLE.satisfies_positivity(Ψ, frequency_table, positivity_constraint=0.1) == true

    Ψ = AIE(
        outcome = :toto, 
        treatment_values = (B=(case="AC", control="AA"), A=(case=1, control=0), C=(control=0, case=2)), 
        treatment_confounders = (B=(), A=(), C=())
    )
    expected_joint_levels = Set([
        (1, "AC", 0),
        (0, "AC", 0),
        (1, "AA", 0),
        (0, "AA", 0),
        (1, "AC", 2),
        (0, "AC", 2),
        (1, "AA", 2),
        (0, "AA", 2)])
    @test expected_joint_levels == Set(TMLE.joint_levels(Ψ))
end

@testset "Test selectcols" begin
    dataset = DataFrame(
        A = [1, 1, 0, 1, 0, 2, 2, 1],
        B = ["AC", "CC", "AA", "AA", "AA", "AA", "AA", "AA"],
        C = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    )
    # Check columns are not copied
    selected_cols = TMLE.selectcols(dataset, [:A, :B])
    @test selected_cols.A === dataset.A
    @test selected_cols.B === dataset.B

    # Check columns are copied
    selected_cols = TMLE.selectcols(dataset, (:A, ); copycols=true)
    @test selected_cols.A !== dataset.A
    @test selected_cols.A == dataset.A

    # No column results in empty dataframe
    selected_cols = TMLE.selectcols(dataset, [])
    @test selected_cols == DataFrame(INTERCEPT=[1, 1, 1, 1, 1, 1, 1, 1])
end

@testset "supervised_learner_supports_weights" begin
    models = default_models()
    # Model that does not support weights: LogisticClassifier
    @test TMLE.supervised_learner_supports_weights(LogisticClassifier()) == false
    # Model that supports weights: LinearBinaryClassifier
    @test TMLE.supervised_learner_supports_weights(TMLE.LinearBinaryClassifier()) == true
    # simple pipelines
    @test all(TMLE.supervised_learner_supports_weights.(values(models))) == true # all default models support weights
    @test TMLE.supervised_learner_supports_weights(Pipeline(OneHotEncoder(), LogisticClassifier())) == false # does not support weights
    # Nested pipeline
    pipe = Pipeline(OneHotEncoder(), Pipeline(Standardizer(), TMLE.LinearBinaryClassifier()))
    @test TMLE.supervised_learner_supports_weights(pipe) == true # supports weights
    pipe = Pipeline(OneHotEncoder(), Pipeline(Standardizer(), LogisticClassifier()))
    @test TMLE.supervised_learner_supports_weights(pipe) == false # does not support weights
    # Test clean error
    @test_throws ArgumentError("Only learners of type `Supervised` and `SupervisedPipeline` are supported for CCW-TMLE. String is not.") TMLE.supervised_learner_supports_weights("Anything Else")
end

@testset "Test check_inputs" begin
    # Check treatment levels
    Ψ = ATE(;
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=[:W]
    )
    # The treatment levels correctly appear in the dataset
    dataset = DataFrame(Y=rand(10), T=rand(0:1, 10), W=rand(10))
    @test TMLE.check_inputs(Ψ, dataset, nothing) isa Any
    # The treatment levels do not appear in the dataset
    dataset = DataFrame(Y=rand(10), T=rand(2:3, 10), W=rand(10))
    msg = "The treatment variable T's, 'control' level: '0' in Ψ does not match any level in the dataset: [2, 3]"
    @test_throws ArgumentError(msg) TMLE.check_inputs(Ψ, dataset, nothing)

    # Check with prevalence
    prevalence = 0.1
    Ψ = CM(
        outcome = :Y, 
        treatment_values = (T=1,), 
        treatment_confounders = [:W]
    )
    ## The outcome must be binary
    dataset = DataFrame(
        Y = categorical([1, 0, 1, 0, 1, 1, 2]),
        T = categorical([1, 1, 0, 1, 0, 2, 2]),
        W = rand(7)
    )
    @test_throws ArgumentError("Outcome column must be binary when prevalence is specified.") TMLE.check_inputs(Ψ, dataset, prevalence)
    ## The number of controls must be larger than the number of cases
    dataset = DataFrame(
        Y = categorical([1, 0, 1, 0, 1, 1, 0]),
        T = categorical([1, 1, 0, 1, 0, 2, 2]),
        W = rand(7)
    )
    @test_throws ArgumentError("The dataset must contain more controls (0) than cases (1) when prevalence is provided.") TMLE.check_inputs(Ψ, dataset, prevalence)
end

@testset "Test get_fluctuation_dataset" begin
    dataset = DataFrame(
        Y = categorical([1, 0, 1, 0, 0, 0, 0, 0]),
        T = categorical([1, 1, 0, 1, 0, 2, missing, 0]),
        W = rand(8)
    )
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=[:W]
    )
    relevant_factors = TMLE.get_relevant_factors(Ψ)
    # No prevalence: missing values relevant to the estimation process are filtered
    prevalence = nothing
    fluctuation_dataset = TMLE.get_fluctuation_dataset(dataset, relevant_factors; prevalence=prevalence)
    @test fluctuation_dataset == dataset[Not([7]), :]
    # Prevalence: the surplus of controls are dropped, 2 controls per case are inferred
    prevalence = 0.1
    expected_log = (:info, "Dropping 1 control(s) to ensure equal number of controls per case (J=2). You can pre-drop these controls yourself to prevent this operation.")
    fluctuation_dataset = @test_logs expected_log TMLE.get_fluctuation_dataset(dataset, relevant_factors; prevalence=prevalence, verbosity = 1)
    @test nrow(fluctuation_dataset) == 6
    # If no missing values are present and the number of controls per case is an integer, 
    # these operations are no-ops, the dataframe will not be === because of column selection
    # but each column is ===
    dataset = DataFrame(
        Y = categorical([1, 0, 1, 0]),
        T = categorical([1, 1, 0, 1]),
        W = rand(4)
    )
    fluctuation_dataset = TMLE.get_fluctuation_dataset(dataset, relevant_factors; prevalence=prevalence)
    @test fluctuation_dataset.Y === dataset.Y
    @test fluctuation_dataset.T === dataset.T
    @test fluctuation_dataset.W === dataset.W
end

@testset "Test choose_initial_dataset" begin
    src_dataset = "src_dataset"
    fluctuation_dataset = "fluctuation_dataset"
    @test src_dataset === TMLE.choose_initial_dataset(src_dataset, fluctuation_dataset;
        train_validation_indices=nothing, 
        prevalence=nothing
    )
    @test fluctuation_dataset ===TMLE.choose_initial_dataset(src_dataset, fluctuation_dataset;
        train_validation_indices=nothing, 
        prevalence=0.1
    )
    @test fluctuation_dataset === TMLE.choose_initial_dataset(src_dataset, fluctuation_dataset;
        train_validation_indices=[], 
        prevalence=nothing
    )
    @test fluctuation_dataset === TMLE.choose_initial_dataset(src_dataset, fluctuation_dataset;
        train_validation_indices=[], 
        prevalence=0.1
    )
end

end;

true