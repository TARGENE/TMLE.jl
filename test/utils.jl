module TestUtils

using Test
using TMLE
using MLJBase
using CategoricalArrays
using MLJLinearModels
using MLJModels
using DataFrames

@testset "Test get_confounders_subset" begin
    # Test with a tuple of confounders and no subset
    confounders = (:W₁, :W₂, :W₃)
    @test TMLE.get_confounders_subset(confounders, nothing) == confounders

    # Test with a tuple of confounders and a subset
    subset = (:W₁, :W₂)
    @test TMLE.get_confounders_subset(confounders, subset) == (:W₁, :W₂)

    # This signature is for collaborative strategies, by default return nothing
    @test TMLE.get_confounders_subset(nothing) === nothing
end

@testset "Test nrows and selectrows for Riesz Learning" begin
    T = DataFrame(A=1:10, B=11:20, C=21:30)
    W = DataFrame(D=31:40, E=41:50)
    X = (T, W)
    @test MLJBase.nrows(X) == 10
    @test MLJBase.selectrows(X, 1:5) == (T[1:5, :], W[1:5, :])
    @test MLJBase.selectrows(X, [6, 7, 9]) == (T[[6, 7, 9], :], W[[6, 7, 9], :])
end

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

end;

true