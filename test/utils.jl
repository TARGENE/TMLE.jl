module TestUtils

using Test
using Tables
using TableOperations
using TMLE
using MLJBase
using StableRNGs
using Distributions
using CategoricalArrays
using MLJGLMInterface: LinearBinaryClassifier
using MLJLinearModels
using MLJModels

function test_params_match(parameters, expected_params)
    for (param, expected_param) in zip(parameters, expected_params)
        @test typeof(param) == typeof(expected_param)
        @test param.target == expected_param.target
        @test param.treatment == expected_param.treatment
        @test param.confounders == expected_param.confounders
        @test param.covariates == expected_param.covariates
    end
end

@testset "Test joint_treatment" begin
    T = (
        T₁ = categorical([0, 1, 2, 1]), 
        T₂ = categorical(["a", "b", "a", "b"])
    )
    jointT = TMLE.joint_treatment(T)
    @test jointT == categorical(["0_&_a", "1_&_b", "2_&_a", "1_&_b"])

    T = (
        T₁ = categorical([0, 1, 2, 1]), 
    )
    jointT = TMLE.joint_treatment(T)
    @test jointT == categorical(["0", "1", "2", "1"])
end


@testset "Test indicator_fns" begin
    # Conditional Mean
    Ψ = CM(
        target=:y, 
        treatment=(T₁="A", T₂=1),
        confounders=[:W]
    )

    @test TMLE.indicator_fns(Ψ, TMLE.joint_name) == Dict{String, Float64}("A_&_1" => 1)
    # ATE
    Ψ = ATE(
        target=:y, 
        treatment=(T₁=(case="A", control="B"), T₂=(control=0, case=1)),
        confounders=[:W]
    )
    @test TMLE.indicator_fns(Ψ, TMLE.joint_name) == Dict{String, Float64}(
        "A_&_1" => 1,
        "B_&_0" => -1
    )
    # 2-points IATE
    Ψ = IATE(
        target=:y, 
        treatment=(T₁=(case="A", control="B"), T₂=(case=1, control=0)),
        confounders=[:W]
    )
    @test TMLE.indicator_fns(Ψ, TMLE.joint_name) == Dict{String, Float64}(
        "A_&_1" => 1,
        "A_&_0" => -1,
        "B_&_1" => -1,
        "B_&_0" => 1
    )
    # 3-points IATE
    Ψ = IATE(
        target=:y, 
        treatment=(T₁=(case="A", control="B"), T₂=(case=1, control=0), T₃=(control="D", case="C")),
        confounders=[:W]
    )
    @test TMLE.indicator_fns(Ψ, TMLE.joint_name) == Dict{String, Float64}(
        "A_&_1_&_D" => -1,
        "A_&_1_&_C" => 1,
        "B_&_0_&_D" => -1,
        "B_&_0_&_C" => 1,
        "B_&_1_&_C" => -1,
        "A_&_0_&_D" => 1,
        "B_&_1_&_D" => 1,
        "A_&_0_&_C" => -1
    )
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
    @test expectation == repeat([proba], n)

    # Probabilistic Regressor
    y = rand(n)
    mach = machine(ConstantRegressor(), X, y)
    fit!(mach; verbosity=0)
    ŷ = MLJBase.predict(mach)
    expectation = TMLE.expected_value(ŷ)
    @test expectation ≈ repeat([mean(y)], n) atol=1e-10

    # Deterministic Regressor
    mach = machine(LinearRegressor(), X, y)
    fit!(mach; verbosity=0)
    ŷ = MLJBase.predict(mach)
    expectation = TMLE.expected_value(ŷ)
    @test expectation == ŷ
end

@testset "Test indicator_values" begin
    indicators = Dict(
        "b_&_c_&_true"  => -1,
        "a_&_c_&_true"  => 1,
        "b_&_d_&_false" => -1,
        "b_&_c_&_false" => 1,
        "a_&_d_&_true"  => -1,
        "a_&_c_&_false" => -1,
        "a_&_d_&_false" => 1,
        "b_&_d_&_true"  => 1 
    )
    jointT = categorical([
        "b_&_c_&_true", "a_&_c_&_true", "b_&_d_&_false",
        "b_&_c_&_false", "a_&_d_&_true", "a_&_c_&_false",
        "a_&_d_&_false", "b_&_d_&_true", "q_&_d_&_false"
    ])
    # The las combination does not appear in the indicators
    @test TMLE.indicator_values(indicators, jointT) ==
        [-1, 1, -1, 1, -1, -1, 1, 1, 0]
end


@testset "Test counterfactualTreatment" begin
    vals = (true, "a")
    T = (
        t₁ = categorical([true, false, false]),
        t₂ = categorical(["a", "a", "c"])
    )
    cfT = TMLE.counterfactualTreatment(vals, T)
    @test cfT == (
        t₁ = categorical([true, true, true]),
        t₂ = categorical(["a", "a", "a"])
    )
end

@testset "Test compute_covariate" begin
    # First case: 1 categorical variable
    # Using a trivial classifier
    # that outputs the proportions of of the classes
    jointT = categorical(["a", "b", "c", "a", "a", "b", "a"])
    W = MLJBase.table(rand(7, 3))

    Gmach = machine(ConstantClassifier(), 
                    W,
                    jointT)
    fit!(Gmach, verbosity=0)

    Ψ = ATE(
        target =:y, 
        treatment=(t₁=(case="a", control="b"),),
        confounders = [:x1, :x2, :x3]
    )
    indicator_fns = TMLE.indicator_fns(Ψ, TMLE.joint_name)

    cov = TMLE.compute_covariate(jointT, W, Gmach, indicator_fns)
    @test cov == [1.75,
                 -3.5,
                 0.0,
                 1.75,
                 1.75,
                 -3.5,
                 1.75]

    # Second case: 2 binary variables
    # Using a trivial classifier
    # that outputs the proportions of of the classes
    jointT = categorical(["1_&_1", "0_&_1", "0_&_1", "1_&_1", "1_&_1", "1_&_0", "0_&_0"])
    W = MLJBase.table(rand(7, 3))

    Gmach = machine(ConstantClassifier(), 
                    W, 
                    jointT)
    fit!(Gmach, verbosity=0)
    Ψ = IATE(
        target =:y, 
        treatment=(t₁=(case=1, control=0), t₂=(case=1, control=0)),
        confounders = [:x1, :x2, :x3]
    )
    indicator_fns = TMLE.indicator_fns(Ψ, TMLE.joint_name)

    cov = TMLE.compute_covariate(jointT, W, Gmach, indicator_fns)
    @test cov == [2.3333333333333335,
                 -3.5,
                 -3.5,
                 2.3333333333333335,
                 2.3333333333333335,
                 -7.0,
                 7.0]

    # Third case: 3 mixed categorical variables
    # Using a trivial classifier
    # that outputs the proportions of of the classes
    jointT = categorical(
        ["a_&_3_&_true", "a_&_2_&_false", "b_&_1_&_true", 
        "b_&_1_&_false", "c_&_2_&_false", "b_&_2_&_false", 
        "b_&_2_&_false"])
    W = MLJBase.table(rand(7, 3))

    Gmach = machine(ConstantClassifier(), 
                    W, 
                    jointT)
    fit!(Gmach, verbosity=0)
    Ψ = IATE(
        target =:y, 
        treatment=(t₁=(case="a", control="b"), 
                   t₂=(case=1, control=2), 
                   t₃=(case=true, control=false)),
        confounders = [:x1, :x2, :x3]
    )
    indicator_fns = TMLE.indicator_fns(Ψ, TMLE.joint_name)

    cov = TMLE.compute_covariate(jointT, W, Gmach, indicator_fns)
    @test cov == [0,
                  7.0,
                 -7,
                  7,
                  0,
                 -3.5,
                 -3.5]
end

@testset "Test compute_offset" begin
    n = 10
    X = rand(n, 3)

    # When Y is binary
    y = categorical([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    mach = machine(ConstantClassifier(), MLJBase.table(X), y)
    fit!(mach, verbosity=0)
    ŷ = MLJBase.predict(mach)
    # Should be equal to logit(Ê[Y|X])= logit(4/10) = -0.4054651081081643
    @test TMLE.compute_offset(ŷ) == repeat([-0.4054651081081643], n)

    # When Y is continuous
    y = [1., 2., 3, 4, 5, 6, 7, 8, 9, 10]
    mach = machine(MLJModels.DeterministicConstantRegressor(), MLJBase.table(X), y)
    fit!(mach, verbosity=0)
    ŷ = predict(mach)
    # Should be equal to Ê[Y|X] = 5.5
    @test TMLE.compute_offset(ŷ) == repeat([5.5], n)
    
end


@testset "Test parameters_from_yaml" begin
    # No covariate
    param_file = joinpath("data", "parameters.yaml")
    parameters = parameters_from_yaml(param_file)
    expected_params =[
        IATE(;
            target=:Y1, 
            treatment=(T2 = (case = 1, control = 0), T1 = (case = 1, control = 0)), 
            confounders=[:W1], 
            covariates=Symbol[]
        ),
        IATE(;
            target=:Y2, 
            treatment=(T2 = (case = 1, control = 0), T1 = (case = 1, control = 0)), 
            confounders=[:W1], 
            covariates=Symbol[]
        ),
        ATE(;
            target=:Y1, 
            treatment=(T2 = (case = 1, control = 0), T1 = (case = 1, control = 0)), 
            confounders=[:W1], 
            covariates=Symbol[]
        ),
        ATE(;
            target=:Y2, 
            treatment=(T2 = (case = 1, control = 0), T1 = (case = 1, control = 0)), 
            confounders=[:W1], 
            covariates=Symbol[]
        ),
        CM(;target=:Y1, treatment=(T2 = 0, T1 = 1), confounders=[:W1], covariates=Symbol[]),
        CM(;target=:Y2, treatment=(T2 = 0, T1 = 1), confounders=[:W1], covariates=Symbol[])
    ]
    test_params_match(parameters, expected_params)
    # With covariate
    param_file = joinpath("data", "parameters_with_covariates.yaml")
    parameters = parameters_from_yaml(param_file)
    expected_params = [
        IATE(;
            target=:Y1, 
            treatment=(T2 = (case = "AC", control = "CC"), T1 = (case = 2, control = 1)), 
            confounders=[:W1], 
            covariates=[:C1])
        ATE(;
            target=:Y1, 
            treatment=(T2 = (case = "AC", control = "CC"), T1 = (case = 2, control = 0)), 
            confounders=[:W1], 
            covariates=[:C1])
        CM(target=:Y1, treatment=(T2 = 0, T1 = 0), confounders=[:W1], covariates=[:C1])
    ]
    test_params_match(parameters, expected_params)

end

@testset "Test fluctuation_input" begin
    X = TMLE.fluctuation_input([1., 2.], [1., 2])
    @test X.covariate isa Vector{Float64}
    @test X.offset isa Vector{Float64}

    X = TMLE.fluctuation_input([1., 2.], [1.f0, 2.f0])
    @test X.covariate isa Vector{Float64}
    @test X.offset isa Vector{Float64}

    X = TMLE.fluctuation_input([1.f0, 2.f0], [1., 2.])
    @test X.covariate isa Vector{Float32}
    @test X.offset isa Vector{Float32}

end

end;

true