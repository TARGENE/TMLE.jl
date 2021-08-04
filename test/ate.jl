module TestATE

include("utils.jl")

using TMLE
using Random
using Test
using Distributions
using MLJ
using StableRNGs


LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity = 0


function categorical_problem(rng;n=100)
    p_w() = 0.3
    pa_given_w(w) = 1 ./ (1 .+ exp.(-0.5w .+ 1))
    py_given_aw(a, w) = 1 ./ (1 .+ exp.(2w .- 3a .+ 1))
    # Sample from dataset
    Unif = Uniform(0, 1)
    w = rand(rng, Unif, n) .< p_w()
    t = rand(rng, Unif, n) .< pa_given_w(w)
    y = rand(rng, Unif, n) .< py_given_aw(t, w)
    # Convert to dataframe to respect the Tables.jl
    # and convert types
    W = (W=convert(Array{Float64}, w),)
    t = categorical(t)
    y = categorical(y)
    # Compute the theoretical ATE
    ATE₁ = py_given_aw(1, 1)*p_w() + (1-p_w())*py_given_aw(1, 0)
    ATE₀ = py_given_aw(0, 1)*p_w() + (1-p_w())*py_given_aw(0, 0)
    ATE = ATE₁ - ATE₀
    
    return t, W, y, ATE
end

"""
From https://www.degruyter.com/document/doi/10.2202/1557-4679.1043/html
The theoretical ATE is 1
"""
function continuous_problem(rng;n=100)
    Unif = Uniform(0, 1)
    W = float(rand(rng, Bernoulli(0.5), n, 3))
    t = rand(rng, Unif, n) .< TMLE.expit(0.5W[:, 1] + 1.5W[:, 2] - W[:,3])
    y = t + 2W[:, 1] + 3W[:, 2] - 4W[:, 3] + rand(rng, Normal(0,1), n)
    # Type coercion
    W = MLJ.table(W)
    t = categorical(t)
    return t, W, y, 1
end

@testset "Test target/treatment types" begin
    ate_estimator = ATEEstimator(
            LogisticClassifier(),
            LogisticClassifier(),
            Bernoulli()
            )
    W = (col1=[1, 1, 0], col2=[0, 0, 1])

    t = categorical([false, true, true])
    y =  categorical(["a", "b", "c"])
    @test_throws MethodError MLJ.fit(ate_estimator, 0, t, W, y)

    t = categorical([1, 2, 2])
    y =  categorical([false, true, true])
    @test_throws MethodError MLJ.fit(ate_estimator, 0, t, W, y)

end


@testset "Testing the various intermediate functions" begin
    # Let's process the different functions in order of call by fit! 
    # and check intermediate and final outputs
    t_target = categorical([false, true, true, false, true, true, false])
    y = categorical([true, true, false, false, false, true, true])
    W = (W1=categorical([true, true, false, true, true, true, true]),)
    
    # Converting to NamedTuples
    T = (t=float(t_target),)
    X = merge(T, W)

    # The following dummy model will have the predictive distribution
    # [false, true] => [0.42857142857142855 0.5714285714285714]
    # for both the treatment and the target
    dummy_model = @pipeline OneHotEncoder() ConstantClassifier()
    # Testing compute_offset function
    target_expectation_mach = machine(dummy_model, X, y)
    MLJ.fit!(target_expectation_mach, verbosity=0)
    offset = TMLE.compute_offset(target_expectation_mach, X)
    @test offset isa Vector{Float64}
    @test all(isapprox.(offset, 0.28768, atol=1e-5))

    # Testing compute_covariate function
    # The likelihood function is given bu the previous distribution
    treatment_likelihood_mach = machine(dummy_model, W, t_target)
    MLJ.fit!(treatment_likelihood_mach, verbosity=0)
    covariate = TMLE.compute_covariate(treatment_likelihood_mach, W, T, t_target)
    @test covariate isa Vector{Float64}
    @test covariate ≈ [-2.33, 1.75, 1.75, -2.33, 1.75, 1.75, -2.33] atol=1e-2

    # Testing compute_fluctuation function
    fluctuator = TMLE.glm(reshape(covariate, :, 1), y, Bernoulli(); offset=offset)
    fluct = TMLE.compute_fluctuation(fluctuator, 
                                                 target_expectation_mach,
                                                 treatment_likelihood_mach,
                                                 W, 
                                                 t_target)
    # Not sure how to test that the operation is correct
    @test fluct isa Vector{Float64}
    @test all(isapprox.(fluct, [0.6644,
                                0.4977,
                                0.4977,
                                0.6644,
                                0.4977,
                                0.4977,
                                0.6644],
                    atol = 1e-4))
end


@testset "Test ATE TMLE fit asymptotic behavior on binary target" begin
    ate_estimator = ATEEstimator(
        LogisticClassifier(),
        LogisticClassifier(),
        Bernoulli()
                        )
    
    abs_mean_errors, abs_var_errors = asymptotics(ate_estimator, categorical_problem)

    # Check the average and variances decrease with n 
    @test abs_mean_errors == sort(abs_mean_errors, rev=true)
    @test abs_var_errors == sort(abs_var_errors, rev=true)
    # Check the error's close to the target for large samples
    @test all(abs_mean_errors .< [0.2, 0.02, 0.006, 0.003])
end

@testset "Test ATE TMLE fit asymptotic behavior on continuous target" begin

    ate_estimator = ATEEstimator(
        LinearRegressor(),
        LogisticClassifier(),
        Normal()
    )
    
    abs_mean_errors, abs_var_errors = asymptotics(ate_estimator, continuous_problem)

    # Check the average and variances decrease with n 
    @test abs_mean_errors == sort(abs_mean_errors, rev=true)
    @test abs_var_errors == sort(abs_var_errors, rev=true)
    # Check the error's close to the target for large samples
    @test all(abs_mean_errors .< [0.2, 0.06, 0.02, 0.006])
end

@testset "Non regression test of ATE for continuous target" begin
    ate_estimator = ATEEstimator(
        LinearRegressor(),
        LogisticClassifier(),
        Normal()
                        )
    n = 1000
    estimates = []
    for i in 1:1000
        rng = StableRNG(i)
        t, W, y, ATE = continuous_problem(rng; n=n)
        fitresult, _, _ = MLJ.fit(ate_estimator, 0, t, W, y)
        push!(estimates, fitresult.estimate)
    end
    @test mean(estimates) ≈ 1.0000 atol=1e-4
    @test var(estimates) ≈ 0.0063 atol=1e-4
end


end

true