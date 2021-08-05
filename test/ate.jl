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

# Here I illustrate the Double Robust behavior by
# misspecifying one of the models and the TMLE still converges
grid = (
    (problem=continuous_problem, 
    family=Normal(), 
    subgrid=((LinearRegressor(), ConstantClassifier(), [19, 4, 1.7, 0.5], [0.07, 0.003, 0.0006, 4.7e-5]),
             (MLJ.DeterministicConstantRegressor(), LogisticClassifier(), [64, 8.7, 3.5, 0.8], [0.33, 0.02, 0.002, 9.7e-5]))
    ),
    (problem=categorical_problem, 
    family=Bernoulli(), 
    subgrid=((LogisticClassifier(), ConstantClassifier(), [14, 4, 2, 0.4], [0.009, 0.0006, 0.0002, 6.1e-6]),
             (ConstantClassifier(), LogisticClassifier(), [11, 4.3, 1.5, 0.6], [0.007, 0.002, 0.0002, 1.2e-5]))
    )
)
rng = StableRNG(1234)
Ns = [100, 1000, 10000, 100000]
@testset "ATE TMLE Double Robustness on $(replace(string(problem_set.problem), '_' => ' '))
            - E[Y|W,T] is a $(string(typeof(y_model)))
            - p(T|W) is a $(string(typeof(t_model)))" (
    for problem_set in grid, (y_model, t_model, expected_bias_upb, expected_var_upb) in problem_set.subgrid
        tmle = ATEEstimator(
            y_model,
            t_model,
            problem_set.family
            )
        abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            problem_set.problem,
            rng,
            Ns
            )
        # Check the bias and variance are converging to 0
        @test all(abs_mean_rel_errors .< expected_bias_upb)
        @test all(abs_vars .< expected_var_upb)
end);


end

true