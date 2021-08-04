module TestInteractionATE

include("utils.jl")

using Test
using TMLE
using MLJ
using Distributions
using Random
using StableRNGs
using Tables

mutable struct InteractionTransformer <: Static end
    
function MLJ.transform(a::InteractionTransformer, _, X)
    Xmatrix = MLJ.matrix(X)
    nrows, ncols = size(Xmatrix)
    ninter = Int(ncols*(ncols-1)/2)
    Xinteracts = Matrix{Float64}(undef, nrows, ninter)
    i = 0
    for col₁ in 1:(ncols-1)
        for col₂ in (col₁+1):ncols
            i += 1
            Xinteracts[:, i] = Xmatrix[:, col₁] .* Xmatrix[:, col₂]
        end
    end
    return MLJ.table(hcat(Xmatrix, Xinteracts))
end

LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity = 0


function categorical_problem(rng;n)
    p_w() = 0.4
    pt1_given_w(w) = 1 ./ (1 .+ exp.(-0.5w .+ 1))
    pt2_given_w(w) = 1 ./ (1 .+ exp.(0.5w .+ 1))
    py_given_t1t2w(t1, t2, w) = 1 ./ (1 .+ exp.(2w .- 3t1 .+ 3t2 .+ 1))
    # Sample from dataset
    Unif = Uniform(0, 1)
    w = rand(rng, Unif, n) .< p_w()
    t₁ = rand(rng, Unif, n) .< pt1_given_w(w)
    t₂ = rand(rng, Unif, n) .< pt2_given_w(w)
    y = rand(rng, Unif, n) .< py_given_t1t2w(t₁, t₂, w)
    # Format dataset and convert types
    W = (W=convert(Array{Float64}, w),)
    T = (t₁ = categorical(t₁), t₂ = categorical(t₂))
    y = categorical(y)
    # Compute the theoretical ATE
    ATE₁ = (py_given_t1t2w(1, 1, 1) - py_given_t1t2w(1, 0, 1) - py_given_t1t2w(0, 1, 1) + py_given_t1t2w(0, 0, 1))*p_w()
    ATE₀ = (py_given_t1t2w(1, 1, 0) - py_given_t1t2w(1, 0, 0) - py_given_t1t2w(0, 1, 0) + py_given_t1t2w(0, 0, 0))*(1 - p_w())
    ATE = ATE₁ + ATE₀
    
    return T, W, y, ATE
end


function continuous_problem(rng;n)
    Unif = Uniform(0, 1)
    W = float(rand(rng, Bernoulli(0.5), n, 3))
    t₁ = rand(rng, Unif, n) .< TMLE.expit(0.5W[:, 1] + 1.5W[:, 2] - W[:,3])
    t₂ = rand(rng, Unif, n) .< TMLE.expit(-0.5W[:, 1] + 2.5W[:, 2] + W[:,3])
    y = t₁ + 2t₂ -3(t₁ .* t₂) + 2W[:, 1] + 3W[:, 2] - 4W[:, 3] + rand(rng, Normal(0,1), n)
    # Data formatting and Type coercion
    W = MLJ.table(W)
    T = (t₁ = categorical(t₁), t₂ = categorical(t₂))
    return T, W, y, -3
end


@testset "Test Helper functions" begin
    # Test tomultivariate function
    # The mapping is fixed and harcoded
    T = (t1 = categorical([true, false, false, true, true, true, false]),
         t2 = categorical([true, true, true, true, true, false, false]))
    W = MLJ.table(rand(7, 3))
    t_target = TMLE.tomultivariate(T)
    @test t_target == categorical([1, 3, 3, 1, 1, 2, 4])
    # Test compute_covariate
    
    t_likelihood_estimate = machine(ConstantClassifier(), W, t_target)
    fit!(t_likelihood_estimate)
    
    Tnames = Tables.columnnames(T)
    T = NamedTuple{Tnames}([float(Tables.getcolumn(T, colname)) for colname in Tnames])
    cov = TMLE.compute_covariate(t_likelihood_estimate, W, T, t_target)
    @test cov == [2.3333333333333335,
                 -3.5,
                 -3.5,
                 2.3333333333333335,
                 2.3333333333333335,
                 -7.0,
                 7.0]

end


@testset "Binary Target Interaction ATE Asymptotic Behavior" begin
    interaction_estimator = InteractionATEEstimator(
        LogisticClassifier(),
        LogisticClassifier(),
        Bernoulli()
        )

    abs_mean_errors, abs_var_errors = asymptotics(interaction_estimator, categorical_problem)

    # Check the average and variances decrease with n 
    @test abs_mean_errors == sort(abs_mean_errors, rev=true)
    @test abs_var_errors == sort(abs_var_errors, rev=true)
    # Check the error's close to the target for large samples
    @test all(abs_mean_errors .< [0.51, 0.083, 0.03, 0.006])
    @test all(abs_var_errors .< [0.09, 0.005, 0.0003, 9.2e-6])
end


@testset "Continuous Target Interaction ATE Asymptotic Behavior" begin
    # The complexity of the model can be captured by neither 
    # a Linear regression of a Logistic regression
    # The estimation will fail if we don't provide at least one good estimator 
    # I add interaction terms for the linear regressor
    cond_expectation_estimator = @pipeline InteractionTransformer LinearRegressor
    interaction_estimator = InteractionATEEstimator(
        cond_expectation_estimator,
        LogisticClassifier(),
        Normal()
        )

    abs_mean_errors, abs_var_errors = asymptotics(interaction_estimator, continuous_problem)

    # Check the average and variances decrease with n 
    @test abs_mean_errors == sort(abs_mean_errors, rev=true)
    @test abs_var_errors == sort(abs_var_errors, rev=true)
    # Check the error's close to the target for large samples
    @test all(abs_mean_errors .< [0.6, 0.2, 0.06, 0.02])
    @test all(abs_var_errors .< [0.08, 0.008, 0.002, 7.2e-5])
end

end

true